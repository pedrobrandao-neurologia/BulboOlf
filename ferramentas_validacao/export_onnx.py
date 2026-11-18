"""
Exporta as redes do pipeline Deep-MI/olf-bulb-segmentation (Estrada et al., NeuroImage 2021)
para ONNX, com substituições numericamente equivalentes e exportáveis:
  * MaxPool2d(return_indices) + MaxUnpool2d  ->  argmax por janela 2x2 (1a ocorrência, ordem linha-coluna)
  * Self_Attn N x N                          ->  atenção calculada em blocos de linhas (mesmo resultado,
                                                 menos memória no navegador)
Valida cada modelo exportado contra o modelo PyTorch original.
"""
import sys, os, types, json, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

REPO = sys.argv[1]
OUT = sys.argv[2]
sys.path.insert(0, REPO)
from models.FastSurferCNN import FastSurferCNN
from models.AttFastSurferCNN import AttFastSurferCNN, Self_Attn
import yaml

torch.set_grad_enabled(False)
os.makedirs(OUT, exist_ok=True)

SEG_PARAMS = {'num_channels': 3, 'num_filters': 64, 'kernel_h': 3, 'kernel_w': 3, 'stride_conv': 1,
              'pool': 2, 'stride_pool': 2, 'num_classes': 2, 'kernel_c': 1, 'kernel_d': 3, 'dilation': 1}
LOC_PARAMS = {'num_channels': 3, 'num_filters': 64, 'kernel_h': 5, 'kernel_w': 5, 'stride_conv': 1,
              'pool': 2, 'stride_pool': 2, 'num_classes': 2, 'kernel_c': 1, 'kernel_d': 1, 'dilation': 1}


def load_state(path):
    st = torch.load(path, map_location='cpu', weights_only=True)['model_state_dict']
    return {(k[7:] if k.startswith('module.') else k): v for k, v in st.items()}


class WinPool(nn.Module):
    """MaxPool 2x2/2 que devolve o índice (0..3) da 1a ocorrência do máximo em cada janela."""
    def forward(self, x):
        n, c, h, w = x.shape
        xr = x.reshape(n, c, h // 2, 2, w // 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(n, c, h // 2, w // 2, 4)
        vals, idx = torch.max(xr, dim=-1)
        return vals, idx


class WinUnpool(nn.Module):
    """Inverso exato do MaxUnpool2d para os índices de WinPool."""
    def forward(self, x, idx):
        n, c, h2, w2 = x.shape
        ar = torch.arange(4, dtype=idx.dtype).view(1, 1, 1, 1, 4)
        onehot = (idx.unsqueeze(-1) == ar).to(x.dtype)
        out = x.unsqueeze(-1) * onehot
        return out.reshape(n, c, h2, w2, 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(n, c, h2 * 2, w2 * 2)


def chunked_attn_forward(self, x):
    b, c, w, h = x.size()
    n = w * h
    q = self.query_conv(x).view(b, -1, n).permute(0, 2, 1)
    k = self.key_conv(x).view(b, -1, n)
    v = self.value_conv(x).view(b, -1, n)
    chunk = 2304
    if n <= chunk:
        att = torch.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, att.permute(0, 2, 1))
    else:
        outs = []
        for s in range(0, n, chunk):
            att = torch.softmax(torch.bmm(q[:, s:s + chunk, :], k), dim=-1)
            outs.append(torch.bmm(v, att.permute(0, 2, 1)))
        out = torch.cat(outs, dim=2)
    out = out.view(b, c, w, h)
    return self.gamma * out + x


def make_exportable(model):
    for name, m in model.named_modules():
        if hasattr(m, 'maxpool') and isinstance(m.maxpool, nn.MaxPool2d):
            m.maxpool = WinPool()
        if hasattr(m, 'unpool') and isinstance(m.unpool, nn.MaxUnpool2d):
            m.unpool = WinUnpool()
        if isinstance(m, Self_Attn):
            m.forward = types.MethodType(chunked_attn_forward, m)
    return model


def build(arc, params, path):
    M = FastSurferCNN if arc == 'loc' else AttFastSurferCNN
    m = M(dict(params)); m.load_state_dict(load_state(path)); m.eval()
    return m


def realistic_input(size, n, seed=0):
    """Fatias '3 canais' com estrutura (gradientes + blocos + ruído) em [0,1], incluindo fundo zero e empates."""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(n, 3, size, size)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing='ij')
    for i in range(n):
        blob = torch.exp(-((xx - 0.2 * i / n) ** 2 + yy ** 2) / 0.3)
        base = 0.6 * blob + 0.2 * torch.rand(size, size, generator=g)
        base[: size // 6] = 0  # fundo constante (gera empates no maxpool)
        for ch in range(3):
            x[i, ch] = (base + 0.02 * ch).clamp(0, 1)
    return x


def main():
    import onnxruntime as ort
    seg_cfg = yaml.safe_load(open(os.path.join(REPO, 'SegModels/AttFastSurferCNN/AttFastSurferCNN_weights.yml')))
    loc_cfg = yaml.safe_load(open(os.path.join(REPO, 'LocModels/FastSurferCNN/FastSurferCNN_weights.yml')))
    jobs = [('loc', LOC_PARAMS, os.path.join(REPO, 'LocModels/FastSurferCNN', f), 192) for f in loc_cfg.values()]
    jobs += [('seg', SEG_PARAMS, os.path.join(REPO, 'SegModels/AttFastSurferCNN', f), 96) for f in seg_cfg.values()]
    only = os.environ.get('ONLY')
    report = []
    for arc, params, path, size in jobs:
        name = os.path.basename(path).replace('.pkl', '')
        if only and only not in name:
            continue
        ref = build(arc, params, path)
        exp = make_exportable(build(arc, params, path))
        x = realistic_input(size, 2)
        t0 = time.time()
        y_ref = ref(x)
        y_exp = exp(x)
        d_torch = (y_ref - y_exp).abs().max().item()
        onnx_path = os.path.join(OUT, name + '.onnx')
        torch.onnx.export(exp, (x,), onnx_path, input_names=['input'], output_names=['logits'],
                          dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
                          opset_version=17, dynamo=False, do_constant_folding=True)
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        y_onnx = sess.run(None, {'input': x.numpy()})[0]
        d_onnx = float(np.abs(y_onnx - y_ref.numpy()).max())
        agree = float(((y_onnx.argmax(1)) == (y_ref.numpy().argmax(1))).mean())
        gammas = [float(m.gamma.item()) for m in ref.modules() if isinstance(m, Self_Attn)]
        rec = dict(model=name, arc=arc, size=size, max_abs_diff_torch=d_torch, max_abs_diff_onnx=d_onnx,
                   argmax_agreement=agree, logit_range=[float(y_ref.min()), float(y_ref.max())],
                   gammas=gammas, mb=round(os.path.getsize(onnx_path) / 1e6, 2), secs=round(time.time() - t0, 1))
        print(json.dumps(rec), flush=True)
        report.append(rec)
    json.dump(report, open(os.path.join(OUT, 'export_report.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
