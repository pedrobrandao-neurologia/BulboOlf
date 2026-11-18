import torch, torch.nn as nn, sys, os
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
for kind, size in (('loc', 192), ('seg', 96)):
    for i, plane in enumerate(('axial', 'coronal', 'sagittal')):
        g = torch.Generator().manual_seed(100 + i + (10 if kind == 'seg' else 0))
        conv = nn.Conv2d(3, 2, 5, padding=2)
        with torch.no_grad():
            w = torch.zeros(2, 3, 5, 5)
            w[1, 1] = 1.0 / 25 * 3.0                       # média local do canal central
            w += 0.15 * (torch.rand(2, 3, 5, 5, generator=g) - 0.5)  # assimetria (detecta erros de orientação)
            conv.weight.copy_(w)
            conv.bias.copy_(torch.tensor([0.0, -1.2 - 0.1 * i]))
        m = nn.Sequential(conv).eval()
        x = torch.rand(2, 3, size, size)
        torch.onnx.export(m, (x,), f'{out}/standin_{kind}_{plane}.onnx', input_names=['input'], output_names=['logits'],
                          dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}}, opset_version=17, dynamo=False)
print('ok')
