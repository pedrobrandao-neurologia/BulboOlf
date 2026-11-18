"""
Executa o pipeline ORIGINAL do repositório Deep-MI/olf-bulb-segmentation (mesmo código de
pré/pós-processamento: conform, reamostragem, localização, recorte, clean_seg, estatísticas),
trocando apenas o motor de inferência (PyTorch -> ONNX Runtime) por modelos ONNX equivalentes.

uso: python3 py_ref_driver.py REPO IMG OUTDIR SUBID MODEL(1-5) MAP(real|standin) ONNXDIR
"""
import sys, os, json, types
import numpy as np

REPO, IMG, OUTDIR, SUBID, MODEL, MAPMODE, ONNXDIR = sys.argv[1:8]
sys.path.insert(0, REPO)
os.chdir(REPO)
import onnxruntime as ort
import run_pipeline as rp
from models.OBNet import OBNet
import torch

_sessions = {}

def onnx_for(pkl_path):
    base = os.path.basename(pkl_path).replace('.pkl', '')
    if MAPMODE == 'real':
        return os.path.join(ONNXDIR, base + '.onnx')
    # stand-ins: um por plano / tipo
    kind = 'loc' if 'FastSurferCNN_' in base and 'Att' not in base else 'seg'
    plane = [p for p in ('axial', 'coronal', 'sagittal') if p in base][0]
    return os.path.join(ONNXDIR, f'standin_{kind}_{plane}.onnx')

def sess_for(path):
    if path not in _sessions:
        so = ort.SessionOptions(); so.intra_op_num_threads = 2
        _sessions[path] = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
    return _sessions[path]

_orig_load = OBNet.load_weights
def load_weights(self, current_model):
    self._current_pkl = current_model
    return _orig_load(self, current_model)
OBNet.load_weights = load_weights

def predict(self, img, batch_size, model):
    # replica de OBNet.predict (ToTensorTest: /255, clip 0..1, HWC->CHW) usando ONNX Runtime
    sess = sess_for(onnx_for(self._current_pkl))
    x = np.clip(img.astype(np.float32) / 255.0, 0.0, 1.0).transpose(0, 3, 1, 2).copy()
    outs = []
    for s in range(0, x.shape[0], batch_size):
        outs.append(sess.run(None, {'input': x[s:s + batch_size]})[0])
        print('batch', s // batch_size + 1, 'of', -(-x.shape[0] // batch_size), flush=True)
    logits = np.concatenate(outs, 0)
    return logits.transpose(0, 2, 3, 1)
OBNet.predict = predict

# captura de valores intermediários para comparação com a versão JavaScript
_orig_loc = OBNet.run_localization
def run_localization(self, t2_img):
    cm, cm_logits, resampled = _orig_loc(self, t2_img)
    np.save(os.path.join(OUTDIR, SUBID, 'loc_p1.npy'), cm_logits.astype(np.float32))
    json.dump({'loc_cm': None if cm is None else [int(v) for v in cm],
               'loc_affine': resampled.affine.tolist(), 'loc_shape': list(resampled.shape),
               't2_affine': t2_img.affine.tolist(), 't2_shape': list(t2_img.shape)},
              open(os.path.join(OUTDIR, SUBID, 'loc_info.json'), 'w'))
    return cm, cm_logits, resampled
OBNet.run_localization = run_localization

sys.argv = ['run_pipeline.py', '-in', IMG, '-out', OUTDIR, '-sid', SUBID, '-ncuda', '-batch', '4',
            '-model', MODEL, '-logits']
os.makedirs(os.path.join(OUTDIR, SUBID), exist_ok=True)
args, FLAGS = rp.option_parse()
rp.ob_pipeline(args, FLAGS)
