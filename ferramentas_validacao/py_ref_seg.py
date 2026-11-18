"""Segmentação de referência com as funções ORIGINAIS do repositório (conform, resample_to_output, OBNet.run_segmentation,
clean_seg, calculate_stats), recortando na coordenada encontrada pela localização automática (run anterior)."""
import sys, os, json, types, numpy as np
REPO, IMG, OUT, MODEL, ONNXDIR, LOCJSON = sys.argv[1:7]
sys.path.insert(0, REPO); os.chdir(REPO)
import nibabel as nib, nibabel.processing, onnxruntime as ort
import run_pipeline as rp
from utils import conform, stats
from utils.image_utils import clean_seg
from models.OBNet import OBNet
cur = {'path': None, 'sess': None}
def sess_for(p):
    if cur['path'] != p:
        cur['sess'] = None
        so = ort.SessionOptions(); so.intra_op_num_threads = 2; so.enable_cpu_mem_arena = False
        cur['sess'] = ort.InferenceSession(p, so, providers=['CPUExecutionProvider']); cur['path'] = p
    return cur['sess']
_ol = OBNet.load_weights
def lw(self, m): self._pkl = m; return _ol(self, m)
OBNet.load_weights = lw
def predict(self, img, batch_size, model):
    s = sess_for(os.path.join(ONNXDIR, os.path.basename(self._pkl).replace('.pkl', '.onnx')))
    x = np.clip(img.astype(np.float32) / 255.0, 0.0, 1.0).transpose(0, 3, 1, 2).copy()
    out = np.concatenate([s.run(None, {'input': x[i:i + batch_size]})[0] for i in range(0, len(x), batch_size)], 0)
    return out.transpose(0, 2, 3, 1)
if os.environ.get("TORCH") != "1": OBNet.predict = predict
sys.argv = ['run_pipeline.py', '-in', IMG, '-out', OUT, '-sid', 'chris', '-ncuda', '-batch', '4', '-model', MODEL]
args, FLAGS = rp.option_parse()
import logging; logger = logging.getLogger('x'); logger.addHandler(logging.StreamHandler()); logger.setLevel(logging.INFO)
t2 = conform.conform(nib.load(IMG), FLAGS, logger)
t2 = nibabel.processing.resample_to_output(t2, FLAGS['spacing'], order=1)
L = json.load(open(LOCJSON))
loc_aff = np.array(L['loc_affine']); cm = np.array(L['loc_cm'] + [1])
ras = loc_aff @ cm
xyz = (np.linalg.inv(t2.affine) @ ras).astype(np.int16)
print('xyz', xyz[:3], 'ras', ras[:3])
p = 48; t2_arr = t2.get_fdata()
crop = t2_arr[xyz[0]-p:xyz[0]+p, xyz[1]-p:xyz[1]+p, xyz[2]-p:xyz[2]+p]
zero = np.array([xyz[0]-p, xyz[1]-p, xyz[2]-p, 1]); aff = t2.affine.copy(); aff[:3, 3] = (t2.affine @ zero)[:3]
crop_img = nib.Nifti1Image(crop, aff, t2.header)
net = OBNet(args, FLAGS, logger)
pred, logits = net.run_segmentation(crop)
pred_img = clean_seg(nib.Nifti1Image(pred, aff, t2.header), ras)
lab = np.rint(pred_img.get_fdata()).astype(np.uint8)
os.makedirs(os.path.join(OUT, 'chris', 'stats'), exist_ok=True)
np.save(os.path.join(OUT, 'labels.npy'), lab); np.save(os.path.join(OUT, 'crop.npy'), crop.astype(np.float32))
np.save(os.path.join(OUT, 'logits_p1.npy'), logits[..., 1, :].astype(np.float32))
json.dump({'xyz': xyz[:3].tolist(), 'ras': ras[:3].tolist(), 'aff': aff.tolist()}, open(os.path.join(OUT, 'info.json'), 'w'))
stats.calculate_stats(args, os.path.join(OUT, 'chris'), image=crop_img, prediction=pred_img, logits=logits,
                      cm={'ras': ras, 'xyz': xyz, 'zero_xyz': zero}, cm_logits=None, logger=logger)
print(open(os.path.join(OUT, 'chris', 'stats', 'segmentation_stats.csv')).read())
