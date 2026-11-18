import json, sys, numpy as np, nibabel as nib, pandas as pd
ref, js_json, js_lab = sys.argv[1:4]
J = json.load(open(js_json))
L = json.load(open(f'{ref}/loc_info.json'))
print('t2 dims   py', L['t2_shape'], ' js', J['t2Dims'])
print('loc cm    py', L['loc_cm'], ' js', J['locCm'])
log = open(f'{ref}/log.txt').read()
import re
m = re.search(r'Crop image from coordinate\s+(-?\d+),\s*(-?\d+)\s*,\s*(-?\d+)', log)
print('crop xyz  py', [int(x) for x in m.groups()], ' js', J['xyz'])
# t2 conformado
t2py = nib.load(f'{ref}/mri/orig.nii.gz')  # salvo como uint8 (apenas para conferência grosseira)
t2js = np.fromfile(js_lab + '.t2.f32', dtype=np.float32).reshape(J['t2Dims'][::-1]).transpose(2,1,0)
print('t2 affine py\n', np.round(t2py.affine,4), '\njs\n', np.round(np.array(J['t2Affine']),4))
d = np.abs(np.asarray(t2py.dataobj, dtype=np.float64) - np.clip(t2js, 0, 255).astype(np.uint8)); print('t2 (uint8) max|diff|', d.max(), 'frac>1:', (d>1).mean())
# heatmap localização
hp = np.load(f'{ref}/loc_p1.npy'); hj = np.fromfile(js_lab + '.heat.f32', dtype=np.float32).reshape(J['locDims'][::-1]).transpose(2,1,0)
print('loc heat max|diff|', np.abs(hp - hj).max())
# segmentação
sp = np.rint(np.asarray(nib.load(f'{ref}/mri/ob_seg.nii.gz').dataobj)).astype(np.uint8)
sj = np.fromfile(js_lab, dtype=np.uint8).reshape(96,96,96).transpose(2,1,0)
for lab in (1,2):
    a, b = sp == lab, sj == lab
    print(f'label {lab}: py {a.sum()} js {b.sum()} dice {2*(a&b).sum()/max(1,a.sum()+b.sum()):.6f}')
print('voxels discordantes:', (sp != sj).sum())
st = pd.read_csv(f'{ref}/stats/segmentation_stats.csv')
for _, r in st.iterrows():
    jr = J['stats']['rows'][r.StructName]
    print(r.StructName, 'vol py', r.Volume_mm3, 'js', jr['Volume_mm3'], '| mean py', r.normMean, 'js', jr['normMean'], '| ent py', r.Entropy, 'js', jr['Entropy'], '| CV py', r.CV, 'js', jr['CV'], '| CM py', r.CM, 'js', jr['CM'])
lc = pd.read_csv(f'{ref}/stats/localization_stats.csv'); print(lc.to_string()); print('js dist', J['stats']['locSegDist'], 'roi', J['roi'], 'flag', J['stats']['flag'])
