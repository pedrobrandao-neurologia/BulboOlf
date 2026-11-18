import nibabel as nib, numpy as np, os
truth = np.load('dcmtest/truth.npy').astype(np.float64)
g = np.load('dcmtest/geom.npy'); row, col, nrm, ipp0 = g[0:3], g[3:6], g[6:9], g[9:12]; ps_r, ps_c, dz = g[12:15]
A = np.eye(4); A[:3,0]=row*ps_c; A[:3,1]=col*ps_r; A[:3,2]=nrm*dz; A[:3,3]=ipp0; A = np.diag([-1,-1,1,1]) @ A
xf = {'signed_slope': lambda t: (t-1000)*1.5+3, 'jpeg_signed': lambda t: (t-1000)*1.5+3, 'enh': lambda t: t*2}
for d in ['exp_le','imp_le','exp_be','signed_slope','rle','deflate','enh','jpeg_sv1','jpeg_p14','jpeg_signed']:
    im = nib.load(f'js_{d}.nii.gz'); a = im.get_fdata()
    exp = xf.get(d, lambda t: t)(truth).transpose(2,1,0)
    qa = im.get_qform(); sa = im.affine
    line = f'{d:13s} dtype={im.get_data_dtype()} maxΔdados={np.abs(a-exp).max():.3g} maxΔsform={np.abs(sa-A).max():.2e} maxΔqform={np.abs(qa-A).max():.2e}'
    p = f'd2n/{d}/{d}.nii.gz'
    if os.path.exists(p) and not d.startswith('jpeg'):
        r = nib.load(p); c1 = nib.as_closest_canonical(im); c2 = nib.as_closest_canonical(r)
        same = c1.shape == c2.shape
        line += f' | dcm2niix: forma igual={same}'
        if same: line += f' Δdados={np.abs(c1.get_fdata()-c2.get_fdata()).max():.3g} Δaffine={np.abs(c1.affine-c2.affine).max():.2e}'
    print(line)
