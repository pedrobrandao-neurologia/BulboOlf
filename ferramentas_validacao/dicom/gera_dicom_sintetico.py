import numpy as np, pydicom, os, sys, shutil, random
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import (ExplicitVRLittleEndian, ImplicitVRLittleEndian, ExplicitVRBigEndian, RLELossless,
  DeflatedExplicitVRLittleEndian, generate_uid, MRImageStorage, EnhancedMRImageStorage)
from pydicom.sequence import Sequence
out = sys.argv[1]; shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
rng = np.random.default_rng(1)
nx, ny, nz = 72, 80, 50   # cols, rows, slices
# volume suave + ruído; índice [k, row, col]
z, y, x = np.mgrid[0:nz, 0:ny, 0:nx]
vol = (800 + 600*np.sin(x/7.0)*np.cos(y/9.0) + 300*np.cos(z/5.0) + rng.normal(0, 40, (nz, ny, nx))).clip(0, 4000).astype(np.uint16)
np.save(os.path.join(out, 'truth.npy'), vol)
# orientação oblíqua
th = np.deg2rad(12); ph = np.deg2rad(-7)
row = np.array([np.cos(th), np.sin(th), 0.0])
col = np.array([-np.sin(th)*np.cos(ph), np.cos(th)*np.cos(ph), np.sin(ph)])
col -= row*row.dot(col); col /= np.linalg.norm(col)
nrm = np.cross(row, col)
ps = [0.8, 0.75]   # [entre linhas, entre colunas]
dz = 0.7
ipp0 = np.array([-30.0, -60.0, 10.0])
study = generate_uid()
def base(series_uid, desc, num, acq='3D', seq='*spc'):
    ds = Dataset()
    ds.Modality='MR'; ds.StudyInstanceUID=study; ds.SeriesInstanceUID=series_uid; ds.SeriesNumber=num
    ds.SeriesDescription=desc; ds.ProtocolName=desc; ds.MRAcquisitionType=acq; ds.SequenceName=seq
    ds.PatientID='SUJ001'; ds.PatientName='Teste^Sintetico'; ds.StudyDate='20261118'; ds.Manufacturer='SIEMENS'
    ds.ManufacturerModelName='Prisma'; ds.MagneticFieldStrength=3; ds.EchoTime=400; ds.RepetitionTime=3200
    ds.FrameOfReferenceUID=study+'.1'
    return ds
def classic(folder, ts, desc, num, signed=False, slope=None, compress=None, shuffle=True):
    os.makedirs(folder, exist_ok=True)
    suid = generate_uid()
    order = list(range(nz)); random.Random(num).shuffle(order)
    for k in order:
        ds = base(suid, desc, num)
        meta = FileMetaDataset(); meta.MediaStorageSOPClassUID=MRImageStorage; sop=generate_uid(); meta.MediaStorageSOPInstanceUID=sop
        meta.TransferSyntaxUID = ExplicitVRLittleEndian if compress else ts
        ds.file_meta=meta; ds.SOPClassUID=MRImageStorage; ds.SOPInstanceUID=sop
        ds.ImageType=['ORIGINAL','PRIMARY','M','ND']
        ds.InstanceNumber=k+1; ds.ImagePositionPatient=[f'{v:.5f}' for v in ipp0 + k*dz*nrm]
        ds.ImageOrientationPatient=[f'{v:.6f}' for v in list(row)+list(col)]
        ds.PixelSpacing=ps; ds.SliceThickness=dz; ds.SpacingBetweenSlices=dz
        ds.Rows=ny; ds.Columns=nx; ds.SamplesPerPixel=1; ds.PhotometricInterpretation='MONOCHROME2'
        ds.BitsAllocated=16; ds.BitsStored=16 if not signed else 16; ds.HighBit=15; ds.PixelRepresentation=1 if signed else 0
        a = vol[k].astype(np.int16) - (1000 if signed else 0) if signed else vol[k]
        if signed: a = (vol[k].astype(np.int32) - 1000).astype(np.int16)
        if slope: ds.RescaleSlope=slope[0]; ds.RescaleIntercept=slope[1]
        ds.PixelData=a.tobytes()
        if compress == 'rle': ds.compress(RLELossless, encoding_plugin='pydicom')
        name = os.path.join(folder, f'IM{random.Random(k*7+num).randint(0,10**6):07d}')
        if ts == ExplicitVRBigEndian:
            ds.PixelData = a.byteswap().tobytes()
            ds.save_as(name, enforce_file_format=True, little_endian=False, implicit_vr=False)
        else:
            ds.save_as(name, enforce_file_format=True)
    return suid
def enhanced(path, desc, num):
    suid = generate_uid()
    ds = base(suid, desc, num)
    meta = FileMetaDataset(); meta.MediaStorageSOPClassUID=EnhancedMRImageStorage; sop=generate_uid(); meta.MediaStorageSOPInstanceUID=sop
    meta.TransferSyntaxUID=ExplicitVRLittleEndian; ds.file_meta=meta; ds.SOPClassUID=EnhancedMRImageStorage; ds.SOPInstanceUID=sop
    ds.ImageType=['ORIGINAL','PRIMARY','M','NONE']; ds.InstanceNumber=1
    ds.Rows=ny; ds.Columns=nx; ds.SamplesPerPixel=1; ds.PhotometricInterpretation='MONOCHROME2'
    ds.BitsAllocated=16; ds.BitsStored=16; ds.HighBit=15; ds.PixelRepresentation=0
    order = list(range(nz)); random.Random(3).shuffle(order)
    ds.NumberOfFrames=nz
    sh = Dataset()
    pm = Dataset(); pm.PixelSpacing=ps; pm.SliceThickness=dz; sh.PixelMeasuresSequence=Sequence([pm])
    po = Dataset(); po.ImageOrientationPatient=[f'{v:.6f}' for v in list(row)+list(col)]; sh.PlaneOrientationSequence=Sequence([po])
    pv = Dataset(); pv.RescaleSlope=2; pv.RescaleIntercept=0; pv.RescaleType='US'; sh.PixelValueTransformationSequence=Sequence([pv])
    ds.SharedFunctionalGroupsSequence=Sequence([sh])
    pf=[]
    for k in order:
        f = Dataset(); pp = Dataset(); pp.ImagePositionPatient=[f'{v:.5f}' for v in ipp0 + k*dz*nrm]; f.PlanePositionSequence=Sequence([pp])
        fc = Dataset(); fc.StackID='1'; fc.InStackPositionNumber=k+1; f.FrameContentSequence=Sequence([fc]); pf.append(f)
    ds.PerFrameFunctionalGroupsSequence=Sequence(pf)
    ds.PixelData=np.stack([vol[k] for k in order]).tobytes()
    ds.save_as(path, enforce_file_format=True)
# série T2 3D em várias codificações (cada uma num "sujeito" diferente para testar separadamente)
classic(f'{out}/exp_le', ExplicitVRLittleEndian, 'T2_SPACE_sag_0.7iso', 5)
classic(f'{out}/imp_le', ImplicitVRLittleEndian, 'T2_SPACE_sag_0.7iso', 5)
classic(f'{out}/exp_be', ExplicitVRBigEndian, 'T2_SPACE_sag_0.7iso', 5)
classic(f'{out}/signed_slope', ExplicitVRLittleEndian, 'T2 3D', 6, signed=True, slope=(1.5, 3))
classic(f'{out}/rle', None, 'T2_SPACE', 7, compress='rle')
classic(f'{out}/deflate', DeflatedExplicitVRLittleEndian, 'T2_SPACE', 8)
os.makedirs(f'{out}/enh', exist_ok=True); enhanced(f'{out}/enh/enh.dcm', 'T2W_3D_VISTA', 9)
np.save(f'{out}/geom.npy', np.concatenate([row, col, nrm, ipp0, [ps[0], ps[1], dz]]))
print('ok')
