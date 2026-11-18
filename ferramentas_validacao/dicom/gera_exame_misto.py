import numpy as np, os, shutil
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid, MRImageStorage
out='mix'; shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
study=generate_uid()
def series(desc, num, acq, seq, iop, ps, dz, n, rows, cols, itype='ORIGINAL\\PRIMARY\\M', gap_skip=None):
    suid=generate_uid(); iop=np.array(iop,float); nrm=np.cross(iop[:3],iop[3:])
    for k in range(n):
        if gap_skip is not None and k==gap_skip: continue
        ds=Dataset(); meta=FileMetaDataset(); meta.MediaStorageSOPClassUID=MRImageStorage; sop=generate_uid(); meta.MediaStorageSOPInstanceUID=sop; meta.TransferSyntaxUID=ExplicitVRLittleEndian
        ds.file_meta=meta; ds.SOPClassUID=MRImageStorage; ds.SOPInstanceUID=sop; ds.Modality='MR'; ds.StudyInstanceUID=study; ds.SeriesInstanceUID=suid
        ds.SeriesNumber=num; ds.SeriesDescription=desc; ds.MRAcquisitionType=acq; ds.SequenceName=seq; ds.ImageType=itype.split('\\'); ds.InstanceNumber=k+1
        ds.ImagePositionPatient=list(np.array([0,0,0])+k*dz*nrm); ds.ImageOrientationPatient=list(iop); ds.PixelSpacing=ps; ds.SliceThickness=dz
        ds.Rows=rows; ds.Columns=cols; ds.SamplesPerPixel=1; ds.PhotometricInterpretation='MONOCHROME2'; ds.BitsAllocated=16; ds.BitsStored=12; ds.HighBit=11; ds.PixelRepresentation=0
        ds.PixelData=np.full((rows,cols),k,np.uint16).tobytes(); ds.PatientID='P1'; ds.StudyDate='20261118'
        ds.save_as(f'{out}/s{num}_{k:03d}.dcm', enforce_file_format=True)
ax=[1,0,0,0,1,0]; sag=[0,1,0,0,0,-1]; cor=[1,0,0,0,0,-1]
for i,o in enumerate([ax,sag,cor]): series('localizer', 1, '2D', '*fl2d1', o, [1.6,1.6], 7, 1, 64, 64)
series('t1_mprage_sag_1mm', 2, '3D', '*tfl3d1_16ns', sag, [1,1], 1, 40, 64, 64)
series('t2_tse_cor_2mm', 3, '2D', '*tse2d1_15', cor, [0.4,0.4], 2, 20, 64, 64)
series('T2_SPACE_sag_0.8iso', 4, '3D', '*spcR_154ns', sag, [0.8,0.8], 0.8, 60, 64, 64)
series('T2_SPACE_sag_0.8iso_MPR_cor', 5, '2D', '*spcR_154ns', cor, [0.8,0.8], 0.8, 60, 64, 64, itype='DERIVED\\SECONDARY\\MPR')
series('t2_flair_3d', 6, '3D', '*spcir', sag, [1,1], 1, 30, 64, 64)
series('T2 3D faltando', 7, '3D', '*spc', ax, [0.7,0.7], 0.7, 20, 64, 64, gap_skip=10)
print('ok')
