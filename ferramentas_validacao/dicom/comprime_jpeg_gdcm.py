import gdcm, os, sys
src, dst, mode = sys.argv[1:4]
os.makedirs(dst, exist_ok=True)
ts = {'sv1': gdcm.TransferSyntax.JPEGLosslessProcess14_1, 'p14': gdcm.TransferSyntax.JPEGLosslessProcess14}[mode]
for f in os.listdir(src):
    r = gdcm.ImageReader(); r.SetFileName(os.path.join(src, f)); assert r.Read()
    c = gdcm.ImageChangeTransferSyntax(); c.SetTransferSyntax(gdcm.TransferSyntax(ts)); c.SetInput(r.GetImage()); assert c.Change()
    w = gdcm.ImageWriter(); w.SetFileName(os.path.join(dst, f)); w.SetFile(r.GetFile()); w.SetImage(c.GetOutput()); assert w.Write()
print('ok', mode)
