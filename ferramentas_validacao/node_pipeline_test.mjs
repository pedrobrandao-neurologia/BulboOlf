// Executa o pipeline JS em Node (ONNX Runtime Web/WASM) e grava saídas para comparação com o Python.
import fs from 'node:fs';
import * as ort from 'onnxruntime-web';
import { parseNifti, gunzipIfNeeded } from '../app/js/nifti.js';
import { runPipeline } from '../app/js/pipeline.js';

const [img, modelDir, mapMode, mode, outJson, outLabels, rasArg] = process.argv.slice(2);
ort.env.wasm.numThreads = 1;
const sessions = new Map();
function onnxPath(name) {
  if (mapMode === 'real') return `${modelDir}/${name}.onnx`;
  const kind = name.includes('Att') ? 'seg' : 'loc';
  const plane = ['axial', 'coronal', 'sagittal'].find((p) => name.includes(p));
  return `${modelDir}/standin_${kind}_${plane}.onnx`;
}
async function runner(name, input, n) {
  const p = onnxPath(name);
  if (!sessions.has(p)) sessions.set(p, await ort.InferenceSession.create(fs.readFileSync(p), { executionProviders: ['wasm'] }));
  const S = Math.round(Math.sqrt(input.length / (3 * n)));
  const out = await sessions.get(p).run({ input: new ort.Tensor('float32', input, [n, 3, S, S]) });
  return out.logits.data;
}
const buf = await gunzipIfNeeded(fs.readFileSync(img).buffer.slice(0));
const vol = parseNifti(buf);
let last = 0;
const res = await runPipeline(vol, { mode: mode === 'all' ? 'all' : Number(mode), localization: rasArg ? { ras: JSON.parse(rasArg) } : 'auto', runner, batchLoc: 8, batchSeg: 2,
  onProgress: (p) => { const t = Date.now(); if (t - last > 20000) { last = t; console.log(p.msg); } } });
fs.writeFileSync(outJson, JSON.stringify({ t2Dims: res.t2.dims, t2Affine: res.t2.affine, xyz: res.xyz, zeroXYZ: res.zeroXYZ, cmRas: res.cmRas,
  locCm: res.locInfo.locCm, roi: res.locInfo.roiNVoxels, locDims: res.locInfo.locDims, stats: res.stats, seconds: res.seconds }, null, 1));
fs.writeFileSync(outLabels, Buffer.from(res.labels.buffer));
if (res.locInfo.heat) fs.writeFileSync(outLabels + '.heat.f32', Buffer.from(res.locInfo.heat.buffer));
fs.writeFileSync(outLabels + '.t2.f32', Buffer.from(res.t2.data.buffer));
console.log('done', res.seconds.toFixed(1), 's');
