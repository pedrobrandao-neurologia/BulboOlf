// worker.js — roda o pipeline fora da thread da interface (ONNX Runtime Web: WebGPU ou WASM).
import * as ort from '../vendor/ort.webgpu.min.mjs';
import { parseNifti, gunzipIfNeeded } from './nifti.js';
import { runPipeline, reorientLAS, conformIntensity, needsResample, resampleToOutput } from './pipeline.js';

const base = new URL('../', import.meta.url).href;
ort.env.wasm.wasmPaths = base + 'vendor/';
ort.env.logLevel = 'error';

let backend = null; // 'webgpu' | 'wasm'
const sessions = new Map();
let abort = null;

async function pickBackend(pref) {
  const threads = self.crossOriginIsolated ? Math.max(1, Math.min(16, (navigator.hardwareConcurrency || 4) - 1)) : 1;
  ort.env.wasm.numThreads = threads;
  ort.env.wasm.simd = true;
  let gpu = false;
  if (pref !== 'wasm' && self.navigator?.gpu) {
    try { gpu = !!(await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })); } catch { gpu = false; }
  }
  backend = gpu ? 'webgpu' : 'wasm';
  return { backend, threads, isolated: self.crossOriginIsolated, webgpuAvailable: gpu };
}

async function getSession(name) {
  if (sessions.has(name)) return sessions.get(name);
  // uma sessão por vez: os modelos rodam em sequência e cada sessão retém sua memória de pico
  for (const [k, old] of sessions) { try { await old.release(); } catch {} sessions.delete(k); }
  const res = await fetch(base + 'models/' + name + '.onnx');
  if (!res.ok) throw new Error('Modelo não encontrado: ' + name);
  const bytes = new Uint8Array(await res.arrayBuffer());
  let s;
  try {
    s = await ort.InferenceSession.create(bytes, { executionProviders: [backend], graphOptimizationLevel: 'all', enableCpuMemArena: false });
  } catch (e) {
    if (backend === 'webgpu') { // recai para CPU se o WebGPU falhar neste modelo
      postMessage({ type: 'warn', msg: 'WebGPU falhou (' + (e.message || e) + '); usando WASM.' });
      backend = 'wasm';
      s = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all', enableCpuMemArena: false });
    } else throw e;
  }
  sessions.set(name, s);
  return s;
}

async function runner(name, input, n) {
  const s = await getSession(name);
  const S = Math.round(Math.sqrt(input.length / (3 * n)));
  const out = await s.run({ input: new ort.Tensor('float32', input, [n, 3, S, S]) });
  const data = out.logits.data;
  return data instanceof Float32Array ? data : Float32Array.from(data);
}

self.onmessage = async (ev) => {
  const m = ev.data;
  try {
    if (m.type === 'init') {
      for (const s of sessions.values()) { try { await s.release(); } catch {} }
      sessions.clear();
      postMessage({ type: 'ready', ...(await pickBackend(m.backend)) });
    } else if (m.type === 'run') {
      abort = new AbortController();
      const vol = parseNifti(await gunzipIfNeeded(m.buffer));
      const res = await runPipeline(vol, {
        ...m.opts, runner, signal: abort.signal,
        onProgress: (p) => postMessage({ type: 'progress', id: m.id, ...p }),
      });
      const t2u8 = new Uint8Array(res.t2.data.length);
      for (let i = 0; i < t2u8.length; i++) t2u8[i] = Math.round(res.t2.data[i]);
      const heat = res.locInfo.heat; delete res.locInfo.heat;
      const payload = {
        type: 'result', id: m.id, backend,
        labels: res.labels, cropAffine: res.cropAffine, cropDims: res.cropDims, zeroXYZ: res.zeroXYZ, xyz: res.xyz, cmRas: res.cmRas,
        t2: { dims: res.t2.dims, affine: res.t2.affine, data: t2u8 }, resampled: res.resampled, inputZooms: res.inputZooms,
        stats: res.stats, locInfo: res.locInfo, nModels: res.nModels, mode: res.mode, seconds: res.seconds, outsideVoxels: res.outsideVoxels,
        inputInfo: { dims: vol.dims, nVolumes: vol.nVolumes },
      };
      postMessage(payload, [res.labels.buffer, t2u8.buffer]);
      void heat;
    } else if (m.type === 'prep') { // só pré-processamento (para reabrir a grade do pipeline sem rodar as redes)
      const vol = parseNifti(await gunzipIfNeeded(m.buffer));
      let t2 = reorientLAS(vol);
      t2 = { dims: t2.dims, affine: t2.affine, data: conformIntensity(t2.data) };
      if (needsResample(t2.affine)) t2 = resampleToOutput(t2, [0.8, 0.8, 0.8]);
      const u8 = new Uint8Array(t2.data.length);
      for (let i = 0; i < u8.length; i++) u8[i] = Math.round(t2.data[i]);
      postMessage({ type: 'prep', id: m.id, t2: { dims: t2.dims, affine: t2.affine, data: u8 } }, [u8.buffer]);
    } else if (m.type === 'selftest') {
      const report = [];
      for (const t of m.tests) {
        const input = new Float32Array(await (await fetch(base + t.input)).arrayBuffer());
        const expected = new Float32Array(await (await fetch(base + t.expected)).arrayBuffer());
        const t0 = performance.now();
        const got = await runner(t.model, input, 1);
        // compara probabilidades (softmax): diferenças de logit em áreas de fundo (empates no max-pooling) não alteram p
        let maxd = 0, maxdp = 0, agree = 0, fg = 0; const SS = got.length / 2;
        for (let i = 0; i < got.length; i++) maxd = Math.max(maxd, Math.abs(got[i] - expected[i]));
        for (let i = 0; i < SS; i++) {
          const pg = 1 / (1 + Math.exp(got[i] - got[SS + i])), pe = 1 / (1 + Math.exp(expected[i] - expected[SS + i]));
          maxdp = Math.max(maxdp, Math.abs(pg - pe));
          agree += ((got[SS + i] > got[i]) === (expected[SS + i] > expected[i])) ? 1 : 0;
          fg += expected[SS + i] > expected[i] ? 1 : 0;
        }
        report.push({ model: t.model, maxAbsDiff: maxd, maxProbDiff: maxdp, argmaxAgreement: agree / SS, fgPixels: fg, ms: performance.now() - t0 });
      }
      postMessage({ type: 'selftest', backend, report });
    } else if (m.type === 'cancel') {
      abort?.abort();
    }
  } catch (e) {
    postMessage({ type: 'error', id: m.id, msg: e.message || String(e) });
  }
};
