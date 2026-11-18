// pipeline.js — porte em JavaScript do pipeline de segmentação do bulbo olfatório
// Estrada S, et al. Automated olfactory bulb segmentation on high resolutional T2-weighted MRI.
// NeuroImage 2021;242:118464. Código original: github.com/Deep-MI/olf-bulb-segmentation (Apache-2.0).
//
// Cada etapa replica uma função do repositório original (indicada entre colchetes):
//   [conform.check_orientation] reorientação para LAS
//   [conform.getscale/scalecrop] normalização robusta de intensidades para 0–255
//   [nibabel.processing.resample_to_output] reamostragem linear para 0,8 mm (e 1,6 mm na localização)
//   [OBNet.run_localization] 3 FastSurferCNN (axial/coronal/sagital) -> centro de massa
//   [OBNet.eval] recorte 96³ em torno do centro
//   [OBNet.run_segmentation] ensemble de AttFastSurferCNN (12 ou 3 modelos)
//   [image_utils.clean_seg] componentes conexos -> bulbo esquerdo (1) / direito (2)
//   [stats.calculate_stats] volumes, intensidades, entropia e CV entre modelos

import { zoomsFromAffine } from './nifti.js';

export const LOC_MODELS = [
  { name: 'v1_split_4_FastSurferCNN_axial', plane: 'axial' },
  { name: 'v4_split_4_FastSurferCNN_coronal', plane: 'coronal' },
  { name: 'v1_split_4_FastSurferCNN_sagittal', plane: 'sagittal' },
];
// ordem idêntica a SegModels/AttFastSurferCNN/AttFastSurferCNN_weights.yml
export const SEG_MODELS = [
  'v3_split_1_AttFastSurferCNN_axial', 'v1_split_2_AttFastSurferCNN_axial', 'v3_split_3_AttFastSurferCNN_axial',
  'v4_split_4_AttFastSurferCNN_axial', 'v4_split_1_AttFastSurferCNN_coronal', 'v3_split_2_AttFastSurferCNN_coronal',
  'v4_split_3_AttFastSurferCNN_coronal', 'v3_split_4_AttFastSurferCNN_coronal', 'v3_split_1_AttFastSurferCNN_sagittal',
  'v3_split_2_AttFastSurferCNN_sagittal', 'v2_split_3_AttFastSurferCNN_sagittal', 'v4_split_4_AttFastSurferCNN_sagittal',
].map((name) => ({ name, plane: ['axial', 'coronal', 'sagittal'].find((p) => name.includes(p)) }));

export function segModelsFor(mode) { // mode: 'all' (12) ou 1..4 (split, 3 modelos) — igual ao argumento --model
  if (mode === 'all' || mode === 5) return SEG_MODELS;
  return SEG_MODELS.filter((m) => m.name.includes(`split_${mode}_`));
}

// ---------------- álgebra 4x4 ----------------
export function matMul(A, B) {
  const C = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    let s = 0; for (let k = 0; k < 4; k++) s += A[i][k] * B[k][j]; C[i][j] = s;
  }
  return C;
}
export function matVec(A, v) { return [0, 1, 2, 3].map((i) => A[i][0] * v[0] + A[i][1] * v[1] + A[i][2] * v[2] + A[i][3] * v[3]); }
export function matInv(m) {
  const a = m.map((r) => r.slice()), inv = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
  for (let c = 0; c < 4; c++) {
    let p = c; for (let r = c + 1; r < 4; r++) if (Math.abs(a[r][c]) > Math.abs(a[p][c])) p = r;
    [a[c], a[p]] = [a[p], a[c]]; [inv[c], inv[p]] = [inv[p], inv[c]];
    const d = a[c][c]; if (Math.abs(d) < 1e-12) throw new Error('Matriz affine singular');
    for (let k = 0; k < 4; k++) { a[c][k] /= d; inv[c][k] /= d; }
    for (let r = 0; r < 4; r++) if (r !== c) {
      const f = a[r][c]; if (f === 0) continue;
      for (let k = 0; k < 4; k++) { a[r][k] -= f * a[c][k]; inv[r][k] -= f * inv[c][k]; }
    }
  }
  return inv;
}

// [nibabel.orientations.io_orientation] — fator ortogonal polar (iteração de Newton) + busca gulosa
export function ioOrientation(aff) {
  const zooms = zoomsFromAffine(aff).map((z) => (z === 0 ? 1 : z));
  let R = [0, 1, 2].map((r) => [0, 1, 2].map((c) => aff[r][c] / zooms[c]));
  for (let it = 0; it < 30; it++) { // R <- (R + R^-T)/2
    const d = R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0]) + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]);
    if (Math.abs(d) < 1e-12) break;
    const cof = [
      [R[1][1] * R[2][2] - R[1][2] * R[2][1], -(R[1][0] * R[2][2] - R[1][2] * R[2][0]), R[1][0] * R[2][1] - R[1][1] * R[2][0]],
      [-(R[0][1] * R[2][2] - R[0][2] * R[2][1]), R[0][0] * R[2][2] - R[0][2] * R[2][0], -(R[0][0] * R[2][1] - R[0][1] * R[2][0])],
      [R[0][1] * R[1][2] - R[0][2] * R[1][1], -(R[0][0] * R[1][2] - R[0][2] * R[1][0]), R[0][0] * R[1][1] - R[0][1] * R[1][0]],
    ]; // R^-T = cof / det
    R = R.map((row, i) => row.map((v, j) => 0.5 * (v + cof[i][j] / d)));
  }
  const ornt = [];
  for (let inAx = 0; inAx < 3; inAx++) {
    const col = [R[0][inAx], R[1][inAx], R[2][inAx]];
    let out = 0; for (let k = 1; k < 3; k++) if (Math.abs(col[k]) > Math.abs(col[out])) out = k;
    ornt.push([out, col[out] < 0 ? -1 : 1]);
    R[out] = [0, 0, 0];
  }
  return ornt;
}
export function axCodes(aff) {
  const labels = [['L', 'R'], ['P', 'A'], ['I', 'S']];
  return ioOrientation(aff).map(([ax, f]) => labels[ax][f > 0 ? 1 : 0]);
}

// [conform.check_orientation + calculated_new_ornt + img.as_reoriented] -> LAS
export function reorientLAS(vol) {
  const base = [[0, -1], [1, 1], [2, 1]];
  const iornt = ioOrientation(vol.affine);
  const ornt = iornt.map(([ax, dir]) => [ax, base.find((b) => b[0] === ax)[1] === dir ? 1 : -1]);
  const same = ornt.every(([ax, f], i) => ax === i && f === 1);
  if (same) return vol;
  const [nx, ny, nz] = vol.dims;
  const inDims = vol.dims;
  const outDims = [0, 0, 0];
  ornt.forEach(([ax], i) => { outDims[ax] = inDims[i]; });
  // M: voxel novo -> voxel antigo
  const M = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]];
  ornt.forEach(([ax, f], a) => { M[a][ax] = f; M[a][3] = f === 1 ? 0 : inDims[a] - 1; });
  const affine = matMul(vol.affine, M);
  const out = new Float32Array(vol.data.length);
  const [ox, oy] = outDims;
  const src = vol.data;
  const idx = [0, 0, 0];
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const v = src[i + nx * (j + ny * k)];
    const c = [i, j, k];
    for (let a = 0; a < 3; a++) { const [ax, f] = ornt[a]; idx[ax] = f === 1 ? c[a] : inDims[a] - 1 - c[a]; }
    out[idx[0] + ox * (idx[1] + oy * idx[2])] = v;
  }
  return { dims: outDims, affine, data: out };
}

// [conform.getscale(f_low=0, f_high=0.999) + scalecrop] -> 0..255 (float, sem arredondar)
export function conformIntensity(data) {
  let mn = Infinity, mx = -Infinity, nz = 0;
  for (let i = 0; i < data.length; i++) { const v = data[i]; if (v < mn) mn = v; if (v > mx) mx = v; if (Math.abs(v) >= 1e-15) nz++; }
  if (mn < 0) throw new Error('Intensidade mínima negativa: o pipeline original exige valores ≥ 0 (verifique a imagem).');
  const voxnum = data.length, bins = 1000;
  let srcMin = mn, scale = 1;
  if (mx > mn) {
    const binSize = (mx - mn) / bins;
    // numpy.histogram com bordas = linspace(mn, mx, 1001)
    const step = (mx - mn) / bins;
    const edges = new Float64Array(bins + 1);
    for (let i = 0; i <= bins; i++) edges[i] = mn + i * step;
    edges[bins] = mx;
    const hist = new Float64Array(bins);
    const norm = bins / (mx - mn);
    for (let i = 0; i < voxnum; i++) {
      const v = data[i];
      let b = Math.floor((v - mn) * norm);
      if (b >= bins) b = bins - 1;
      if (v < edges[b]) b -= 1;
      else if (b !== bins - 1 && v >= edges[b + 1]) b += 1;
      hist[b]++;
    }
    const cs = new Float64Array(bins + 1);
    for (let i = 0; i < bins; i++) cs[i + 1] = cs[i] + hist[i];
    const nth = voxnum - Math.trunc((1.0 - 0.999) * nz);
    let idx = -1;
    for (let i = 0; i <= bins; i++) if (cs[i] >= nth) { idx = i - 2; break; }
    const srcMax = idx * binSize + mn;
    srcMin = mn;
    scale = srcMin === srcMax ? 1 : 255 / (srcMax - srcMin);
  }
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) { let v = scale * (data[i] - srcMin); out[i] = v < 0 ? 0 : v > 255 ? 255 : v; }
  return out;
}

// [nibabel.processing.resample_to_output(img, voxel_sizes, order=1)] — grade alinhada ao mundo (RAS+)
export function resampleToOutput(vol, vox) {
  const [nx, ny, nz] = vol.dims;
  const corners = [];
  for (const a of [0, nx - 1]) for (const b of [0, ny - 1]) for (const c of [0, nz - 1]) corners.push(matVec(vol.affine, [a, b, c, 1]));
  const mn = [0, 1, 2].map((d) => Math.min(...corners.map((p) => p[d])));
  const mxw = [0, 1, 2].map((d) => Math.max(...corners.map((p) => p[d])));
  const outDims = [0, 1, 2].map((d) => Math.ceil((mxw[d] - mn[d]) / vox[d]) + 1);
  const outAff = [[vox[0], 0, 0, mn[0]], [0, vox[1], 0, mn[1]], [0, 0, vox[2], mn[2]], [0, 0, 0, 1]];
  const M = matMul(matInv(vol.affine), outAff);
  const [ox, oy, oz] = outDims;
  const out = new Float32Array(ox * oy * oz);
  const src = vol.data, sxy = nx * ny;
  const eps = 1e-9;
  for (let k = 0; k < oz; k++) for (let j = 0; j < oy; j++) {
    const bx = M[0][1] * j + M[0][2] * k + M[0][3];
    const by = M[1][1] * j + M[1][2] * k + M[1][3];
    const bz = M[2][1] * j + M[2][2] * k + M[2][3];
    let o = ox * (j + oy * k);
    for (let i = 0; i < ox; i++, o++) {
      const x = bx + M[0][0] * i, y = by + M[1][0] * i, z = bz + M[2][0] * i;
      if (x < -eps || y < -eps || z < -eps || x > nx - 1 + eps || y > ny - 1 + eps || z > nz - 1 + eps) continue;
      let x0 = Math.floor(x), y0 = Math.floor(y), z0 = Math.floor(z);
      if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0; if (z0 < 0) z0 = 0;
      if (x0 > nx - 1) x0 = nx - 1; if (y0 > ny - 1) y0 = ny - 1; if (z0 > nz - 1) z0 = nz - 1;
      const fx = x - x0, fy = y - y0, fz = z - z0;
      const x1 = Math.min(x0 + 1, nx - 1), y1 = Math.min(y0 + 1, ny - 1), z1 = Math.min(z0 + 1, nz - 1);
      const c000 = src[x0 + nx * y0 + sxy * z0], c100 = src[x1 + nx * y0 + sxy * z0];
      const c010 = src[x0 + nx * y1 + sxy * z0], c110 = src[x1 + nx * y1 + sxy * z0];
      const c001 = src[x0 + nx * y0 + sxy * z1], c101 = src[x1 + nx * y0 + sxy * z1];
      const c011 = src[x0 + nx * y1 + sxy * z1], c111 = src[x1 + nx * y1 + sxy * z1];
      const c00 = c000 + (c100 - c000) * fx, c10 = c010 + (c110 - c010) * fx;
      const c01 = c001 + (c101 - c001) * fx, c11 = c011 + (c111 - c011) * fx;
      const c0 = c00 + (c10 - c00) * fy, c1 = c01 + (c11 - c01) * fy;
      out[o] = c0 + (c1 - c0) * fz;
    }
  }
  return { dims: outDims, affine: outAff, data: out };
}

export function needsResample(affine, target = 0.8) { // np.allclose(zooms, 0.8, rtol=0.05)
  return !zoomsFromAffine(affine).every((z) => Math.abs(z - target) <= 1e-8 + 0.05 * target);
}

// -------- fatiamento por plano: [image_utils.plane_swap / map_size / get_thick_slices] --------
function planeGeom(plane, dims) {
  const [nx, ny, nz] = dims;
  // (d, h, w) -> (i, j, k)
  if (plane === 'axial') return { D: nz, Hm: ny, Wm: nx, vox: (d, h, w) => w + nx * (h + ny * d) };
  if (plane === 'coronal') return { D: ny, Hm: nz, Wm: nx, vox: (d, h, w) => w + nx * (d + ny * h) };
  return { D: nx, Hm: ny, Wm: nz, vox: (d, h, w) => d + nx * (h + ny * w) }; // sagittal
}

/**
 * Executa um modelo 2D em todas as fatias de um plano e acumula softmax (float64) em acc0/acc1.
 * runner(modelName, Float32Array input[n,3,S,S], n) -> Promise<Float32Array logits[n,2,S,S]>
 */
export async function runPlaneModel({ vol, plane, size, modelName, runner, batch, acc0, acc1, keepP1, onSlice, signal }) {
  const g = planeGeom(plane, vol.dims);
  const { D, Hm, Wm } = g, S = size, SS = S * S;
  const offH = Math.floor(Hm / 2) - Math.floor(S / 2), offW = Math.floor(Wm / 2) - Math.floor(S / 2);
  const src = vol.data;
  // fatia 2D normalizada (/255, clip 0..1) no grid SxS da rede
  const sliceCache = new Map();
  const getSlice = (d) => {
    if (sliceCache.has(d)) return sliceCache.get(d);
    const s = new Float32Array(SS);
    let any = false;
    for (let hh = 0; hh < S; hh++) {
      const h = hh + offH; if (h < 0 || h >= Hm) continue;
      for (let ww = 0; ww < S; ww++) {
        const w = ww + offW; if (w < 0 || w >= Wm) continue;
        let v = src[g.vox(d, h, w)] / 255; v = v < 0 ? 0 : v > 1 ? 1 : v;
        s[hh * S + ww] = v; if (v !== 0) any = true;
      }
    }
    const r = { s, any };
    sliceCache.set(d, r);
    if (sliceCache.size > 8) sliceCache.delete(sliceCache.keys().next().value);
    return r;
  };
  const p1Model = keepP1 ? new Float32Array(acc1.length) : null;
  const fgModel = keepP1 ? new Uint8Array(acc1.length) : null;
  let zeroOut = null; // saída para entrada toda zero (determinística) — reaproveitada
  const accumulate = (d, logits, base) => {
    for (let h = 0; h < Hm; h++) {
      const hh = h - offH;
      for (let w = 0; w < Wm; w++) {
        const ww = w - offW;
        let l0 = 0, l1 = 0;
        if (hh >= 0 && hh < S && ww >= 0 && ww < S) { l0 = logits[base + hh * S + ww]; l1 = logits[base + SS + hh * S + ww]; }
        const m = l0 > l1 ? l0 : l1;
        const e0 = Math.exp(l0 - m), e1 = Math.exp(l1 - m), s = e0 + e1;
        const p0 = e0 / s, p1 = e1 / s;
        const vi = g.vox(d, h, w);
        acc0[vi] += p0; acc1[vi] += p1;
        if (keepP1) {
          p1Model[vi] = p1;
          const c0 = Math.min(Math.max(p0, 1e-5), 1 - 1e-5), c1 = Math.min(Math.max(p1, 1e-5), 1 - 1e-5);
          fgModel[vi] = c1 > c0 ? 1 : 0;
        }
      }
    }
  };
  for (let d0 = 0; d0 < D; d0 += batch) {
    if (signal?.aborted) throw new Error('Cancelado');
    const ds = []; for (let d = d0; d < Math.min(D, d0 + batch); d++) ds.push(d);
    const inputs = ds.map((d) => {
      const chans = [-1, 0, 1].map((o) => getSlice(Math.min(Math.max(d + o, 0), D - 1)));
      return { d, chans, zero: !chans.some((c) => c.any) };
    });
    const work = inputs.filter((x) => !x.zero);
    if (inputs.some((x) => x.zero) && !zeroOut) {
      zeroOut = await runner(modelName, new Float32Array(3 * SS), 1);
    }
    if (work.length) {
      const inp = new Float32Array(work.length * 3 * SS);
      work.forEach((x, b) => x.chans.forEach((c, ch) => inp.set(c.s, (b * 3 + ch) * SS)));
      const out = await runner(modelName, inp, work.length);
      work.forEach((x, b) => accumulate(x.d, out, b * 2 * SS));
    }
    inputs.filter((x) => x.zero).forEach((x) => accumulate(x.d, zeroOut, 0));
    onSlice?.(Math.min(D, d0 + batch), D);
  }
  return { p1Model, fgModel };
}

// -------- componentes conexos 3D (26-vizinhança, como skimage.measure.label) --------
export function connectedComponents(mask, dims) {
  const [nx, ny, nz] = dims, n = mask.length;
  const lab = new Int32Array(n);
  const comps = [];
  const stack = new Int32Array(n);
  let next = 0;
  for (let s = 0; s < n; s++) {
    if (!mask[s] || lab[s]) continue;
    next++;
    let sp = 0; stack[sp++] = s; lab[s] = next;
    let cnt = 0, si = 0, sj = 0, sk = 0;
    const vox = [];
    while (sp) {
      const v = stack[--sp];
      const i = v % nx, j = Math.floor(v / nx) % ny, k = Math.floor(v / (nx * ny));
      cnt++; si += i; sj += j; sk += k; vox.push(v);
      for (let dk = -1; dk <= 1; dk++) { const kk = k + dk; if (kk < 0 || kk >= nz) continue;
        for (let dj = -1; dj <= 1; dj++) { const jj = j + dj; if (jj < 0 || jj >= ny) continue;
          for (let di = -1; di <= 1; di++) { const ii = i + di; if (ii < 0 || ii >= nx) continue;
            const u = ii + nx * (jj + ny * kk);
            if (mask[u] && !lab[u]) { lab[u] = next; stack[sp++] = u; } } } }
    }
    comps.push({ id: next, count: cnt, centroid: [si / cnt, sj / cnt, sk / cnt], vox });
  }
  return comps;
}

// [image_utils.clean_seg] — rotula cada componente como 1 (esquerdo) ou 2 (direito) pela posição RAS-x
export function cleanSeg(mask, dims, affine, rasCm) {
  const out = new Uint8Array(mask.length);
  const comps = connectedComponents(mask, dims);
  if (!comps.length) return { labels: out, comps };
  const ax0 = axCodes(affine)[0];
  for (const c of comps) {
    const ras = matVec(affine, [...c.centroid, 1]);
    let lab;
    if (ax0 === 'L') lab = ras[0] > rasCm[0] ? 2 : 1;
    else if (ax0 === 'R') lab = ras[0] < rasCm[0] ? 1 : 2;
    else lab = 0;
    for (const v of c.vox) out[v] = lab;
    c.label = lab;
  }
  return { labels: out, comps };
}

const r4 = (x) => (x === null || Number.isNaN(x) ? NaN : Math.round(x * 1e4) / 1e4);

// [stats.calculate_stats]
export function computeStats({ labels, crop, cropDims, cropAffine, perModel, cmRas, zeroXYZ, locXYZ }) {
  const zooms = zoomsFromAffine(cropAffine).map(Math.fround);
  const voxelSize = Math.fround(Math.fround(zooms[0] * zooms[1]) * zooms[2]);
  const structs = [['Left_OB', 1], ['Right_OB', 2], ['Total_OB', 3]];
  const rows = {};
  const n = labels.length;
  // incerteza por modelo
  const perModelLabels = perModel.map((m) => cleanSeg(m.fg, cropDims, cropAffine, cmRas).labels);
  for (const [name, L] of structs) {
    const sel = (lab) => (L === 3 ? lab > 0 : lab === L);
    let cnt = 0, sum = 0, mn = Infinity, mx = -Infinity, ci = 0, cj = 0, ck = 0;
    const [nx, ny] = cropDims;
    for (let v = 0; v < n; v++) if (sel(labels[v])) {
      cnt++; const x = crop[v]; sum += x; if (x < mn) mn = x; if (x > mx) mx = x;
      ci += v % nx; cj += Math.floor(v / nx) % ny; ck += Math.floor(v / (nx * ny));
    }
    const ents = [], nums = [];
    perModel.forEach((m, i) => {
      const lab = perModelLabels[i];
      let num = 0, e = 0;
      for (let v = 0; v < n; v++) if (sel(lab[v])) {
        num++; const p = Math.min(Math.max(m.p1[v], 1e-5), 1 - 1e-5); e += p * Math.log(p);
      }
      ents.push(num > 0 ? -e / num : 0); nums.push(num);
    });
    const meanEnt = ents.reduce((a, b) => a + b, 0) / ents.length;
    const meanN = nums.reduce((a, b) => a + b, 0) / nums.length;
    const sdN = Math.sqrt(nums.reduce((a, b) => a + (b - meanN) ** 2, 0) / nums.length);
    rows[name] = cnt ? {
      NVoxels: cnt, Volume_mm3: r4(cnt * voxelSize), normMean: r4(sum / cnt), normMin: r4(mn), normMax: r4(mx),
      Entropy: r4(meanEnt), CV: r4(meanN ? sdN / meanN : NaN),
      CM: [ci / cnt, cj / cnt, ck / cnt].map((c, a) => Math.trunc(c + zeroXYZ[a])),
      perModelVolumes: nums.map((x) => r4(x * voxelSize)),
    } : { NVoxels: 0, Volume_mm3: 0, normMean: NaN, normMin: NaN, normMax: NaN, Entropy: r4(meanEnt), CV: r4(meanN ? sdN / meanN : NaN), CM: null, perModelVolumes: nums.map((x) => r4(x * voxelSize)) };
  }
  const has1 = labels.some((v) => v === 1), has2 = labels.some((v) => v === 2);
  let flag = null;
  if (!has1 || !has2) flag = has1 ? 'OB direito não encontrado' : has2 ? 'OB esquerdo não encontrado' : 'Nenhum OB aparente';
  let locSegDist = NaN;
  if (rows.Total_OB.NVoxels && locXYZ) {
    const diff = rows.Total_OB.CM.map((c, a) => locXYZ[a] - c);
    locSegDist = r4(Math.sqrt(diff.reduce((s, d, a) => s + d * d * zooms[a], 0)));
  }
  return { rows, voxelSize, flag, locSegDist };
}

/**
 * Pipeline completo.
 * opts: { mode: 'all'|1..4, localization: 'auto'|{ ras:[x,y,z] }, runner, batchLoc, batchSeg, onProgress, signal }
 */
export async function runPipeline(native, opts) {
  const P = (stage, frac, msg) => opts.onProgress?.({ stage, frac, msg });
  const t0 = performance.now();
  // 1) conform
  P('pre', 0, 'Reorientando (LAS) e normalizando intensidades');
  const las = reorientLAS(native);
  const conf = { dims: las.dims, affine: las.affine, data: conformIntensity(las.data) };
  // 2) reamostrar para 0,8 mm se necessário
  const inputZooms = zoomsFromAffine(conf.affine);
  let t2 = conf, resampled = false;
  if (needsResample(conf.affine)) { P('pre', 0.5, 'Reamostrando para 0,8 mm isotrópico'); t2 = resampleToOutput(conf, [0.8, 0.8, 0.8]); resampled = true; }
  // 3) localização
  let xyz, cmRas, locInfo = {};
  if (opts.localization === 'auto') {
    const loc = resampleToOutput(t2, [1.6, 1.6, 1.6]);
    const n = loc.data.length;
    const acc0 = new Float64Array(n), acc1 = new Float64Array(n);
    for (let m = 0; m < LOC_MODELS.length; m++) {
      const M = LOC_MODELS[m];
      await runPlaneModel({ vol: loc, plane: M.plane, size: 192, modelName: M.name, runner: opts.runner, batch: opts.batchLoc || 4, acc0, acc1, signal: opts.signal,
        onSlice: (i, D) => P('loc', (m + i / D) / LOC_MODELS.length, `Localização ${M.plane} ${i}/${D}`) });
    }
    let cnt = 0, si = 0, sj = 0, sk = 0, roi = 0;
    const [lx, ly] = loc.dims;
    const heat = new Float32Array(n);
    for (let v = 0; v < n; v++) {
      const a1 = acc1[v] / 3, a0 = acc0[v] / 3;
      heat[v] = a1;
      if (a1 >= 0.5) roi++;
      if (a1 > a0) { cnt++; si += v % lx; sj += Math.floor(v / lx) % ly; sk += Math.floor(v / (lx * ly)); }
    }
    if (!cnt) throw new Error('A rede de localização não encontrou a região dos bulbos. Verifique a imagem ou use a localização manual.');
    const cm = [Math.trunc(si / cnt), Math.trunc(sj / cnt), Math.trunc(sk / cnt)];
    cmRas = matVec(loc.affine, [...cm, 1]);
    xyz = matVec(matInv(t2.affine), cmRas).slice(0, 3).map(Math.trunc);
    locInfo = { locCm: cm, roiNVoxels: roi, locDims: loc.dims, locAffine: loc.affine, heat };
  } else {
    const ras = opts.localization.ras;
    cmRas = [ras[0], ras[1], ras[2], 1];
    xyz = matVec(matInv(t2.affine), cmRas).slice(0, 3).map(Math.trunc);
  }
  // 4) recorte 96³
  const pad = 48, S = 96;
  const zeroXYZ = xyz.map((c) => c - pad);
  const [tx, ty, tz] = t2.dims;
  const crop = new Float32Array(S * S * S);
  let outside = 0;
  for (let k = 0; k < S; k++) for (let j = 0; j < S; j++) for (let i = 0; i < S; i++) {
    const a = zeroXYZ[0] + i, b = zeroXYZ[1] + j, c = zeroXYZ[2] + k;
    if (a < 0 || b < 0 || c < 0 || a >= tx || b >= ty || c >= tz) { outside++; continue; }
    crop[i + S * (j + S * k)] = t2.data[a + tx * (b + ty * c)];
  }
  const tr = matVec(t2.affine, [...zeroXYZ, 1]);
  const cropAffine = t2.affine.map((r) => r.slice()); cropAffine[0][3] = tr[0]; cropAffine[1][3] = tr[1]; cropAffine[2][3] = tr[2];
  const cropVol = { dims: [S, S, S], affine: cropAffine, data: crop };
  // 5) segmentação (ensemble)
  const models = segModelsFor(opts.mode);
  const nC = crop.length;
  const acc0 = new Float64Array(nC), acc1 = new Float64Array(nC);
  const perModel = [];
  for (let m = 0; m < models.length; m++) {
    const M = models[m];
    const { p1Model, fgModel } = await runPlaneModel({ vol: cropVol, plane: M.plane, size: 96, modelName: M.name, runner: opts.runner, batch: opts.batchSeg || 2, acc0, acc1, keepP1: true, signal: opts.signal,
      onSlice: (i, D) => P('seg', (m + i / D) / models.length, `Segmentação ${m + 1}/${models.length} (${M.plane}) ${i}/${D}`) });
    perModel.push({ name: M.name, p1: p1Model, fg: fgModel });
  }
  const mask = new Uint8Array(nC);
  for (let v = 0; v < nC; v++) mask[v] = acc1[v] > acc0[v] ? 1 : 0;
  const { labels } = cleanSeg(mask, [S, S, S], cropAffine, cmRas);
  const stats = computeStats({ labels, crop, cropDims: [S, S, S], cropAffine, perModel, cmRas, zeroXYZ, locXYZ: xyz });
  P('done', 1, 'Concluído');
  return {
    labels, cropAffine, cropDims: [S, S, S], crop, t2, resampled, inputZooms, xyz, zeroXYZ, cmRas, outsideVoxels: outside,
    stats, locInfo, mode: opts.mode, nModels: models.length, seconds: (performance.now() - t0) / 1000,
  };
}
