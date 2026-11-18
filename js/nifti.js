// nifti.js — leitura/escrita de NIfTI-1 (.nii / .nii.gz) sem dependências.
// Escolha do affine idêntica ao nibabel (get_best_affine): sform se sform_code≠0, senão qform, senão base.

export async function gunzipIfNeeded(buf) {
  const u8 = new Uint8Array(buf);
  if (u8[0] === 0x1f && u8[1] === 0x8b) {
    const ds = new DecompressionStream('gzip');
    const stream = new Blob([u8]).stream().pipeThrough(ds);
    return await new Response(stream).arrayBuffer();
  }
  return buf;
}

export async function gzip(u8) {
  const cs = new CompressionStream('gzip');
  const stream = new Blob([u8]).stream().pipeThrough(cs);
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function quatToMat(b, c, d) {
  let a = 1.0 - (b * b + c * c + d * d);
  if (a < 1e-7) { // nibabel fillpositive: normaliza (b,c,d) e usa a = 0
    const n = Math.sqrt(b * b + c * c + d * d); b /= n; c /= n; d /= n; a = 0;
  } else a = Math.sqrt(a);
  return [
    [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
    [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
    [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - c * c - b * b],
  ];
}

export function parseNifti(buf) {
  const dv = new DataView(buf);
  let le = true;
  const sz = dv.getInt32(0, true);
  if (sz !== 348) {
    if (dv.getInt32(0, false) === 348) le = false;
    else if (sz === 540 || dv.getInt32(0, false) === 540) throw new Error('NIfTI-2 ainda não é suportado. Converta para NIfTI-1 (dcm2niix gera NIfTI-1 por padrão).');
    else throw new Error('Arquivo não reconhecido como NIfTI-1 (.nii / .nii.gz).');
  }
  const i16 = (o) => dv.getInt16(o, le), f32 = (o) => dv.getFloat32(o, le);
  const dim = []; for (let i = 0; i < 8; i++) dim.push(i16(40 + 2 * i));
  const datatype = i16(70), bitpix = i16(72);
  const pixdim = []; for (let i = 0; i < 8; i++) pixdim.push(f32(76 + 4 * i));
  const voxOffset = Math.max(352, Math.round(f32(108)));
  let slope = f32(112), inter = f32(116);
  if (!isFinite(slope) || slope === 0) { slope = 1; inter = 0; }
  if (!isFinite(inter)) inter = 0;
  const qformCode = i16(252), sformCode = i16(254);
  const nx = dim[1], ny = Math.max(1, dim[2]), nz = Math.max(1, dim[3]);
  if (dim[0] < 3 || nz < 2) throw new Error('A imagem precisa ser um volume 3D.');
  let affine;
  if (sformCode !== 0) {
    affine = [0, 1, 2].map((r) => [0, 1, 2, 3].map((c) => f32(280 + 16 * r + 4 * c)));
    affine.push([0, 0, 0, 1]);
  } else if (qformCode !== 0) {
    const R = quatToMat(f32(256), f32(260), f32(264));
    let qfac = pixdim[0]; if (qfac !== -1 && qfac !== 1) qfac = 1;
    const vox = [pixdim[1], pixdim[2], pixdim[3] * qfac];
    affine = [0, 1, 2].map((r) => [R[r][0] * vox[0], R[r][1] * vox[1], R[r][2] * vox[2], f32(268 + 4 * r)]);
    affine.push([0, 0, 0, 1]);
  } else {
    const z = [pixdim[1] || 1, pixdim[2] || 1, pixdim[3] || 1];
    const c = [(nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2];
    affine = [[-z[0], 0, 0, z[0] * c[0]], [0, z[1], 0, -z[1] * c[1]], [0, 0, z[2], -z[2] * c[2]], [0, 0, 0, 1]];
  }
  const n = nx * ny * nz;
  const data = new Float32Array(n);
  const off = voxOffset;
  const rd = {
    2: (i) => dv.getUint8(off + i), 256: (i) => dv.getInt8(off + i),
    4: (i) => dv.getInt16(off + 2 * i, le), 512: (i) => dv.getUint16(off + 2 * i, le),
    8: (i) => dv.getInt32(off + 4 * i, le), 768: (i) => dv.getUint32(off + 4 * i, le),
    16: (i) => dv.getFloat32(off + 4 * i, le), 64: (i) => dv.getFloat64(off + 8 * i, le),
  }[datatype];
  if (!rd) throw new Error(`Tipo de dado NIfTI não suportado (datatype ${datatype}).`);
  if (off + n * (bitpix / 8) > buf.byteLength) throw new Error('Arquivo NIfTI truncado.');
  for (let i = 0; i < n; i++) data[i] = rd(i) * slope + inter;
  return { dims: [nx, ny, nz], affine, data, datatype, nVolumes: dim[0] > 3 ? Math.max(1, dim[4]) : 1 };
}

// ---------- escrita ----------
export function zoomsFromAffine(a) {
  return [0, 1, 2].map((c) => Math.hypot(a[0][c], a[1][c], a[2][c]));
}

// mat44_to_quatern (nifti1_io.c): quaternion + qfac a partir da parte rotacional do affine
function affineToQuatern(a) {
  let r11 = a[0][0], r12 = a[0][1], r13 = a[0][2], r21 = a[1][0], r22 = a[1][1], r23 = a[1][2], r31 = a[2][0], r32 = a[2][1], r33 = a[2][2];
  const dx = Math.hypot(r11, r21, r31) || 1, dy = Math.hypot(r12, r22, r32) || 1, dz = Math.hypot(r13, r23, r33) || 1;
  r11 /= dx; r21 /= dx; r31 /= dx; r12 /= dy; r22 /= dy; r32 /= dy; r13 /= dz; r23 /= dz; r33 /= dz;
  const det = r11 * (r22 * r33 - r32 * r23) - r12 * (r21 * r33 - r31 * r23) + r13 * (r21 * r32 - r31 * r22);
  let qfac = 1;
  if (det < 0) { qfac = -1; r13 = -r13; r23 = -r23; r33 = -r33; }
  let qa = r11 + r22 + r33 + 1, qb, qc, qd;
  if (qa > 0.5) {
    qa = 0.5 * Math.sqrt(qa); qb = 0.25 * (r32 - r23) / qa; qc = 0.25 * (r13 - r31) / qa; qd = 0.25 * (r21 - r12) / qa;
  } else {
    const xd = 1 + r11 - (r22 + r33), yd = 1 + r22 - (r11 + r33), zd = 1 + r33 - (r11 + r22);
    if (xd > 1) { qb = 0.5 * Math.sqrt(xd); qc = 0.25 * (r12 + r21) / qb; qd = 0.25 * (r13 + r31) / qb; qa = 0.25 * (r32 - r23) / qb; }
    else if (yd > 1) { qc = 0.5 * Math.sqrt(yd); qb = 0.25 * (r12 + r21) / qc; qd = 0.25 * (r23 + r32) / qc; qa = 0.25 * (r13 - r31) / qc; }
    else { qd = 0.5 * Math.sqrt(zd); qb = 0.25 * (r13 + r31) / qd; qc = 0.25 * (r23 + r32) / qd; qa = 0.25 * (r21 - r12) / qd; }
    if (qa < 0) { qb = -qb; qc = -qc; qd = -qd; }
  }
  return { qb, qc, qd, qfac };
}

// datatype: 2 uint8, 4 int16, 512 uint16, 16 float32. slope/inter vão para scl_slope/scl_inter.
// qform: também grava o qform (código 1) além do sform, como o dcm2niix.
export function writeNifti({ dims, affine, data, datatype = 2, slope = 1, inter = 0, descrip = '', qform = false }) {
  const bytesPer = { 2: 1, 4: 2, 512: 2, 16: 4 }[datatype];
  if (!bytesPer) throw new Error(`writeNifti: datatype ${datatype} não suportado`);
  const n = dims[0] * dims[1] * dims[2];
  const buf = new ArrayBuffer(352 + n * bytesPer);
  const dv = new DataView(buf);
  dv.setInt32(0, 348, true);
  const dim = [3, dims[0], dims[1], dims[2], 1, 1, 1, 1];
  dim.forEach((v, i) => dv.setInt16(40 + 2 * i, v, true));
  dv.setInt16(70, datatype, true);
  dv.setInt16(72, bytesPer * 8, true);
  const z = zoomsFromAffine(affine);
  const q = qform ? affineToQuatern(affine) : null;
  const pix = [q ? q.qfac : 1, z[0], z[1], z[2], 1, 1, 1, 1];
  pix.forEach((v, i) => dv.setFloat32(76 + 4 * i, v, true));
  dv.setFloat32(108, 352, true);
  dv.setFloat32(112, slope, true); dv.setFloat32(116, inter, true);
  dv.setUint8(123, 10); // xyzt_units: mm + s
  for (let i = 0; i < Math.min(79, descrip.length); i++) dv.setUint8(148 + i, descrip.charCodeAt(i) & 0x7f);
  dv.setInt16(252, q ? 1 : 0, true); // qform_code
  dv.setInt16(254, 1, true); // sform_code (scanner)
  if (q) {
    dv.setFloat32(256, q.qb, true); dv.setFloat32(260, q.qc, true); dv.setFloat32(264, q.qd, true);
    dv.setFloat32(268, affine[0][3], true); dv.setFloat32(272, affine[1][3], true); dv.setFloat32(276, affine[2][3], true);
  }
  for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) dv.setFloat32(280 + 16 * r + 4 * c, affine[r][c], true);
  const magic = [0x6e, 0x2b, 0x31, 0x00]; // "n+1\0"
  magic.forEach((v, i) => dv.setUint8(344 + i, v));
  if (datatype === 2) new Uint8Array(buf, 352).set(data); // mesma conversão (ToUint8) que setUint8
  else for (let i = 0; i < n; i++) {
    const o = 352 + i * bytesPer, v = data[i];
    if (datatype === 4) dv.setInt16(o, v, true);
    else if (datatype === 512) dv.setUint16(o, v, true);
    else dv.setFloat32(o, v, true);
  }
  return new Uint8Array(buf);
}
