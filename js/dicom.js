// dicom.js — leitura de DICOM de RM e conversão para NIfTI-1 no navegador, sem dependências.
// Etapa anterior ao pipeline: o usuário escolhe a série T2 3D e ela é empilhada num volume NIfTI
// com a mesma geometria que o dcm2niix produziria (affine em RAS a partir de ImagePositionPatient,
// ImageOrientationPatient e PixelSpacing; intensidades com RescaleSlope/Intercept).
//
// Suporta: DICOM "clássico" (uma fatia por arquivo) e DICOM aprimorado multiframe (Enhanced MR),
// sintaxes não comprimidas (implícita/explícita, little/big endian), deflate, RLE Lossless e
// JPEG Lossless (processo 14). JPEG 2000 / JPEG-LS / JPEG com perdas não são decodificados:
// nesses casos o usuário é orientado a usar dcm2niix.

import { writeNifti, gzip } from './nifti.js';

// ---------------- sintaxes de transferência ----------------
const TS = {
  IMPLICIT_LE: '1.2.840.10008.1.2', EXPLICIT_LE: '1.2.840.10008.1.2.1', DEFLATED: '1.2.840.10008.1.2.1.99',
  EXPLICIT_BE: '1.2.840.10008.1.2.2', RLE: '1.2.840.10008.1.2.5',
  JPEG_LOSSLESS: '1.2.840.10008.1.2.4.57', JPEG_LOSSLESS_SV1: '1.2.840.10008.1.2.4.70',
};
const TS_NAMES = {
  '1.2.840.10008.1.2.4.50': 'JPEG com perdas (baseline)', '1.2.840.10008.1.2.4.51': 'JPEG com perdas (estendido)',
  '1.2.840.10008.1.2.4.80': 'JPEG-LS sem perdas', '1.2.840.10008.1.2.4.81': 'JPEG-LS quase sem perdas',
  '1.2.840.10008.1.2.4.90': 'JPEG 2000 sem perdas', '1.2.840.10008.1.2.4.91': 'JPEG 2000',
  '1.2.840.10008.1.2.4.201': 'HTJ2K sem perdas', '1.2.840.10008.1.2.4.202': 'HTJ2K sem perdas (RPCL)', '1.2.840.10008.1.2.4.203': 'HTJ2K',
};
const SUPPORTED_TS = new Set(Object.values(TS));
export function transferSyntaxName(uid) {
  return { [TS.IMPLICIT_LE]: 'implícita LE', [TS.EXPLICIT_LE]: 'explícita LE', [TS.DEFLATED]: 'deflate', [TS.EXPLICIT_BE]: 'explícita BE',
    [TS.RLE]: 'RLE sem perdas', [TS.JPEG_LOSSLESS]: 'JPEG sem perdas', [TS.JPEG_LOSSLESS_SV1]: 'JPEG sem perdas (SV1)' }[uid] || TS_NAMES[uid] || uid;
}

// ---------------- etiquetas usadas ----------------
const T = (g, e) => ((g << 16) | e) >>> 0;
const TAG = {
  TransferSyntax: T(0x0002, 0x0010), SOPClass: T(0x0008, 0x0016), ImageType: T(0x0008, 0x0008), StudyDate: T(0x0008, 0x0020),
  Modality: T(0x0008, 0x0060), Manufacturer: T(0x0008, 0x0070), StudyDescription: T(0x0008, 0x1030), SeriesDescription: T(0x0008, 0x103e),
  ModelName: T(0x0008, 0x1090), PatientID: T(0x0010, 0x0020), ScanningSequence: T(0x0018, 0x0020), MRAcquisitionType: T(0x0018, 0x0023),
  SequenceName: T(0x0018, 0x0024), SliceThickness: T(0x0018, 0x0050), RepetitionTime: T(0x0018, 0x0080), EchoTime: T(0x0018, 0x0081),
  EchoNumbers: T(0x0018, 0x0086), FieldStrength: T(0x0018, 0x0087), SpacingBetweenSlices: T(0x0018, 0x0088), FlipAngle: T(0x0018, 0x1314),
  ProtocolName: T(0x0018, 0x1030), PulseSequenceName: T(0x0018, 0x9005), EffectiveEchoTime: T(0x0018, 0x9082),
  StudyUID: T(0x0020, 0x000d), SeriesUID: T(0x0020, 0x000e), SeriesNumber: T(0x0020, 0x0011), AcquisitionNumber: T(0x0020, 0x0012),
  InstanceNumber: T(0x0020, 0x0013), IPP: T(0x0020, 0x0032), IOP: T(0x0020, 0x0037), StackID: T(0x0020, 0x9056),
  SamplesPerPixel: T(0x0028, 0x0002), Photometric: T(0x0028, 0x0004), NumberOfFrames: T(0x0028, 0x0008), Rows: T(0x0028, 0x0010),
  Columns: T(0x0028, 0x0011), PixelSpacing: T(0x0028, 0x0030), BitsAllocated: T(0x0028, 0x0100), BitsStored: T(0x0028, 0x0101),
  PixelRepresentation: T(0x0028, 0x0103), RescaleIntercept: T(0x0028, 0x1052), RescaleSlope: T(0x0028, 0x1053),
  SharedFG: T(0x5200, 0x9229), PerFrameFG: T(0x5200, 0x9230), PixelData: T(0x7fe0, 0x0010),
  FrameContentSeq: T(0x0020, 0x9111), PlanePositionSeq: T(0x0020, 0x9113), PlaneOrientationSeq: T(0x0020, 0x9116),
  PixelMeasuresSeq: T(0x0028, 0x9110), PixelValueTransfSeq: T(0x0028, 0x9145), MREchoSeq: T(0x0018, 0x9114),
  MRTimingSeq: T(0x0018, 0x9112), FrameTypeSeq: T(0x0018, 0x9226),
  Item: T(0xfffe, 0xe000), ItemDelim: T(0xfffe, 0xe00d), SeqDelim: T(0xfffe, 0xe0dd),
};
// sequências reconhecidas mesmo em VR implícita
const KNOWN_SQ = new Set([TAG.SharedFG, TAG.PerFrameFG, TAG.FrameContentSeq, TAG.PlanePositionSeq, TAG.PlaneOrientationSeq,
  TAG.PixelMeasuresSeq, TAG.PixelValueTransfSeq, TAG.MREchoSeq, TAG.MRTimingSeq, TAG.FrameTypeSeq]);
const LONG_VR = new Set(['OB', 'OD', 'OF', 'OL', 'OV', 'OW', 'SQ', 'SV', 'UC', 'UN', 'UR', 'UT', 'UV']);
const UNDEF = 0xffffffff;

class Truncated extends Error {}

// ---------------- leitor de conjunto de dados ----------------
function parseDataset(dv, start, end, le, explicit, stopAtPixels, out) {
  const ds = new Map();
  let p = start;
  const u16 = (o) => dv.getUint16(o, le), u32 = (o) => dv.getUint32(o, le);
  while (p < end) {
    if (p + 8 > dv.byteLength) { if (end === dv.byteLength) break; throw new Truncated(); } // bytes de preenchimento no fim
    const group = u16(p), elem = u16(p + 2), tag = T(group, elem);
    if (tag === TAG.ItemDelim) { out.end = p + 8; return ds; }
    if (tag === TAG.SeqDelim) { out.end = p + 8; return ds; }
    let vr = null, len, h;
    if (explicit && group !== 0xfffe) {
      vr = String.fromCharCode(dv.getUint8(p + 4), dv.getUint8(p + 5));
      if (LONG_VR.has(vr)) { if (p + 12 > dv.byteLength) throw new Truncated(); len = u32(p + 8); h = 12; } else { len = u16(p + 6); h = 8; }
    } else { len = u32(p + 4); h = 8; }
    const vp = p + h;
    if (tag === TAG.PixelData) {
      ds.set(tag, { vr, off: vp, len, encapsulated: len === UNDEF });
      out.pixelTagFound = true;
      if (stopAtPixels) { out.end = vp; return ds; }
      if (len === UNDEF) { const r = parseFragments(dv, vp, le); ds.get(tag).fragments = r.fragments; ds.get(tag).bot = r.bot; p = r.end; }
      else { if (vp + len > dv.byteLength) throw new Truncated(); p = vp + len; }
      continue;
    }
    const isSQ = vr === 'SQ' || (len === UNDEF) || (!explicit && KNOWN_SQ.has(tag));
    if (isSQ) {
      // UN de comprimento indefinido é codificado em VR implícita (PS3.5 6.2.2)
      const itemsExplicit = vr === 'UN' ? false : explicit;
      const items = [];
      let q = vp; const qEnd = len === UNDEF ? Infinity : vp + len;
      while (q < qEnd) {
        if (q + 8 > dv.byteLength) throw new Truncated();
        const it = T(u16(q), u16(q + 2)), il = u32(q + 4);
        if (it === TAG.SeqDelim) { q += 8; break; }
        if (it !== TAG.Item) throw new Error('Sequência DICOM malformada.');
        const o = {};
        const sub = parseDataset(dv, q + 8, il === UNDEF ? Infinity : q + 8 + il, le, itemsExplicit, false, o);
        items.push(sub);
        q = il === UNDEF ? o.end : q + 8 + il;
      }
      ds.set(tag, { vr: 'SQ', items });
      p = len === UNDEF ? q : vp + len;
      continue;
    }
    if (vp + len > dv.byteLength) throw new Truncated();
    ds.set(tag, { vr, off: vp, len });
    p = vp + len;
  }
  out.end = p;
  return ds;
}

function parseFragments(dv, p, le) {
  const u16 = (o) => dv.getUint16(o, le), u32 = (o) => dv.getUint32(o, le);
  const frags = [];
  let bot = null;
  for (;;) {
    if (p + 8 > dv.byteLength) throw new Truncated();
    const tag = T(u16(p), u16(p + 2)), len = u32(p + 4);
    if (tag === TAG.SeqDelim) return { fragments: frags, bot, end: p + 8 };
    if (tag !== TAG.Item) throw new Error('Pixel data encapsulado malformado.');
    if (p + 8 + len > dv.byteLength) throw new Truncated();
    if (bot === null) { bot = []; for (let i = 0; i < len; i += 4) bot.push(u32(p + 8 + i)); }
    else frags.push({ off: p + 8, len, rel: p });
    p += 8 + len;
  }
}

async function inflateRaw(u8) {
  const ds = new DecompressionStream('deflate-raw');
  return new Uint8Array(await new Response(new Blob([u8]).stream().pipeThrough(ds)).arrayBuffer());
}

// Lê o arquivo (ArrayBuffer) e devolve { ds, dv, le, ts }. stopAtPixels: só o cabeçalho.
export async function readDicom(buf, { stopAtPixels = false } = {}) {
  let dv = new DataView(buf);
  let p = 0, ts = TS.IMPLICIT_LE;
  const hasPreamble = buf.byteLength >= 132 && dv.getUint32(128, false) === 0x4449434d; // "DICM"
  if (hasPreamble) {
    p = 132;
    const o = {};
    // grupo 0002: sempre VR explícita little endian
    let metaEnd = p;
    while (metaEnd + 8 <= buf.byteLength && dv.getUint16(metaEnd, true) === 0x0002) {
      const vr = String.fromCharCode(dv.getUint8(metaEnd + 4), dv.getUint8(metaEnd + 5));
      const len = LONG_VR.has(vr) ? dv.getUint32(metaEnd + 8, true) : dv.getUint16(metaEnd + 6, true);
      metaEnd += (LONG_VR.has(vr) ? 12 : 8) + len;
    }
    const meta = parseDataset(dv, p, metaEnd, true, true, false, o);
    const t = meta.get(TAG.TransferSyntax);
    if (t) ts = str(dv, t);
    p = metaEnd;
  } else {
    // sem preâmbulo: aceita apenas se começar com um grupo plausível
    if (buf.byteLength < 16) throw new Error('não é DICOM');
    const g = dv.getUint16(0, true);
    if (g !== 0x0008 && g !== 0x0002) throw new Error('não é DICOM');
    const vr = String.fromCharCode(dv.getUint8(4), dv.getUint8(5));
    ts = /^[A-Z]{2}$/.test(vr) ? TS.EXPLICIT_LE : TS.IMPLICIT_LE;
  }
  let le = ts !== TS.EXPLICIT_BE, explicit = ts !== TS.IMPLICIT_LE;
  if (ts === TS.DEFLATED) {
    const inflated = await inflateRaw(new Uint8Array(buf, p));
    buf = inflated.buffer; dv = new DataView(buf); p = 0;
  }
  const out = {};
  const ds = parseDataset(dv, p, dv.byteLength, le, explicit, stopAtPixels, out);
  return { ds, dv, le, ts, pixelTagFound: !!out.pixelTagFound };
}

// ---------------- leitura de valores ----------------
function str(dv, el) {
  if (!el || el.len === undefined || el.len === UNDEF) return '';
  let s = '';
  for (let i = 0; i < el.len; i++) { const c = dv.getUint8(el.off + i); if (c === 0) break; s += String.fromCharCode(c); }
  try { s = decodeURIComponent(escape(s)); } catch {} // UTF-8 quando aplicável
  return s.trim();
}
function nums(dv, el, le) { // valores numéricos (binários ou texto DS/IS)
  if (!el || el.len === undefined) return [];
  const vr = el.vr;
  const bin = { US: [2, 'getUint16'], SS: [2, 'getInt16'], UL: [4, 'getUint32'], SL: [4, 'getInt32'], FL: [4, 'getFloat32'], FD: [8, 'getFloat64'] }[vr];
  if (bin) { const r = []; for (let i = 0; i + bin[0] <= el.len; i += bin[0]) r.push(dv[bin[1]](el.off + i, le)); return r; }
  return str(dv, el).split('\\').map((x) => parseFloat(x)).filter((x) => Number.isFinite(x));
}
// tags US (Rows, Columns, BitsAllocated…): em VR implícita chegam sem VR, então lê pelo tamanho
function usVal(dv, el, le) {
  if (!el || !el.len) return undefined;
  if (el.vr && el.vr !== 'US' && el.vr !== 'UL') return nums(dv, el, le)[0];
  if (el.len === 2) return dv.getUint16(el.off, le);
  if (el.len === 4) return dv.getUint32(el.off, le);
  return undefined;
}
const first = (seqEl) => seqEl?.items?.[0];

// ---------------- varredura de arquivos → fatias ----------------
// Cada "fatia" é um quadro 2D com geometria própria (arquivo + índice do quadro).
export async function scanFile(file) {
  const HEAD = 256 * 1024;
  let buf = await file.slice(0, Math.min(file.size, HEAD)).arrayBuffer();
  let r;
  try { r = await readDicom(buf, { stopAtPixels: true }); if (!r.pixelTagFound && file.size > HEAD) throw new Truncated(); }
  catch (e) {
    // leitura parcial falhou (cabeçalho longo, deflate…): tenta o arquivo inteiro
    if (file.size <= HEAD) throw e instanceof Truncated ? new Error('arquivo DICOM truncado') : e;
    buf = await file.arrayBuffer(); r = await readDicom(buf, { stopAtPixels: true });
  }
  const { ds, dv, le, ts } = r;
  const S = (t) => str(dv, ds.get(t)), N = (t) => nums(dv, ds.get(t), le), U = (t) => usVal(dv, ds.get(t), le), I = (t) => N(t)[0]; // I: IS (texto)
  const sop = S(TAG.SOPClass);
  if (sop === '1.2.840.10008.1.3.10') return []; // DICOMDIR
  if (!ds.has(TAG.PixelData) && !r.pixelTagFound) return []; // relatório estruturado, etc.
  const rows = U(TAG.Rows), cols = U(TAG.Columns);
  if (!rows || !cols) return [];
  const common = {
    file, ts, studyUID: S(TAG.StudyUID), seriesUID: S(TAG.SeriesUID), seriesNumber: I(TAG.SeriesNumber),
    seriesDescription: S(TAG.SeriesDescription), protocolName: S(TAG.ProtocolName), studyDescription: S(TAG.StudyDescription),
    studyDate: S(TAG.StudyDate), patientID: S(TAG.PatientID), modality: S(TAG.Modality), manufacturer: S(TAG.Manufacturer),
    model: S(TAG.ModelName), fieldStrength: N(TAG.FieldStrength)[0], imageType: S(TAG.ImageType),
    mrAcqType: S(TAG.MRAcquisitionType), sequenceName: S(TAG.SequenceName) || S(TAG.PulseSequenceName), scanningSequence: S(TAG.ScanningSequence),
    rows, cols, bitsAllocated: U(TAG.BitsAllocated) || 16, bitsStored: U(TAG.BitsStored) || U(TAG.BitsAllocated) || 16, pixelRepresentation: U(TAG.PixelRepresentation) || 0,
    samplesPerPixel: U(TAG.SamplesPerPixel) || 1, photometric: S(TAG.Photometric), sopClass: sop,
    sliceThickness: N(TAG.SliceThickness)[0], spacingBetween: N(TAG.SpacingBetweenSlices)[0], flipAngle: N(TAG.FlipAngle)[0],
    tr: N(TAG.RepetitionTime)[0], te: N(TAG.EchoTime)[0],
  };
  const nFrames = Math.max(1, I(TAG.NumberOfFrames) || 1);
  const perFrame = ds.get(TAG.PerFrameFG)?.items;
  const shared = first(ds.get(TAG.SharedFG));
  const slices = [];
  if (perFrame || shared) { // Enhanced MR multiframe
    const fg = (item, seqTag) => first(item?.get(seqTag));
    const pick = (f, seqTag, tag) => { const a = fg(f, seqTag)?.get(tag) ?? fg(shared, seqTag)?.get(tag); return a; };
    for (let k = 0; k < nFrames; k++) {
      const f = perFrame?.[k];
      const ipp = nums(dv, pick(f, TAG.PlanePositionSeq, TAG.IPP), le);
      const iop = nums(dv, pick(f, TAG.PlaneOrientationSeq, TAG.IOP), le);
      const ps = nums(dv, pick(f, TAG.PixelMeasuresSeq, TAG.PixelSpacing), le);
      const th = nums(dv, pick(f, TAG.PixelMeasuresSeq, TAG.SliceThickness), le)[0];
      const slope = nums(dv, pick(f, TAG.PixelValueTransfSeq, TAG.RescaleSlope), le)[0];
      const inter = nums(dv, pick(f, TAG.PixelValueTransfSeq, TAG.RescaleIntercept), le)[0];
      const te = nums(dv, pick(f, TAG.MREchoSeq, TAG.EffectiveEchoTime), le)[0];
      const tr = nums(dv, pick(f, TAG.MRTimingSeq, TAG.RepetitionTime), le)[0];
      const stack = str(dv, fg(f, TAG.FrameContentSeq)?.get(TAG.StackID));
      const ftype = str(dv, pick(f, TAG.FrameTypeSeq, T(0x0008, 0x9007)));
      slices.push({ ...common, frame: k, nFrames, ipp, iop, ps, slope: slope ?? 1, inter: inter ?? 0, te: te ?? common.te, tr: tr ?? common.tr,
        sliceThickness: th ?? common.sliceThickness, echo: [stack && `pilha ${stack}`, te !== undefined && `TE ${te}`, ftype].filter(Boolean).join(' '),
        imageType: ftype || common.imageType, instance: k + 1 });
    }
  } else {
    const slope = N(TAG.RescaleSlope)[0], inter = N(TAG.RescaleIntercept)[0];
    for (let k = 0; k < nFrames; k++) { // multiframe "antigo": geometria única por arquivo
      slices.push({ ...common, frame: k, nFrames, ipp: N(TAG.IPP), iop: N(TAG.IOP), ps: N(TAG.PixelSpacing), slope: slope ?? 1, inter: inter ?? 0,
        echo: String(I(TAG.EchoNumbers) ?? ''), instance: I(TAG.InstanceNumber) ?? 0, acq: I(TAG.AcquisitionNumber) });
    }
    if (nFrames > 1 && common.spacingBetween) { // posições ao longo da normal
      const s0 = slices[0];
      if (s0.iop.length === 6 && s0.ipp.length === 3) {
        const n = cross(s0.iop.slice(0, 3), s0.iop.slice(3, 6));
        slices.forEach((s, k) => { s.ipp = s0.ipp.map((v, i) => v + k * common.spacingBetween * n[i]); });
      }
    }
  }
  return slices;
}

// ---------------- agrupamento em séries/volumes ----------------
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : NaN; };

export function planeName(iop) {
  if (!iop || iop.length !== 6) return '?';
  const n = cross(iop.slice(0, 3), iop.slice(3, 6)).map(Math.abs);
  const i = n.indexOf(Math.max(...n));
  const name = ['sagital', 'coronal', 'axial'][i];
  return n[i] < 0.95 ? `${name} oblíquo` : name;
}

// Agrupa fatias por estudo e por série/volume. Devolve [{ studyUID, ..., series: [cand] }]
export function groupSeries(slices) {
  const studies = new Map();
  for (const s of slices) {
    if (s.samplesPerPixel !== 1) continue;
    const iopKey = s.iop.length === 6 ? s.iop.map((v) => v.toFixed(3)).join(',') : 'sem-orientacao';
    const key = [s.seriesUID, iopKey, s.rows, s.cols, s.echo].join('|');
    if (!studies.has(s.studyUID)) studies.set(s.studyUID, { studyUID: s.studyUID, studyDate: s.studyDate, studyDescription: s.studyDescription, patientID: s.patientID, groups: new Map(), dir: null });
    const st = studies.get(s.studyUID);
    st.dir = commonPrefix(st.dir, dirOf(s.file));
    if (!st.groups.has(key)) st.groups.set(key, []);
    st.groups.get(key).push(s);
  }
  const out = [];
  for (const st of studies.values()) {
    const series = [...st.groups.values()].map(describeCandidate);
    series.sort((a, b) => (a.seriesNumber ?? 0) - (b.seriesNumber ?? 0) || a.label.localeCompare(b.label));
    const best = series.filter((c) => c.ok).sort((a, b) => b.score - a.score)[0];
    if (best && best.score >= 4) best.suggested = true;
    out.push({ studyUID: st.studyUID, studyDate: st.studyDate, studyDescription: st.studyDescription, patientID: st.patientID, dir: st.dir || [], series });
  }
  suggestIds(out);
  return out;
}
const dirOf = (file) => (file.webkitRelativePath || file.relPath || '').split('/').slice(0, -1);
function commonPrefix(a, b) {
  if (a === null) return b;
  let i = 0; while (i < a.length && i < b.length && a[i] === b[i]) i++;
  return a.slice(0, i);
}
// ID sugerido: pasta própria do estudo (descontando a pasta-mãe comum a todos), senão PatientID
const GENERIC_DIR = /^(dicom|dcm|images?|imagens|series|study|exam[e]?|data|dados|s\d+|se\d+|st\d+|\d+|[0-9a-f]{8,})$/i;
function suggestIds(studies) {
  const global = studies.reduce((g, st) => commonPrefix(g, st.dir), null) || [];
  studies.forEach((st, i) => {
    let cand = studies.length > 1 && st.dir.length > global.length ? [st.dir[global.length]] : [];
    cand = cand.concat([...st.dir].reverse());
    const name = cand.find((d) => d && !GENERIC_DIR.test(d));
    st.suggestedId = name || st.patientID || `sujeito_${i + 1}`;
  });
}

function describeCandidate(sl) {
  const s0 = sl[0];
  const label = s0.seriesDescription || s0.protocolName || `Série ${s0.seriesNumber ?? '?'}`;
  const c = { slices: sl, seriesUID: s0.seriesUID, seriesNumber: s0.seriesNumber, label, protocolName: s0.protocolName, n: sl.length,
    rows: s0.rows, cols: s0.cols, plane: planeName(s0.iop), mrAcqType: s0.mrAcqType, te: s0.te, tr: s0.tr, imageType: s0.imageType,
    sequenceName: s0.sequenceName, ts: s0.ts, manufacturer: s0.manufacturer, model: s0.model, fieldStrength: s0.fieldStrength, modality: s0.modality,
    echo: s0.echo, problems: [], warnings: [] };
  let zSpacing = NaN;
  if (s0.iop.length !== 6 || s0.ps.length < 2 || sl.some((s) => s.ipp.length !== 3)) c.problems.push('sem geometria (posição/orientação)');
  else {
    const n = cross(s0.iop.slice(0, 3), s0.iop.slice(3, 6));
    const d = sl.map((s) => dot(s.ipp, n)).sort((a, b) => a - b);
    const gaps = d.slice(1).map((v, i) => v - d[i]);
    const dup = gaps.filter((g) => Math.abs(g) < 1e-3).length;
    if (dup) c.problems.push(`${dup + 1 > sl.length ? sl.length : dup} posições repetidas (vários volumes na mesma série)`);
    zSpacing = median(gaps.filter((g) => Math.abs(g) >= 1e-3));
    if (!dup && gaps.length && gaps.some((g) => g > 1.5 * zSpacing)) c.problems.push('faltam fatias (intervalo irregular)');
    else if (!dup && gaps.some((g) => Math.abs(g - zSpacing) > 0.01 * zSpacing + 1e-3)) c.warnings.push('espaçamento entre fatias levemente irregular');
  }
  if (sl.length < 3) c.problems.push('menos de 3 imagens');
  if (s0.modality && s0.modality !== 'MR') c.warnings.push(`modalidade ${s0.modality}`);
  if (!SUPPORTED_TS.has(s0.ts)) c.problems.push(`compressão ${transferSyntaxName(s0.ts)} não suportada no navegador`);
  if (s0.photometric === 'MONOCHROME1') c.warnings.push('MONOCHROME1 (escala de cinza invertida)');
  if (s0.bitsAllocated !== 8 && s0.bitsAllocated !== 16 && s0.bitsAllocated !== 32) c.problems.push(`${s0.bitsAllocated} bits por pixel`);
  c.voxel = [s0.ps[1], s0.ps[0], zSpacing];
  c.ok = c.problems.length === 0;
  c.score = scoreT2_3D(c);
  return c;
}

// heurística para sugerir a série T2 3D isotrópica (SPACE, CUBE, VISTA, DRIVE, CISS, FIESTA…)
function scoreT2_3D(c) {
  const txt = `${c.label} ${c.protocolName} ${c.sequenceName}`.toLowerCase();
  let s = 0;
  if (/3d/.test(c.mrAcqType?.toLowerCase())) s += 3;
  if (/(^|[^a-z0-9])t2|spc|space|cube|vista|drive|ciss|fiesta|3d.?tse|tse.?3d|t2w/.test(txt)) s += 3;
  if (/flair|dark.?fluid|t1|mprage|mp2rage|tfl|bravo|spgr|tfe|dwi|dti|diff|ep2d|swi|susc|tof|angio|loc|scout|survey|haste|ssfse|calib|ref|perf|bold/.test(txt)) s -= 4;
  const v = c.voxel.filter(Number.isFinite);
  if (v.length === 3) {
    if (Math.max(...v) <= 0.9) s += 3; else if (Math.max(...v) <= 1.1) s += 1;
    if (Math.max(...v) / Math.min(...v) <= 1.2) s += 1;
  }
  if (c.n >= 60) s += 1;
  if (/derived|mpr|reformat|projection/i.test(c.imageType)) s -= 2;
  return s;
}

export function candidateWarnings(c) {
  const w = [...c.warnings];
  const v = c.voxel;
  if (v.every(Number.isFinite)) {
    if (Math.max(...v) / Math.min(...v) > 1.5) w.push('voxels anisotrópicos (aquisição 2D?) — o método automático exige T2 3D isotrópico');
    else if (v.some((z) => z > 0.9 || z < 0.6)) w.push('fora de 0,7–0,8 mm: será reamostrado, mas o método não foi validado nessa resolução');
  }
  if (c.mrAcqType && c.mrAcqType !== '3D') w.push(`aquisição ${c.mrAcqType}`);
  if (/derived|mpr/i.test(c.imageType)) w.push('imagem derivada/reformatada (MPR): prefira a série original');
  return w;
}

// ---------------- decodificação de pixels ----------------
function frameBytes(r, el, frame, nFrames) {
  const u8 = new Uint8Array(r.dv.buffer, r.dv.byteOffset);
  if (!el.encapsulated) return null;
  const fr = el.fragments;
  if (nFrames === 1) {
    if (fr.length === 1) return u8.subarray(fr[0].off, fr[0].off + fr[0].len);
    return concat(fr.map((f) => u8.subarray(f.off, f.off + f.len)));
  }
  if (el.bot && el.bot.length === nFrames) { // tabela de deslocamentos
    const base = fr.length ? fr[0].rel : 0;
    const start = base + el.bot[frame], stop = frame + 1 < nFrames ? base + el.bot[frame + 1] : Infinity;
    return concat(fr.filter((f) => f.rel >= start && f.rel < stop).map((f) => u8.subarray(f.off, f.off + f.len)));
  }
  if (fr.length === nFrames) return u8.subarray(fr[frame].off, fr[frame].off + fr[frame].len);
  // sem tabela: separa quadros JPEG pelo marcador SOI
  const groups = [];
  for (const f of fr) {
    const isStart = u8[f.off] === 0xff && u8[f.off + 1] === 0xd8;
    if (isStart || !groups.length) groups.push([]);
    groups[groups.length - 1].push(u8.subarray(f.off, f.off + f.len));
  }
  if (groups.length !== nFrames) throw new Error('Não foi possível separar os quadros do pixel data encapsulado.');
  return concat(groups[frame]);
}
function concat(parts) {
  if (parts.length === 1) return parts[0];
  const n = parts.reduce((a, p) => a + p.length, 0), o = new Uint8Array(n);
  let k = 0; for (const p of parts) { o.set(p, k); k += p.length; }
  return o;
}

// valores brutos do quadro (antes de RescaleSlope/Intercept) em Float32Array
function decodeFrame(r, s) {
  const el = r.ds.get(TAG.PixelData);
  if (!el) throw new Error('Arquivo sem pixel data.');
  const n = s.rows * s.cols, bpp = s.bitsAllocated / 8, signed = s.pixelRepresentation === 1;
  const out = new Float32Array(n);
  if (!el.encapsulated) {
    const off = el.off + s.frame * n * bpp;
    if (off + n * bpp > r.dv.byteLength) throw new Error('Pixel data truncado.');
    const dv = r.dv, le = r.le;
    if (bpp === 2) for (let i = 0; i < n; i++) out[i] = signed ? dv.getInt16(off + 2 * i, le) : dv.getUint16(off + 2 * i, le);
    else if (bpp === 1) for (let i = 0; i < n; i++) out[i] = signed ? dv.getInt8(off + i) : dv.getUint8(off + i);
    else for (let i = 0; i < n; i++) out[i] = signed ? dv.getInt32(off + 4 * i, le) : dv.getUint32(off + 4 * i, le);
  } else {
    const bytes = frameBytes(r, el, s.frame, s.nFrames);
    let raw, bits = s.bitsAllocated;
    if (r.ts === TS.RLE) raw = decodeRLE(bytes, n, bpp);
    else if (r.ts === TS.JPEG_LOSSLESS || r.ts === TS.JPEG_LOSSLESS_SV1) { const j = decodeJpegLossless(bytes); raw = j.data; bits = j.precision; }
    else throw new Error(`Compressão ${transferSyntaxName(r.ts)} não suportada.`);
    if (raw.length < n) throw new Error('Quadro comprimido com tamanho inesperado.');
    for (let i = 0; i < n; i++) {
      let v = raw[i];
      if (signed) { const m = 2 ** bits; if (v >= m / 2) v -= m; }
      out[i] = v;
    }
  }
  return out;
}

// RLE Lossless (PS3.5 anexo G): um segmento por byte da amostra, do mais significativo ao menos
function decodeRLE(bytes, n, bpp) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const nSeg = dv.getUint32(0, true);
  if (nSeg !== bpp) throw new Error(`RLE com ${nSeg} segmentos não suportado.`);
  const out = bpp === 1 ? new Uint8Array(n) : bpp === 2 ? new Uint16Array(n) : new Uint32Array(n);
  for (let sgi = 0; sgi < nSeg; sgi++) {
    const start = dv.getUint32(4 + 4 * sgi, true), end = sgi + 1 < nSeg ? dv.getUint32(8 + 4 * sgi, true) : bytes.length;
    const shift = 8 * (bpp - 1 - sgi);
    let p = start, k = 0;
    while (k < n && p < end) {
      const h = (bytes[p++] << 24) >> 24;
      if (h >= 0) { for (let i = 0; i <= h; i++) { const b = bytes[p++]; if (k < n) out[k++] |= b << shift; } }
      else if (h !== -128) { const b = bytes[p++]; for (let i = 0; i < 1 - h && k < n; i++) out[k++] |= b << shift; }
    }
  }
  return out;
}

// JPEG sem perdas, processo 14 (ITU T.81 anexo H) — 1 componente, preditores 1–7, transformação de ponto, reinícios
export function decodeJpegLossless(bytes) {
  let p = 0;
  const u16 = (o) => (bytes[o] << 8) | bytes[o + 1];
  const huff = [];
  let P = 0, Y = 0, X = 0, Nf = 0, Ri = 0, compId = 0;
  if (u16(0) !== 0xffd8) throw new Error('JPEG sem marcador SOI.');
  p = 2;
  for (;;) {
    while (bytes[p] !== 0xff && p < bytes.length) p++;
    while (bytes[p] === 0xff) p++;
    const m = bytes[p++];
    if (p >= bytes.length) throw new Error('JPEG truncado.');
    if (m === 0xd8 || (m >= 0xd0 && m <= 0xd7) || m === 0x01) continue;
    const len = u16(p), seg = p + 2, next = p + len;
    if (m === 0xc4) { // DHT
      let q = seg;
      while (q < next) {
        const th = bytes[q] & 0x0f; q++;
        const counts = bytes.subarray(q, q + 16); q += 16;
        const total = counts.reduce((a, b) => a + b, 0);
        const sym = bytes.subarray(q, q + total); q += total;
        huff[th] = buildHuffman(counts, sym);
      }
    } else if (m === 0xc3 || m === 0xc7 || m === 0xcb || m === 0xcf) {
      if (m !== 0xc3) throw new Error('JPEG hierárquico/aritmético não suportado.');
      P = bytes[seg]; Y = u16(seg + 1); X = u16(seg + 3); Nf = bytes[seg + 5]; compId = bytes[seg + 6];
      if (Nf !== 1) throw new Error('JPEG sem perdas com mais de um componente não suportado.');
    } else if (m >= 0xc0 && m <= 0xcf && m !== 0xc4 && m !== 0xc8 && m !== 0xcc) {
      throw new Error('JPEG com perdas: não suportado (use dcm2niix).');
    } else if (m === 0xdd) { Ri = u16(seg); }
    else if (m === 0xda) { // SOS
      const Ns = bytes[seg];
      if (Ns !== 1) throw new Error('JPEG intercalado não suportado.');
      if (bytes[seg + 1] !== compId) throw new Error('JPEG: componente da varredura não corresponde ao quadro.');
      const td = bytes[seg + 2] >> 4;
      const Ss = bytes[seg + 1 + 2 * Ns], Al = bytes[seg + 3 + 2 * Ns] & 0x0f;
      const out = decodeScan(bytes, next, X, Y, P, Ss, Al, huff[td], Ri);
      return { data: out, width: X, height: Y, precision: P };
    } else if (m === 0xd9) throw new Error('JPEG sem varredura.');
    p = next;
  }
}
function buildHuffman(counts, sym) {
  // tabela canônica: maxcode/valptr/mincode por comprimento
  const maxcode = new Int32Array(18).fill(-1), valptr = new Int32Array(17), mincode = new Int32Array(17);
  let code = 0, k = 0;
  for (let l = 1; l <= 16; l++) {
    valptr[l] = k; mincode[l] = code;
    code += counts[l - 1]; k += counts[l - 1];
    maxcode[l] = counts[l - 1] ? code - 1 : -1;
    code <<= 1;
  }
  maxcode[17] = 0x7fffffff;
  // tabela rápida de 8 bits
  const fast = new Int32Array(256).fill(-1);
  code = 0; k = 0;
  for (let l = 1; l <= 8; l++) {
    for (let i = 0; i < counts[l - 1]; i++, k++, code++) {
      const shift = 8 - l;
      for (let j = 0; j < (1 << shift); j++) fast[(code << shift) | j] = (sym[k] << 8) | l;
    }
    code <<= 1;
  }
  return { maxcode, valptr, mincode, sym: Uint8Array.from(sym), fast };
}
function decodeScan(bytes, p, X, Y, P, Ss, Al, H, Ri) {
  if (!H) throw new Error('Tabela de Huffman ausente.');
  const out = new Uint32Array(X * Y);
  const mask = P >= 16 ? 0xffff : (1 << P) - 1;
  const mod = P >= 16 ? 0x10000 : 1 << P;
  let buf = 0, bits = 0, hitMarker = false;
  const fill = () => {
    while (bits <= 24) {
      let b = 0;
      if (!hitMarker && p < bytes.length) {
        b = bytes[p];
        if (b === 0xff) {
          const n = bytes[p + 1];
          if (n === 0x00) p += 2; else { hitMarker = true; b = 0; }
        } else p++;
      }
      buf = (buf << 8) | b; bits += 8;
      buf >>>= 0;
    }
  };
  const getBits = (n) => { if (!n) return 0; if (bits < n) fill(); bits -= n; return (buf >>> bits) & ((1 << n) - 1); };
  const decodeSym = () => {
    if (bits < 16) fill();
    const peek = (buf >>> (bits - 8)) & 0xff;
    const f = H.fast[peek];
    if (f >= 0) { bits -= f & 0xff; return f >> 8; }
    let code = (buf >>> (bits - 9)) & 0x1ff, l = 9;
    while (code > H.maxcode[l]) { l++; if (l > 16) throw new Error('Código de Huffman inválido.'); code = (buf >>> (bits - l)) & ((1 << l) - 1); }
    bits -= l;
    return H.sym[H.valptr[l] + code - H.mincode[l]];
  };
  const diff = () => {
    const t = decodeSym();
    if (t === 0) return 0;
    if (t === 16) return 32768;
    let v = getBits(t);
    if (v < (1 << (t - 1))) v -= (1 << t) - 1;
    return v;
  };
  const restart = () => { // alinha ao byte e consome o marcador RSTn
    bits = 0; buf = 0;
    while (p < bytes.length && !(bytes[p] === 0xff && bytes[p + 1] >= 0xd0 && bytes[p + 1] <= 0xd7)) p++;
    p += 2; hitMarker = false;
  };
  const init = 1 << (P - Al - 1);
  // T.81 H.1.2.1: no início da varredura e de cada intervalo de reinício (que contém linhas inteiras),
  // a primeira amostra usa 2^(P-Pt-1) e o restante dessa linha usa Ra; o início das demais linhas usa Rb.
  let count = 0, firstRow = 0;
  for (let y = 0; y < Y; y++) {
    for (let x = 0; x < X; x++) {
      if (Ri && count === Ri) { restart(); count = 0; firstRow = y; }
      const i = y * X + x;
      let pred;
      if (y === firstRow) pred = x === 0 ? init : out[i - 1];
      else if (x === 0) pred = out[i - X];
      else {
        const Ra = out[i - 1], Rb = out[i - X], Rc = out[i - X - 1];
        switch (Ss) {
          case 1: pred = Ra; break; case 2: pred = Rb; break; case 3: pred = Rc; break;
          case 4: pred = Ra + Rb - Rc; break; case 5: pred = Ra + ((Rb - Rc) >> 1); break;
          case 6: pred = Rb + ((Ra - Rc) >> 1); break; case 7: pred = (Ra + Rb) >> 1; break;
          default: pred = Ra;
        }
      }
      out[i] = (((pred + diff()) % mod) + mod) & mask;
      count++;
    }
  }
  if (Al) for (let i = 0; i < out.length; i++) out[i] <<= Al;
  return out;
}

// ---------------- conversão para NIfTI ----------------
// cand: elemento de groupSeries().series. Devolve { bytes (NIfTI .nii.gz), dims, affine, zooms, meta, warnings }.
export async function convertToNifti(cand, onProgress) {
  const sl = cand.slices;
  const s0 = sl[0];
  const r = s0.iop.slice(0, 3), c = s0.iop.slice(3, 6), n = cross(r, c);
  const sorted = [...sl].sort((a, b) => dot(a.ipp, n) - dot(b.ipp, n));
  const nx = s0.cols, ny = s0.rows, nz = sorted.length, nvox = nx * ny;
  const data = new Float32Array(nvox * nz);
  const slopes = new Set(sorted.map((s) => `${s.slope}|${s.inter}`));
  const byFile = new Map();
  sorted.forEach((s, k) => { if (!byFile.has(s.file)) byFile.set(s.file, []); byFile.get(s.file).push([s, k]); });
  let done = 0;
  for (const [file, list] of byFile) {
    const rd = await readDicom(await file.arrayBuffer());
    for (const [s, k] of list) {
      if (s.rows !== ny || s.cols !== nx) throw new Error('Imagens com matrizes diferentes na mesma série.');
      const px = decodeFrame(rd, s);
      const off = k * nvox;
      if (slopes.size === 1) data.set(px, off);
      else for (let i = 0; i < nvox; i++) data[off + i] = px[i] * s.slope + s.inter;
      done++;
    }
    if (onProgress && (done % 16 === 0 || done === nz)) { onProgress(done / nz); await new Promise((res) => setTimeout(res, 0)); }
  }
  // affine: voxel (i=coluna, j=linha, k=fatia) -> LPS (DICOM) -> RAS (NIfTI)
  const dc = s0.ps[1], dr = s0.ps[0];
  const pFirst = sorted[0].ipp, pLast = sorted[nz - 1].ipp;
  const sv = nz > 1 ? pLast.map((v, i) => (v - pFirst[i]) / (nz - 1)) : n.map((v) => v * (s0.sliceThickness || 1));
  const lps = [0, 1, 2].map((i) => [r[i] * dc, c[i] * dr, sv[i], pFirst[i]]);
  const affine = [[-lps[0][0], -lps[0][1], -lps[0][2], -lps[0][3]], [-lps[1][0], -lps[1][1], -lps[1][2], -lps[1][3]], lps[2], [0, 0, 0, 1]];
  // tipo de dado: inteiros brutos + scl_slope/inter quando possível (igual ao dcm2niix), senão float32
  let datatype = 16, slope = 1, inter = 0, store = data;
  if (slopes.size === 1) {
    let mn = Infinity, mx = -Infinity, allInt = true;
    for (let i = 0; i < data.length; i++) { const v = data[i]; if (v < mn) mn = v; if (v > mx) mx = v; if (allInt && v !== Math.round(v)) allInt = false; }
    if (allInt && mn >= -32768 && mx <= 32767) datatype = 4;
    else if (allInt && mn >= 0 && mx <= 65535) datatype = 512;
    if (datatype !== 16) { slope = s0.slope; inter = s0.inter; }
    else { const a = s0.slope, b = s0.inter; for (let i = 0; i < data.length; i++) data[i] = data[i] * a + b; }
  }
  const zooms = [dc, dr, Math.hypot(...sv)];
  const descrip = `BulboOlf DICOM;TE=${fmtNum(s0.te)};TR=${fmtNum(s0.tr)};${(s0.manufacturer || '').slice(0, 20)}`;
  const nii = writeNifti({ dims: [nx, ny, nz], affine, data: store, datatype, slope, inter, descrip, qform: true });
  const warnings = candidateWarnings(cand);
  const spacingDicom = s0.spacingBetween;
  if (spacingDicom && Math.abs(spacingDicom - zooms[2]) > 0.05 * zooms[2]) warnings.push(`SpacingBetweenSlices (${spacingDicom}) difere do espaçamento calculado pelas posições (${zooms[2].toFixed(3)} mm); usado o das posições`);
  const meta = {
    seriesNumber: cand.seriesNumber, seriesDescription: cand.label, protocolName: cand.protocolName, sequenceName: cand.sequenceName,
    mrAcqType: cand.mrAcqType, te: cand.te, tr: cand.tr, manufacturer: cand.manufacturer, model: cand.model, fieldStrength: cand.fieldStrength,
    nImages: nz, files: byFile.size, transferSyntax: transferSyntaxName(cand.ts), plane: cand.plane, studyDate: s0.studyDate,
    converted: new Date().toISOString(),
  };
  return { bytes: await gzip(nii), dims: [nx, ny, nz], affine, zooms, meta, warnings };
}
const fmtNum = (v) => (Number.isFinite(v) ? Number(v.toFixed(2)) : '');

// ---------------- utilidades de arquivos ----------------
// arquivos que certamente não são DICOM (CDs de PACS trazem visualizadores, HTML etc.)
export function maybeDicom(f) {
  if (/\.(nii|nii\.gz|gz|zip|txt|htm|html|xml|xsl|css|js|exe|dll|inf|ini|ico|png|jpe?g|bmp|gif|pdf|json|csv|md|bat|sh|cfg|log|mov|mp4|jar|class|cab|msi|sys|dat|db|plist)$/i.test(f.name)) return false;
  if (/^(dicomdir|autorun\.inf|\.ds_store|thumbs\.db|lockfile|version)$/i.test(f.name)) return false;
  return f.size >= 256;
}

// expande pastas soltas na área de arrastar (DataTransferItem.webkitGetAsEntry)
export async function filesFromDataTransfer(dt) {
  const entries = [...dt.items].map((it) => it.webkitGetAsEntry?.()).filter(Boolean);
  if (!entries.length) return [...dt.files];
  const out = [];
  const walk = async (entry, path) => {
    if (entry.isFile) {
      const f = await new Promise((res, rej) => entry.file(res, rej));
      try { Object.defineProperty(f, 'relPath', { value: path + f.name }); } catch {}
      out.push(f);
    } else if (entry.isDirectory) {
      const rd = entry.createReader();
      for (;;) {
        const batch = await new Promise((res, rej) => rd.readEntries(res, rej));
        if (!batch.length) break;
        for (const e of batch) await walk(e, path + entry.name + '/');
      }
    }
  };
  for (const e of entries) await walk(e, '');
  return out;
}
