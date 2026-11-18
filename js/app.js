// app.js — interface do BulboOlf (thread principal)
import { Niivue, NVImage } from '../vendor/niivue.js';
import { parseNifti, gunzipIfNeeded, writeNifti, gzip, zoomsFromAffine } from './nifti.js';
import { axCodes } from './pipeline.js';
import { store } from './store.js';
import { scanFile, groupSeries, convertToNifti, candidateWarnings, maybeDicom, filesFromDataTransfer } from './dicom.js';

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const fmt = (x, d = 1) => (x === null || x === undefined || Number.isNaN(x) ? '—' : Number(x).toLocaleString('pt-BR', { minimumFractionDigits: d, maximumFractionDigits: d }));

const state = {
  subjects: [], sel: null, engine: null, running: false, queue: [],
  grid: 'native', tool: null, osPts: [], pen: 1, t2cache: new Map(), viewerUid: null, viewerGrid: null,
};

// ---------------- service worker (offline + isolamento para WASM multithread) ----------------
if ('serviceWorker' in navigator && location.protocol !== 'file:') {
  navigator.serviceWorker.register('sw.js').then(() => {
    if (!crossOriginIsolated && navigator.serviceWorker.controller === null) {
      navigator.serviceWorker.ready.then(() => { if (!sessionStorage.getItem('coiReload')) { sessionStorage.setItem('coiReload', '1'); location.reload(); } });
    }
  }).catch(() => {});
}

// ---------------- worker ----------------
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
const pending = new Map();
let reqId = 0;
function call(type, payload, transfer = [], onProgress) {
  const id = ++reqId;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject, onProgress, type });
    worker.postMessage({ type, id, ...payload }, transfer);
  });
}
worker.onmessage = (ev) => {
  const m = ev.data;
  if (m.type === 'ready') {
    state.engine = m;
    $('#engine').textContent = m.backend === 'webgpu' ? 'Motor: WebGPU (GPU)' : `Motor: WASM (CPU, ${m.threads} thread${m.threads > 1 ? 's' : ''})`;
    $('#engine').title = m.isolated ? 'Isolamento de origem ativo' : 'Sem isolamento de origem: WASM em 1 thread';
    return;
  }
  if (m.type === 'warn') { hint(m.msg); if (state.engine) { state.engine.backend = 'wasm'; $('#engine').textContent = 'Motor: WASM (CPU)'; } return; }
  if (m.type === 'selftest') { showSelfTest(m); return; }
  const p = pending.get(m.id);
  if (!p) return;
  if (m.type === 'progress') { p.onProgress?.(m); return; }
  pending.delete(m.id);
  if (m.type === 'error') p.reject(new Error(m.msg)); else p.resolve(m);
};
worker.postMessage({ type: 'init', backend: 'auto' });

// ---------------- sujeitos ----------------
const uid = () => Math.random().toString(36).slice(2, 10);
function idFromName(n) { return n.replace(/\.nii(\.gz)?$/i, '').replace(/[_-]?(t2|T2w?|SPACE|space|cube|CUBE)[^/]*$/, '') || n; }

async function addFiles(files) {
  const nii = files.filter((f) => /\.nii(\.gz)?$/i.test(f.name));
  const others = files.filter((f) => !nii.includes(f));
  for (const f of nii) await addNifti(f);
  if (nii.length) { renderTable(); if (!state.sel && state.subjects.length) select(state.subjects[0].uid); }
  const dcm = others.filter(maybeDicom);
  if (dcm.length) await importDicom(dcm);
  else if (others.length) hint(`Ignorado (não é NIfTI nem DICOM): ${others.slice(0, 3).map((f) => f.name).join(', ')}${others.length > 3 ? '…' : ''}`);
}

// extra: { id, group, dicom (metadados da série), convWarn (alertas da conversão) } quando vem do DICOM
async function addNifti(f, extra = {}) {
  const { convWarn = [], ...rest } = extra;
  const s = { uid: uid(), id: idFromName(f.name), fileName: f.name, file: f, group: guessGroup(f.name), age: '', sex: '', smell: '', qc: 'pendente', notes: '', status: 'novo', ...rest };
  try {
    const vol = parseNifti(await gunzipIfNeeded(await f.arrayBuffer()));
    const z = zoomsFromAffine(vol.affine);
    const warn = [];
    if (z.some((v) => v > 0.9 || v < 0.6)) warn.push(`voxels ${z.map((v) => v.toFixed(2)).join('×')} mm: o modelo automático foi treinado em T2 3D isotrópico de 0,7–0,8 mm`);
    if (Math.max(...z) / Math.min(...z) > 1.5) warn.push('voxels anisotrópicos (aquisição 2D?) — prefira a planimetria manual');
    if (vol.nVolumes > 1) warn.push(`arquivo 4D (${vol.nVolumes} volumes): será usado o primeiro`);
    for (const w of convWarn) if (!/anisotr|0,7–0,8/.test(w)) warn.push(w); // evita repetir os alertas de voxel acima
    s.info = { dims: vol.dims, zooms: z, codes: axCodes(vol.affine).join(''), warn };
  } catch (e) { s.status = 'erro'; s.error = e.message; }
  state.subjects.push(s);
  await store.put(s);
  return s;
}

// ---------------- DICOM: escolha da série T2 3D -> conversão para NIfTI -> fila do pipeline ----------------
function progress(frac, msg) { $('#progBar').style.width = `${Math.round(frac * 100)}%`; $('#progMsg').textContent = msg; }
async function importDicom(files) {
  if (state.dicomBusy) { hint('Já há uma importação DICOM em andamento.'); return; }
  state.dicomBusy = true;
  try {
    const slices = [];
    let notDicom = 0;
    for (let i = 0; i < files.length; i++) {
      try { const r = await scanFile(files[i]); if (!r.length) notDicom++; slices.push(...r); } catch { notDicom++; }
      if (i % 25 === 0 || i === files.length - 1) progress((i + 1) / files.length, `Lendo cabeçalhos DICOM ${i + 1}/${files.length}…`);
    }
    const studies = groupSeries(slices).filter((st) => st.series.length);
    if (!studies.length) { progress(0, ''); hint(`Nenhuma imagem DICOM encontrada em ${files.length} arquivo(s).`); return; }
    progress(1, `${slices.length} imagens DICOM em ${studies.length} estudo(s)${notDicom ? ` · ${notDicom} arquivo(s) ignorado(s)` : ''}. Escolha a série T2 3D.`);
    const choices = await chooseSeries(studies);
    if (!choices?.length) { progress(0, 'Importação DICOM cancelada.'); return; }
    const saveNii = $('#dicomSaveNii').checked;
    let last = null;
    for (const [n, ch] of choices.entries()) {
      const tag = `${ch.id} (${n + 1}/${choices.length})`;
      try {
        const r = await convertToNifti(ch.cand, (f) => progress(f, `${tag}: convertendo DICOM → NIfTI… ${Math.round(f * 100)}%`));
        const name = `${ch.id.replace(/[^\w.-]+/g, '_')}_T2_3D.nii.gz`;
        const file = new File([r.bytes], name, { type: 'application/gzip' });
        if (saveNii) download(name, r.bytes, 'application/gzip');
        last = await addNifti(file, { id: ch.id, group: ch.group, dicom: r.meta, convWarn: r.warnings });
        renderTable();
      } catch (e) {
        hint(`Erro ao converter ${ch.id}: ${e.message}`);
        const s = { uid: uid(), id: ch.id, fileName: `DICOM série ${ch.cand.seriesNumber ?? ''} ${ch.cand.label}`, group: ch.group, age: '', sex: '', smell: '', qc: 'pendente', notes: '', status: 'erro', error: `Conversão DICOM → NIfTI falhou: ${e.message}. Converta com dcm2niix -z y e adicione o NIfTI.` };
        state.subjects.push(s); await store.put(s);
      }
    }
    renderTable();
    if (last) await select(last.uid); // o visualizador limpa a mensagem de progresso: selecionar antes
    progress(1, `Conversão concluída: ${choices.length} série(s) adicionada(s) como NIfTI. Clique em "Processar".`);
  } finally { state.dicomBusy = false; }
}

function chooseSeries(studies) {
  const dlg = $('#dlgDicom'), box = $('#dicomStudies');
  const f2 = (v) => (Number.isFinite(v) ? v.toLocaleString('pt-BR', { maximumFractionDigits: 2 }) : '?');
  const fdate = (d) => (/^\d{8}$/.test(d) ? `${d.slice(6, 8)}/${d.slice(4, 6)}/${d.slice(0, 4)}` : d || '');
  box.innerHTML = studies.map((st, si) => {
    const sug = st.series.find((c) => c.suggested);
    const rows = st.series.map((c, ci) => {
      const warns = candidateWarnings(c);
      const tags = [c.suggested ? '<span class="tag sug">★ sugerida</span>' : '',
        ...c.problems.map((p) => `<span class="tag bad">${esc(p)}</span>`), ...warns.map((w) => `<span class="tag warn">${esc(w)}</span>`)].join('');
      const desc = [c.protocolName && c.protocolName !== c.label ? c.protocolName : '', c.sequenceName].filter(Boolean).join(' · ');
      return `<tr class="${c.ok ? '' : 'dis'} ${c.suggested ? 'sug' : ''}">
        <td><input type="radio" name="st${si}" value="${ci}" id="st${si}_${ci}" ${c.ok ? '' : 'disabled'} ${c === sug ? 'checked' : ''}></td>
        <td>${c.seriesNumber ?? ''}</td>
        <td><label for="st${si}_${ci}"><b>${esc(c.label)}</b>${desc ? `<br><span class="muted">${esc(desc)}</span>` : ''}</label></td>
        <td>${esc(c.mrAcqType || '?')}</td><td>${esc(c.plane)}</td><td>${c.n}</td><td>${c.cols}×${c.rows}</td>
        <td class="nw">${c.voxel.map(f2).join(' × ')}</td><td class="nw">${f2(c.te)} / ${f2(c.tr)}</td><td>${tags}</td></tr>`;
    }).join('');
    const meta = [fdate(st.studyDate), st.studyDescription, st.dir.length ? `pasta: ${st.dir.join('/')}` : ''].filter(Boolean).map(esc).join(' · ');
    return `<div class="study" data-si="${si}">
      <div class="study-head">
        <label class="field">ID do sujeito <input type="text" class="st-id" value="${esc(st.suggestedId)}"></label>
        <label class="field">Grupo <select class="st-group">${['', 'AF', 'Controle', 'Outro'].map((g) => `<option ${g === guessGroup(st.suggestedId) ? 'selected' : ''}>${g}</option>`).join('')}</select></label>
        <div class="meta">Estudo ${si + 1}/${studies.length}${meta ? ' · ' + meta : ''}</div>
      </div>
      <div class="series-wrap"><table class="series-table">
        <thead><tr><th></th><th>Nº</th><th>Série</th><th>Aquis.</th><th>Plano</th><th>Imagens</th><th>Matriz</th><th>Voxel (mm)</th><th>TE / TR (ms)</th><th>Situação</th></tr></thead>
        <tbody>${rows}
          <tr><td><input type="radio" name="st${si}" value="-1" id="st${si}_none" ${sug ? '' : 'checked'}></td><td></td><td colspan="8"><label for="st${si}_none" class="muted">Não importar este estudo</label></td></tr>
        </tbody></table></div>
    </div>`;
  }).join('');
  return new Promise((resolve) => {
    const finish = (val) => { dlg.onclose = null; if (dlg.open) dlg.close(); resolve(val); };
    $('#dicomCancel').onclick = () => finish(null);
    dlg.onclose = () => finish(null); // Esc
    $('#dicomConvert').onclick = () => {
      const out = [];
      box.querySelectorAll('.study').forEach((el) => {
        const st = studies[Number(el.dataset.si)];
        const ci = Number(el.querySelector('input[type=radio]:checked')?.value ?? -1);
        if (ci < 0) return;
        out.push({ cand: st.series[ci], id: el.querySelector('.st-id').value.trim() || st.suggestedId, group: el.querySelector('.st-group').value });
      });
      if (!out.length) { hint('Nenhuma série escolhida.'); }
      finish(out);
    };
    dlg.showModal();
  });
}
function guessGroup(n) { return /(^|[^a-z])(af|frda|fa|friedreich|pac)/i.test(n) ? 'AF' : /(^|[^a-z])(ctrl|controle|hc|control|cn)/i.test(n) ? 'Controle' : ''; }

function renderTable() {
  const tb = $('#subjTable tbody');
  tb.innerHTML = '';
  for (const s of state.subjects) {
    const tr = document.createElement('tr');
    tr.dataset.uid = s.uid;
    if (s.uid === state.sel) tr.classList.add('sel');
    const st = s.status === 'ok' ? 'ok' : s.status === 'rodando' || s.status === 'na fila' ? 'run' : s.status === 'erro' ? 'err' : '';
    const vol = s.auto ? fmt(s.auto.stats.rows.Total_OB.Volume_mm3) : '—';
    tr.innerHTML = `<td>${esc(s.id)}</td><td>${esc(s.group || '—')}</td><td><span class="st ${st}">${esc(s.status)}</span></td><td>${vol}</td>`;
    tr.onclick = () => select(s.uid);
    tb.appendChild(tr);
  }
}
const esc = (t) => String(t ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const cur = () => state.subjects.find((s) => s.uid === state.sel);

async function select(u) {
  state.sel = u;
  renderTable(); renderDetails();
  const s = cur();
  if (s?.auto && state.grid === 'native') setGrid('pipeline', false);
  await showViewer();
}

// ---------------- detalhes ----------------
function renderDetails() {
  const s = cur(), el = $('#details');
  if (!s) { el.innerHTML = '<p class="muted">Nenhum sujeito selecionado.</p>'; return; }
  const a = s.auto, rows = a?.stats.rows;
  let h = `
  <div class="field-row">
    <label class="field">ID <input type="text" data-k="id" value="${esc(s.id)}"></label>
    <label class="field">Grupo <select data-k="group">${['', 'AF', 'Controle', 'Outro'].map((g) => `<option ${g === s.group ? 'selected' : ''}>${g}</option>`).join('')}</select></label>
  </div>
  <div class="field-row">
    <label class="field">Idade <input type="number" data-k="age" value="${esc(s.age)}"></label>
    <label class="field">Sexo <select data-k="sex">${['', 'F', 'M'].map((g) => `<option ${g === s.sex ? 'selected' : ''}>${g}</option>`).join('')}</select></label>
  </div>
  <div class="field-row">
    <label class="field">Escore olfativo <input type="number" step="any" data-k="smell" value="${esc(s.smell)}"></label>
    <label class="field">Controle de qualidade <select data-k="qc">${['pendente', 'ok', 'revisar', 'excluir'].map((g) => `<option ${g === s.qc ? 'selected' : ''}>${g}</option>`).join('')}</select></label>
  </div>
  <div class="kv"><div>Arquivo</div><div>${esc(s.fileName)}</div>`;
  if (s.dicom) {
    const d = s.dicom, f2 = (v) => (Number.isFinite(v) ? fmt(v, v % 1 ? 1 : 0) : '?');
    h += `<div>Origem DICOM</div><div>série ${esc(d.seriesNumber ?? '')} · ${esc(d.seriesDescription)}</div>
      <div>Aquisição</div><div>${esc(d.mrAcqType || '?')} · ${esc(d.plane)} · TE ${f2(d.te)} / TR ${f2(d.tr)} ms</div>
      <div>Equipamento</div><div>${esc([d.manufacturer, d.model, d.fieldStrength ? `${d.fieldStrength} T` : ''].filter(Boolean).join(' '))}</div>
      <div>Conversão</div><div>${d.nImages} imagens · ${esc(d.transferSyntax)}</div>`;
  }
  if (s.info) h += `<div>Matriz / voxel</div><div>${s.info.dims.join('×')} · ${s.info.zooms.map((v) => v.toFixed(2)).join('×')} mm · ${s.info.codes}</div>`;
  if (s.center) h += `<div>Centro manual (mm)</div><div>${s.center.map((v) => v.toFixed(1)).join(', ')}</div>`;
  h += '</div>';
  (s.info?.warn || []).forEach((w) => { h += `<div class="flag">⚠ ${esc(w)}</div>`; });
  if (s.error) h += `<div class="flag bad">${esc(s.error)}</div>`;
  if (a) {
    h += `<h3>Segmentação automática</h3>
    <table class="res-table"><thead><tr><th></th><th>Volume (mm³)</th><th>Voxels</th><th>Entropia</th><th>CV</th></tr></thead><tbody>
      <tr><td><span class="dot l"></span>Esquerdo</td><td>${fmt(rows.Left_OB.Volume_mm3)}</td><td>${rows.Left_OB.NVoxels}</td><td>${fmt(rows.Left_OB.Entropy, 3)}</td><td>${fmt(rows.Left_OB.CV, 3)}</td></tr>
      <tr><td><span class="dot r"></span>Direito</td><td>${fmt(rows.Right_OB.Volume_mm3)}</td><td>${rows.Right_OB.NVoxels}</td><td>${fmt(rows.Right_OB.Entropy, 3)}</td><td>${fmt(rows.Right_OB.CV, 3)}</td></tr>
      <tr><td><b>Total</b></td><td><b>${fmt(rows.Total_OB.Volume_mm3)}</b></td><td>${rows.Total_OB.NVoxels}</td><td>${fmt(rows.Total_OB.Entropy, 3)}</td><td>${fmt(rows.Total_OB.CV, 3)}</td></tr>
    </tbody></table>
    <div class="kv">
      <div>Modelos no ensemble</div><div>${a.nModels}</div>
      <div>Localização</div><div>${a.locMode === 'auto' ? 'automática' : 'manual'}</div>
      <div>Distância localização–segmentação</div><div>${fmt(a.stats.locSegDist, 2)} mm</div>
      <div>Reamostrado para 0,8 mm</div><div>${a.resampled ? 'sim' : 'não'}</div>
      <div>Motor / tempo</div><div>${a.backend} · ${fmt(a.seconds, 0)} s</div>
    </div>`;
    if (a.stats.flag) h += `<div class="flag bad">${esc(a.stats.flag)}</div>`;
    if (a.outsideVoxels > 0) h += `<div class="flag">O recorte de 96³ ultrapassou a borda da imagem (${a.outsideVoxels} voxels preenchidos com zero).</div>`;
    if (rows.Total_OB.CV > 0.2) h += '<div class="flag">CV alto entre os modelos do ensemble — revisar visualmente.</div>';
  }
  if (s.corrected) {
    const c = s.corrected;
    h += `<h3>Volumes revisados/manuais</h3><div class="kv">
      <div><span class="dot l"></span>Esquerdo</div><div>${fmt(c.L)} mm³</div>
      <div><span class="dot r"></span>Direito</div><div>${fmt(c.R)} mm³</div>
      <div>Total</div><div>${fmt(c.L + c.R)} mm³</div>
      <div>Grade</div><div>${c.grid === 'pipeline' ? 'pipeline 0,8 mm' : 'imagem original'} (voxel ${fmt(c.voxelSize, 4)} mm³)</div></div>`;
  }
  if (s.os?.L || s.os?.R) {
    h += `<h3>Profundidade do sulco olfatório</h3><div class="kv">
      <div>Esquerdo</div><div>${fmt(s.os?.L?.depth, 2)} mm</div><div>Direito</div><div>${fmt(s.os?.R?.depth, 2)} mm</div></div>`;
  }
  h += `<label class="field">Notas <textarea data-k="notes">${esc(s.notes)}</textarea></label>
  <button class="btn small" id="btnRemove">Remover sujeito</button>`;
  el.innerHTML = h;
  el.querySelectorAll('[data-k]').forEach((inp) => inp.addEventListener('change', async () => {
    s[inp.dataset.k] = inp.value; await store.put(s); renderTable();
  }));
  $('#btnRemove').onclick = async () => {
    state.subjects = state.subjects.filter((x) => x.uid !== s.uid); await store.del(s.uid);
    state.sel = state.subjects[0]?.uid || null; renderTable(); renderDetails(); showViewer();
  };
}

// ---------------- processamento ----------------
function opts(s) {
  const loc = $('#optLoc').value;
  const gpu = state.engine?.backend === 'webgpu';
  return {
    mode: $('#optMode').value === 'all' ? 'all' : Number($('#optMode').value),
    localization: loc === 'auto' ? 'auto' : { ras: s.center },
    batchLoc: 4, batchSeg: gpu ? 1 : 2,
  };
}
async function process(s) {
  if (!s.file) { s.status = 'erro'; s.error = 'Arquivo de imagem indisponível nesta sessão: adicione o NIfTI novamente.'; return; }
  if ($('#optLoc').value === 'manual' && !s.center) { s.status = 'erro'; s.error = 'Localização manual: marque o centro dos bulbos antes de processar.'; return; }
  s.status = 'rodando'; s.error = null; renderTable();
  const buf = await s.file.arrayBuffer();
  const o = opts(s);
  const r = await call('run', { buffer: buf, opts: o }, [buf], (p) => {
    const w = p.stage === 'loc' ? 0.05 + 0.45 * p.frac : p.stage === 'seg' ? 0.5 + 0.5 * p.frac : p.stage === 'pre' ? 0.05 * p.frac : 1;
    const loc = o.localization === 'auto';
    const frac = loc ? w : p.stage === 'seg' ? 0.05 + 0.95 * p.frac : w;
    $('#progBar').style.width = `${Math.round(frac * 100)}%`;
    $('#progMsg').textContent = `${s.id}: ${p.msg}`;
  });
  state.t2cache.set(s.uid, r.t2);
  s.auto = {
    stats: r.stats, labels: r.labels, cropAffine: r.cropAffine, cropDims: r.cropDims, zeroXYZ: r.zeroXYZ, xyz: r.xyz, cmRas: r.cmRas,
    t2dims: r.t2.dims, t2affine: r.t2.affine, resampled: r.resampled, inputZooms: r.inputZooms, nModels: r.nModels, mode: r.mode,
    seconds: r.seconds, backend: r.backend, outsideVoxels: r.outsideVoxels, locMode: o.localization === 'auto' ? 'auto' : 'manual',
    date: new Date().toISOString(),
  };
  s.status = 'ok';
  await store.put(s);
}
async function runQueue(list) {
  if (state.running) return;
  state.running = true; $('#btnCancel').disabled = false;
  list.forEach((s) => { s.status = 'na fila'; }); renderTable();
  for (const s of list) {
    if (!state.running) { s.status = s.auto ? 'ok' : 'novo'; continue; }
    try { await process(s); } catch (e) { s.status = 'erro'; s.error = e.message; await store.put(s); }
    renderTable(); if (s.uid === state.sel) { renderDetails(); if (s.auto) { setGrid('pipeline', false); await showViewer(true); } }
  }
  state.running = false; $('#btnCancel').disabled = true;
  $('#progMsg').textContent = 'Fila concluída.'; $('#progBar').style.width = '100%';
  renderTable();
}

// ---------------- visualizador (NiiVue) ----------------
const nv = new Niivue({ backColor: [0, 0, 0, 1], show3Dcrosshair: false, dragAndDropEnabled: false, isColorbar: false, crosshairColor: [1, 0.85, 0.2, 0.45], loadingText: '' });
await nv.attachToCanvas($('#gl'));
nv.setSliceType(nv.sliceTypeCoronal);
nv.setDrawOpacity(0.55);
nv.setCrosshairWidth(1);

async function getT2(s) {
  if (state.t2cache.has(s.uid)) return state.t2cache.get(s.uid);
  if (!s.file) throw new Error('Arquivo de imagem indisponível: adicione o NIfTI novamente.');
  const buf = await s.file.arrayBuffer();
  $('#progMsg').textContent = `${s.id}: reconstruindo a grade do pipeline…`;
  const r = await call('prep', { buffer: buf }, [buf]);
  state.t2cache.set(s.uid, r.t2);
  return r.t2;
}
function fullGridLabels(s, t2) {
  const [tx, ty, tz] = t2.dims, S = 96, out = new Uint8Array(tx * ty * tz), z = s.auto.zeroXYZ, L = s.auto.labels;
  for (let k = 0; k < S; k++) for (let j = 0; j < S; j++) for (let i = 0; i < S; i++) {
    const v = L[i + S * (j + S * k)]; if (!v) continue;
    const a = z[0] + i, b = z[1] + j, c = z[2] + k;
    if (a >= 0 && b >= 0 && c >= 0 && a < tx && b < ty && c < tz) out[a + tx * (b + ty * c)] = v;
  }
  return out;
}
async function clearViewer() {
  try { nv.closeDrawing(); } catch {}
  for (const v of [...nv.volumes]) nv.removeVolume(v);
}
async function showViewer(force = false) {
  const s = cur();
  $('#viewerEmpty').style.display = s ? 'none' : 'flex';
  if (!s) { await clearViewer(); return; }
  if (!force && state.viewerUid === s.uid && state.viewerGrid === state.grid) return;
  try {
    await clearViewer();
    if (state.grid === 'pipeline' && s.auto) {
      const t2 = await getT2(s);
      const bg = writeNifti({ dims: t2.dims, affine: t2.affine, data: t2.data, datatype: 2 });
      nv.addVolume(await NVImage.loadFromUrl({ url: 'pipeline_t2.nii', name: 'pipeline_t2.nii', buffer: bg.buffer }));
      const lab = s.corrected?.grid === 'pipeline' && s.corrected.bitmap ? null : fullGridLabels(s, t2);
      if (lab) {
        const seg = writeNifti({ dims: t2.dims, affine: t2.affine, data: lab, datatype: 2 });
        nv.loadDrawing(await NVImage.loadFromUrl({ url: 'ob_seg.nii', name: 'ob_seg.nii', buffer: seg.buffer }));
      } else { nv.createEmptyDrawing(); nv.drawBitmap.set(s.corrected.bitmap); nv.refreshDrawing(true); }
      try { const m = s.auto.cmRas; nv.scene.crosshairPos = nv.mm2frac([m[0], m[1], m[2]]); setZoom(4); } catch {}
    } else {
      if (!s.file) { hint('Arquivo de imagem indisponível nesta sessão: adicione o NIfTI novamente.'); return; }
      const buf = await s.file.arrayBuffer();
      nv.addVolume(await NVImage.loadFromUrl({ url: s.fileName, name: s.fileName, buffer: buf }));
      if (s.corrected?.grid === 'native' && s.corrected.bitmap) { nv.createEmptyDrawing(); nv.drawBitmap.set(s.corrected.bitmap); nv.refreshDrawing(true); }
      setZoom(1);
    }
    nv.setDrawOpacity(Number($('#opacity').value));
    state.viewerUid = s.uid; state.viewerGrid = state.grid;
    $('#progMsg').textContent = '';
  } catch (e) { hint('Erro ao exibir: ' + e.message); }
}
function setZoom(z) { // mesma fórmula do zoom pela roda do NiiVue (centrado na cruz)
  try {
    const mm = nv.frac2mm(nv.scene.crosshairPos);
    nv.setPan2Dxyzmm([(1 - z) * mm[0], (1 - z) * mm[1], (1 - z) * mm[2], z]);
    $$('#zoomSeg button').forEach((b) => b.classList.toggle('on', Number(b.dataset.zoom) === z));
  } catch {}
}
function setGrid(g, refresh = true) {
  state.grid = g;
  $$('#gridSeg button').forEach((b) => b.classList.toggle('on', b.dataset.grid === g));
  if (refresh) showViewer(true);
}

// ---------------- ferramentas ----------------
function hint(t) { $('#toolHint').textContent = t || ''; }
function setTool(t) {
  state.tool = state.tool === t ? null : t;
  state.osPts = [];
  $('#penBox').hidden = state.tool !== 'edit';
  nv.setDrawingEnabled(state.tool === 'edit');
  if (state.tool === 'edit') { nv.setPenValue(state.pen, $('#penFill').checked); }
  ['toolCenter', 'toolEdit', 'toolSulcusL', 'toolSulcusR'].forEach((id) => $('#' + id).classList.toggle('primary', state.tool === { toolCenter: 'center', toolEdit: 'edit', toolSulcusL: 'osL', toolSulcusR: 'osR' }[id]));
  const msg = {
    center: 'Clique no ponto médio entre os dois bulbos olfatórios (corte coronal). Será usado na localização manual.',
    edit: 'Pinte com o mouse: 1 = bulbo esquerdo (vermelho), 2 = bulbo direito (verde). Planimetria: contorne o bulbo em cada corte coronal, do polo anterior até a transição para o trato. Ao terminar, clique em "Registrar volumes".',
    osL: 'Sulco ESQUERDO, no corte coronal do plano tangente à face posterior dos globos oculares: clique (1) borda inferior do giro reto, (2) borda inferior do giro orbital medial e (3) ponto mais profundo do sulco.',
    osR: 'Sulco DIREITO: mesmos 3 cliques — giro reto, giro orbital medial, fundo do sulco.',
  }[state.tool];
  hint(msg || '');
}
nv.onLocationChange = async (d) => {
  const s = cur(); if (!s || !state.tool) return;
  const mm = [d.mm[0], d.mm[1], d.mm[2]];
  if (state.tool === 'center') {
    s.center = mm; await store.put(s); setTool(null); hint(`Centro marcado em ${mm.map((v) => v.toFixed(1)).join(', ')} mm.`); renderDetails();
  } else if (state.tool === 'osL' || state.tool === 'osR') {
    state.osPts.push(mm);
    if (state.osPts.length < 3) { hint(`Ponto ${state.osPts.length}/3 registrado.`); return; }
    const [p1, p2, p3] = state.osPts;
    const u = p2.map((v, i) => v - p1[i]), w = p3.map((v, i) => v - p1[i]);
    const cr = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
    const depth = Math.hypot(...cr) / Math.hypot(...u);
    const side = state.tool === 'osL' ? 'L' : 'R';
    s.os = { ...(s.os || {}), [side]: { depth, points: state.osPts } };
    await store.put(s); setTool(null); hint(`Profundidade do sulco ${side === 'L' ? 'esquerdo' : 'direito'}: ${depth.toFixed(2)} mm`); renderDetails();
  }
};
async function applyDrawing() {
  const s = cur(); const bm = nv.drawBitmap;
  if (!s || !bm) { hint('Nenhum desenho ativo.'); return; }
  let l = 0, r = 0; for (let i = 0; i < bm.length; i++) { if (bm[i] === 1) l++; else if (bm[i] === 2) r++; }
  const back = nv.volumes[0];
  const z = back.hdr.pixDims.slice(1, 4);
  const vs = Math.abs(z[0] * z[1] * z[2]);
  s.corrected = { L: l * vs, R: r * vs, voxelSize: vs, grid: state.viewerGrid, bitmap: new Uint8Array(bm), date: new Date().toISOString() };
  await store.put(s); renderDetails(); hint(`Volumes registrados: E ${fmt(l * vs)} mm³ · D ${fmt(r * vs)} mm³.`);
}

// ---------------- exportação ----------------
function download(name, data, mime) {
  const url = URL.createObjectURL(new Blob([data], { type: mime }));
  const a = document.createElement('a'); a.href = url; a.download = name; document.body.appendChild(a); a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 2000);
}
function csv() {
  const cols = ['sub_id', 'arquivo', 'grupo', 'idade', 'sexo', 'escore_olfativo', 'qc', 'n_modelos', 'localizacao', 'motor', 'reamostrado_0.8mm', 'voxel_entrada_mm',
    'OB_E_vol_mm3', 'OB_D_vol_mm3', 'OB_total_vol_mm3', 'OB_E_nvox', 'OB_D_nvox', 'OB_E_entropia', 'OB_D_entropia', 'OB_total_entropia', 'OB_E_CV', 'OB_D_CV', 'OB_total_CV',
    'OB_E_intensidade_media', 'OB_D_intensidade_media', 'dist_loc_seg_mm', 'alerta',
    'OB_E_vol_revisado_mm3', 'OB_D_vol_revisado_mm3', 'OB_total_vol_revisado_mm3', 'grade_revisao', 'sulco_E_mm', 'sulco_D_mm', 'notas', 'data_processamento', 'serie_dicom', 'dicom_TE_ms', 'dicom_TR_ms'];
  const q = (v) => { if (v === undefined || v === null || Number.isNaN(v)) return ''; const t = String(v); return /[",\n;]/.test(t) ? `"${t.replace(/"/g, '""')}"` : t; };
  const lines = [cols.join(',')];
  for (const s of state.subjects) {
    const a = s.auto, R = a?.stats.rows, c = s.corrected;
    lines.push([s.id, s.fileName, s.group, s.age, s.sex, s.smell, s.qc, a?.nModels, a?.locMode, a?.backend, a ? (a.resampled ? 'sim' : 'nao') : '', s.info?.zooms.map((v) => v.toFixed(3)).join('x'),
      R?.Left_OB.Volume_mm3, R?.Right_OB.Volume_mm3, R?.Total_OB.Volume_mm3, R?.Left_OB.NVoxels, R?.Right_OB.NVoxels, R?.Left_OB.Entropy, R?.Right_OB.Entropy, R?.Total_OB.Entropy,
      R?.Left_OB.CV, R?.Right_OB.CV, R?.Total_OB.CV, R?.Left_OB.normMean, R?.Right_OB.normMean, a?.stats.locSegDist, a?.stats.flag,
      c ? c.L.toFixed(3) : '', c ? c.R.toFixed(3) : '', c ? (c.L + c.R).toFixed(3) : '', c?.grid, s.os?.L?.depth?.toFixed(3), s.os?.R?.depth?.toFixed(3), s.notes, a?.date,
      s.dicom ? `${s.dicom.seriesNumber ?? ''} ${s.dicom.seriesDescription}`.trim() : '', s.dicom?.te, s.dicom?.tr].map(q).join(','));
  }
  return '﻿' + lines.join('\n');
}
async function saveSeg() {
  const s = cur(); if (!s?.auto && !nv.drawBitmap) { hint('Nada para salvar.'); return; }
  if (s.auto) {
    const nii = writeNifti({ dims: s.auto.cropDims, affine: s.auto.cropAffine, data: s.auto.labels, datatype: 2 });
    download(`${s.id}_ob_seg_auto.nii.gz`, await gzip(nii), 'application/gzip');
  }
  if (nv.drawBitmap && state.tool === 'edit') await nv.saveImage({ filename: `${s.id}_ob_seg_revisado.nii.gz`, isSaveDrawing: true });
}
function sessionJSON() {
  const enc = (u8) => { let b = ''; for (let i = 0; i < u8.length; i += 0x8000) b += String.fromCharCode(...u8.subarray(i, i + 0x8000)); return btoa(b); };
  return JSON.stringify({ app: 'BulboOlf', version: 1, saved: new Date().toISOString(), subjects: state.subjects.map((s) => ({
    ...s, file: undefined, auto: s.auto ? { ...s.auto, labels: enc(s.auto.labels) } : undefined,
    corrected: s.corrected ? { ...s.corrected, bitmap: undefined } : undefined })) });
}
async function loadSession(text) {
  const dec = (b64) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const j = JSON.parse(text);
  for (const s of j.subjects) {
    if (s.auto) s.auto.labels = dec(s.auto.labels);
    const ex = state.subjects.find((x) => x.uid === s.uid);
    if (ex) Object.assign(ex, { ...s, file: ex.file }); else state.subjects.push(s);
    await store.put(state.subjects.find((x) => x.uid === s.uid));
  }
  renderTable(); renderDetails();
}

// ---------------- autoteste / pré-carregamento ----------------
async function prefetch() {
  const names = ['v1_split_4_FastSurferCNN_axial', 'v4_split_4_FastSurferCNN_coronal', 'v1_split_4_FastSurferCNN_sagittal',
    'v3_split_1_AttFastSurferCNN_axial', 'v1_split_2_AttFastSurferCNN_axial', 'v3_split_3_AttFastSurferCNN_axial', 'v4_split_4_AttFastSurferCNN_axial',
    'v4_split_1_AttFastSurferCNN_coronal', 'v3_split_2_AttFastSurferCNN_coronal', 'v4_split_3_AttFastSurferCNN_coronal', 'v3_split_4_AttFastSurferCNN_coronal',
    'v3_split_1_AttFastSurferCNN_sagittal', 'v3_split_2_AttFastSurferCNN_sagittal', 'v2_split_3_AttFastSurferCNN_sagittal', 'v4_split_4_AttFastSurferCNN_sagittal'];
  const files = [...names.map((n) => `models/${n}.onnx`), 'vendor/ort-wasm-simd-threaded.asyncify.wasm', 'vendor/ort-wasm-simd-threaded.asyncify.mjs'];
  for (let i = 0; i < files.length; i++) { await fetch(files[i]); $('#progBar').style.width = `${Math.round(100 * (i + 1) / files.length)}%`; $('#progMsg').textContent = `Baixando modelos ${i + 1}/${files.length}`; }
  $('#progMsg').textContent = 'Modelos disponíveis offline.';
}
function showSelfTest(m) {
  const ok = m.report.every((r) => r.argmaxAgreement > 0.999 && r.maxProbDiff < 1e-3);
  $('#dlgBody').innerHTML = `<h3>Autoteste (${m.backend})</h3><p>Compara a saída das redes neste computador com a saída de referência (ONNX Runtime, Python) para fatias reais de RM.</p>
  <table><thead><tr><th>Modelo</th><th>máx |Δ prob.|</th><th>máx |Δ logit|</th><th>Concordância de classe</th><th>Pixels de bulbo (ref.)</th><th>Tempo</th></tr></thead><tbody>
  ${m.report.map((r) => `<tr><td>${r.model}</td><td>${r.maxProbDiff.toExponential(1)}</td><td>${r.maxAbsDiff.toExponential(1)}</td><td>${(100 * r.argmaxAgreement).toFixed(3)}%</td><td>${r.fgPixels}</td><td>${r.ms.toFixed(0)} ms</td></tr>`).join('')}
  </tbody></table><p class="muted">Diferenças de logit em áreas de fundo (zero) são esperadas: empates no max-pooling são desfeitos de forma diferente em cada motor, sem efeito na probabilidade.</p><p><b>${ok ? '✓ Resultado compatível com a referência.' : '⚠ Diferenças acima do esperado neste motor. Recarregue e force o modo WASM, ou não use estes resultados.'}</b></p>`;
  $('#dlg').showModal();
}

// ---------------- eventos ----------------
$('#fileInput').onchange = (e) => { addFiles([...e.target.files]); e.target.value = ''; };
const drop = $('#drop');
['dragenter', 'dragover'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', async (e) => addFiles(await filesFromDataTransfer(e.dataTransfer)));
$('#dicomInput').onchange = (e) => { const fs = [...e.target.files]; e.target.value = ''; addFiles(fs); };
$('#btnRunSel').onclick = () => { const s = cur(); if (s) runQueue([s]); };
$('#btnRunAll').onclick = () => runQueue(state.subjects.filter((s) => !s.auto && s.file));
$('#btnCancel').onclick = () => { state.running = false; worker.postMessage({ type: 'cancel' }); };
$('#btnCsv').onclick = () => download(`bulbo_olfatorio_${new Date().toISOString().slice(0, 10)}.csv`, csv(), 'text/csv');
$('#btnSaveSession').onclick = () => download(`bulboolf_sessao_${new Date().toISOString().slice(0, 10)}.json`, sessionJSON(), 'application/json');
$('#sessionInput').onchange = async (e) => { const f = e.target.files[0]; if (f) await loadSession(await f.text()); e.target.value = ''; };
$('#btnPrefetch').onclick = prefetch;
$('#optEngine').onchange = (e) => { $('#engine').textContent = 'Reiniciando motor…'; worker.postMessage({ type: 'init', backend: e.target.value }); };
$('#btnSelfTest').onclick = () => worker.postMessage({ type: 'selftest', tests: [
  { model: 'v4_split_4_FastSurferCNN_coronal', input: 'selftest/loc_in.f32', expected: 'selftest/loc_out.f32' },
  { model: 'v3_split_1_AttFastSurferCNN_axial', input: 'selftest/seg_in.f32', expected: 'selftest/seg_out.f32' }] });
$('#dlgClose').onclick = () => $('#dlg').close();
$$('[data-view]').forEach((b) => b.onclick = () => {
  $$('[data-view]').forEach((x) => x.classList.toggle('on', x === b));
  nv.setSliceType({ coronal: nv.sliceTypeCoronal, axial: nv.sliceTypeAxial, sagittal: nv.sliceTypeSagittal, multi: nv.sliceTypeMultiplanar }[b.dataset.view]);
});
$$('#gridSeg button').forEach((b) => b.onclick = () => {
  if (b.dataset.grid === 'pipeline' && !cur()?.auto) { hint('A grade do pipeline existe após o processamento automático.'); return; }
  setGrid(b.dataset.grid);
});
$$('#zoomSeg button').forEach((b) => b.onclick = () => setZoom(Number(b.dataset.zoom)));
$('#crossChk').onchange = (e) => { nv.setCrosshairWidth(e.target.checked ? 1 : 0); nv.drawScene(); };
$('#opacity').oninput = (e) => nv.setDrawOpacity(Number(e.target.value));
$('#toolCenter').onclick = () => { if (state.grid !== 'native') setGrid('native'); setTool('center'); };
$('#toolEdit').onclick = () => setTool('edit');
$('#toolSulcusL').onclick = () => setTool('osL');
$('#toolSulcusR').onclick = () => setTool('osR');
$$('#penBox [data-pen]').forEach((b) => b.onclick = () => {
  state.pen = Number(b.dataset.pen); $$('#penBox [data-pen]').forEach((x) => x.classList.toggle('on', x === b));
  nv.setPenValue(state.pen, $('#penFill').checked);
});
$('#penFill').onchange = () => nv.setPenValue(state.pen, $('#penFill').checked);
$('#penUndo').onclick = () => nv.drawUndo();
$('#penEmpty').onclick = () => { nv.createEmptyDrawing(); nv.setDrawingEnabled(true); nv.setPenValue(state.pen, $('#penFill').checked); };
$('#penApply').onclick = applyDrawing;
$('#btnSaveSeg').onclick = saveSeg;

// ---------------- restaurar sessão local ----------------
state.subjects = (await store.all()).map((s) => ({ ...s, status: s.auto ? 'ok' : s.status === 'erro' ? 'erro' : 'novo' }));
renderTable();
if (state.subjects.length) select(state.subjects[0].uid);
window.__bulbo = { state, nv }; // depuração
