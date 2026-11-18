import fs from 'node:fs';
import path from 'node:path';
import { scanFile, groupSeries, convertToNifti, maybeDicom } from '../../js/dicom.js';
const [dir, outFile] = process.argv.slice(2);
const walk = (d) => fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => e.isDirectory() ? walk(path.join(d, e.name)) : [path.join(d, e.name)]);
const files = walk(dir).filter((p) => !p.endsWith('.npy')).map((p) => { const b = fs.readFileSync(p); const f = new File([b], path.basename(p)); Object.defineProperty(f, 'relPath', { value: path.relative(dir, p) }); return f; }).filter(maybeDicom);
const t0 = performance.now();
const slices = [];
for (const f of files) { try { slices.push(...await scanFile(f)); } catch (e) { console.log('skip', f.name, e.message); } }
const studies = groupSeries(slices);
for (const st of studies) for (const c of st.series) console.log(JSON.stringify({ num: c.seriesNumber, label: c.label, n: c.n, plane: c.plane, voxel: c.voxel.map((v) => +v.toFixed(3)), score: c.score, ok: c.ok, sug: !!c.suggested, problems: c.problems, ts: c.ts }));
const cand = studies[0].series.find((c) => c.suggested) || studies[0].series[0];
const r = await convertToNifti(cand);
fs.writeFileSync(outFile, r.bytes);
console.log('dims', r.dims, 'zooms', r.zooms.map((v) => v.toFixed(4)), 'warn', r.warnings, 'ms', (performance.now() - t0).toFixed(0));
