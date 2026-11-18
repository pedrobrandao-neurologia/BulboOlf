// sw.js — cache offline + cabeçalhos COOP/COEP (permitem WASM multithread no GitHub Pages)
const CACHE = 'bulboolf-v2';
const SHELL = ['./', 'index.html', 'css/app.css', 'js/app.js', 'js/pipeline.js', 'js/nifti.js', 'js/dicom.js', 'js/worker.js', 'js/store.js',
  'vendor/niivue.js', 'vendor/ort.webgpu.min.mjs', 'manifest.webmanifest', 'icons/icon-192.png', 'icons/icon-512.png'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});

function withIsolation(res) {
  if (!res || res.status === 0) return res;
  const h = new Headers(res.headers);
  h.set('Cross-Origin-Embedder-Policy', 'require-corp');
  h.set('Cross-Origin-Opener-Policy', 'same-origin');
  h.set('Cross-Origin-Resource-Policy', 'same-origin');
  return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
}

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  e.respondWith((async () => {
    const cache = await caches.open(CACHE);
    const isCode = /\.(html|js|mjs|css|webmanifest)$/.test(url.pathname) || url.pathname.endsWith('/');
    if (isCode) { // rede primeiro (atualizações), cache como reserva
      try {
        const res = await fetch(req);
        if (res.ok) cache.put(req, res.clone());
        return withIsolation(res);
      } catch { return withIsolation(await cache.match(req)); }
    }
    const hit = await cache.match(req); // modelos, wasm, ícones: cache primeiro
    if (hit) return withIsolation(hit);
    const res = await fetch(req);
    if (res.ok) cache.put(req, res.clone());
    return withIsolation(res);
  })());
});
