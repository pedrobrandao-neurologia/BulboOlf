// store.js — persistência local (IndexedDB) dos sujeitos e resultados. Nada sai do navegador.
const DB = 'bulboolf', STORE = 'subjects';
let dbp = null;
function open() {
  if (dbp) return dbp;
  dbp = new Promise((res, rej) => {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(STORE, { keyPath: 'uid' });
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
  return dbp;
}
async function tx(mode, fn) {
  const db = await open();
  return new Promise((res, rej) => {
    const t = db.transaction(STORE, mode);
    const out = fn(t.objectStore(STORE));
    t.oncomplete = () => res(out?.result ?? out);
    t.onerror = () => rej(t.error);
  });
}
export const store = {
  async all() { try { return await tx('readonly', (s) => s.getAll()); } catch { return []; } },
  async put(obj) { try { await tx('readwrite', (s) => s.put(obj)); } catch (e) { console.warn('IndexedDB', e); } },
  async del(uid) { try { await tx('readwrite', (s) => s.delete(uid)); } catch (e) { console.warn('IndexedDB', e); } },
};
