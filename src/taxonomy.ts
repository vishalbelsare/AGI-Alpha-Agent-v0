// SPDX-License-Identifier: Apache-2.0
export interface TaxonomyNode {
  id: string;
  parent: string | null;
}

export interface HyperGraph {
  nodes: Record<string, TaxonomyNode>;
}

/**
 * Mine a taxonomy from a list of insight runs. Each run may define
 * `params.sector` which becomes a node in the taxonomy.
 */
export function mineTaxonomy(runs: Array<{ params?: { sector?: string } }>): HyperGraph {
  const graph: HyperGraph = { nodes: {} };
  for (const r of runs) {
    const sec = r.params?.sector;
    if (sec && !graph.nodes[sec]) {
      graph.nodes[sec] = { id: sec, parent: null };
    }
  }
  return graph;
}

/**
 * Remove nodes not present in `valid`.
 */
export function pruneTaxonomy(graph: HyperGraph, valid: Set<string>): HyperGraph {
  const out: HyperGraph = { nodes: {} };
  for (const id of valid) {
    const n = graph.nodes[id];
    if (n) out.nodes[id] = n;
  }
  return out;
}

const DB_NAME = 'sectorTaxonomy';
const NODE_STORE = 'nodes';
const META_STORE = 'meta';
const VERSION_KEY = 'version';
const CURRENT_VERSION = 1;

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(NODE_STORE)) db.createObjectStore(NODE_STORE);
      if (!db.objectStoreNames.contains(META_STORE)) db.createObjectStore(META_STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function withStore<T>(mode: IDBTransactionMode, store: string, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return openDB().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const tx = db.transaction(store, mode);
        const st = tx.objectStore(store);
        const req = fn(st);
        tx.oncomplete = () => resolve(req.result as T);
        tx.onerror = () => reject(tx.error);
      }),
  );
}

export async function saveTaxonomy(graph: HyperGraph): Promise<void> {
  await withStore('readwrite', NODE_STORE, (s) => {
    s.clear();
    for (const n of Object.values(graph.nodes)) s.put(n, n.id);
    return s.put(0, '__dummy__');
  });
  await withStore('readwrite', META_STORE, (s) => s.put(CURRENT_VERSION, VERSION_KEY));
}

export async function loadTaxonomy(): Promise<HyperGraph> {
  const version = await withStore<number>('readonly', META_STORE, (s) => s.get(VERSION_KEY));
  if (version !== CURRENT_VERSION) {
    await withStore('readwrite', NODE_STORE, (s) => {
      s.clear();
      return s.put(0, '__dummy__');
    });
    await withStore('readwrite', META_STORE, (s) => s.put(CURRENT_VERSION, VERSION_KEY));
    return { nodes: {} };
  }
  const nodes = (await withStore< TaxonomyNode[] >('readonly', NODE_STORE, (s) => s.getAll())) || [];
  const out: HyperGraph = { nodes: {} };
  for (const n of nodes) {
    if (n && n.id) out.nodes[n.id] = { id: n.id, parent: n.parent ?? null };
  }
  return out;
}
