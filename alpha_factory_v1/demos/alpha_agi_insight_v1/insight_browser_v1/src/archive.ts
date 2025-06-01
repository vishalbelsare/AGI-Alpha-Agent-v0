// SPDX-License-Identifier: Apache-2.0
export interface RunEntry {
  id?: number;
  gen: number;
  params: any;
  pop: any[];
  ts: number;
}

export class Archive {
  private db: IDBDatabase | null = null;
  constructor(private name = 'insight-archive') {}

  async open(): Promise<IDBDatabase> {
    if (this.db) return this.db;
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(this.name, 1);
      req.onupgradeneeded = () => {
        req.result.createObjectStore('runs', { keyPath: 'id', autoIncrement: true });
      };
      req.onsuccess = () => {
        this.db = req.result;
        resolve(this.db);
      };
      req.onerror = () => reject(req.error);
    });
  }

  private store(mode: IDBTransactionMode) {
    if (!this.db) throw new Error('DB not opened');
    return this.db.transaction('runs', mode).objectStore('runs');
  }

  async add(gen: number, params: any, pop: any[]): Promise<number> {
    await this.open();
    return new Promise((resolve, reject) => {
      const req = this.store('readwrite').add({ gen, params, pop, ts: Date.now() });
      req.onsuccess = () => resolve(req.result as number);
      req.onerror = () => reject(req.error);
    });
  }

  async list(): Promise<RunEntry[]> {
    await this.open();
    return new Promise((resolve, reject) => {
      const req = this.store('readonly').getAll();
      req.onsuccess = () => {
        const runs = req.result as RunEntry[];
        runs.sort((a, b) => a.ts - b.ts);
        resolve(runs);
      };
      req.onerror = () => reject(req.error);
    });
  }

  async prune(max: number): Promise<void> {
    const runs = await this.list();
    if (runs.length <= max) return;
    const toRemove = runs.slice(0, runs.length - max);
    await new Promise<void>((resolve, reject) => {
      const tx = this.db!.transaction('runs', 'readwrite');
      const store = tx.objectStore('runs');
      for (const r of toRemove) store.delete(r.id!);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  async selectParents(count: number): Promise<RunEntry[]> {
    const runs = await this.list();
    const result: RunEntry[] = [];
    for (let i = 0; i < Math.min(count, runs.length); i++) {
      const idx = Math.floor(Math.random() * runs.length);
      result.push(runs[idx]);
    }
    return result;
  }
}
