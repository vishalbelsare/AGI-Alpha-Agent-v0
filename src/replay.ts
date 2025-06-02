// SPDX-License-Identifier: Apache-2.0
export interface ReplayDelta {
  [key: string]: any;
}

export interface ReplayFrame {
  id: number;
  parent: number | null;
  delta: ReplayDelta;
  timestamp: number;
}

const DB_NAME = 'replay';
const FRAME_STORE = 'frames';

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(FRAME_STORE)) db.createObjectStore(FRAME_STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function withStore<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return openDB().then(
    db => new Promise<T>((resolve, reject) => {
      const tx = db.transaction(FRAME_STORE, mode);
      const st = tx.objectStore(FRAME_STORE);
      const req = fn(st);
      tx.oncomplete = () => resolve(req.result as T);
      tx.onerror = () => reject(tx.error);
    })
  );
}

export class ReplayDB {
  constructor(private name = DB_NAME) {}

  async open(): Promise<void> {
    await openDB();
  }

  async addFrame(parent: number | null, delta: ReplayDelta): Promise<number> {
    const id = Date.now() + Math.floor(Math.random() * 1000);
    const frame: ReplayFrame = { id, parent, delta, timestamp: Date.now() };
    await withStore('readwrite', (s) => s.put(frame, id));
    return id;
  }

  async getFrame(id: number): Promise<ReplayFrame | undefined> {
    return withStore('readonly', (s) => s.get(id));
  }

  async exportThread(id: number): Promise<ReplayFrame[]> {
    const out: ReplayFrame[] = [];
    let cur: number | null = id;
    while (cur) {
      const f = await this.getFrame(cur);
      if (!f) break;
      out.unshift(f);
      cur = f.parent;
    }
    return out;
  }

  static async cidForFrames(frames: ReplayFrame[]): Promise<string> {
    const deltas = frames.map((f) => f.delta);
    const buf = new TextEncoder().encode(JSON.stringify(deltas));
    const hash = await crypto.subtle.digest('SHA-256', buf);
    return Array.from(new Uint8Array(hash))
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('');
  }

  async computeCid(id: number): Promise<string> {
    const frames = await this.exportThread(id);
    return ReplayDB.cidForFrames(frames);
  }

  async share(id: number): Promise<{ cid: string; data: string }> {
    const frames = await this.exportThread(id);
    const cid = await ReplayDB.cidForFrames(frames);
    const data = JSON.stringify(frames.map((f) => ({ parent: f.parent, delta: f.delta })));
    return { cid, data };
  }

  async importFrames(json: string): Promise<number> {
    const records: Array<{ parent: number | null; delta: ReplayDelta }> = JSON.parse(json);
    let parent: number | null = null;
    let last = 0;
    for (const r of records) {
      last = await this.addFrame(parent, r.delta);
      parent = last;
    }
    return last;
  }
}

