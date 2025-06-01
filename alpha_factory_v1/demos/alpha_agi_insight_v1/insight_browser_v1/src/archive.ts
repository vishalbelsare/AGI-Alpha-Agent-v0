// SPDX-License-Identifier: Apache-2.0
import { createStore, set, get, del, keys, values } from './utils/keyval.js';

export interface InsightRun {
  id: number;
  seed: number;
  params: any;
  paretoFront: any[];
  parents: number[];
  score: number;
  novelty: number;
  timestamp: number;
}

export class Archive {
  private store;
  constructor(private name = 'insight-archive') {
    this.store = createStore(this.name, 'runs');
  }

  async open(): Promise<void> {
    await this.store.dbp;
  }

  private _vector(front: any[]): [number, number] {
    if (!front.length) return [0, 0];
    const l = front.reduce((s, d) => s + (d.logic ?? 0), 0) / front.length;
    const f = front.reduce((s, d) => s + (d.feasible ?? 0), 0) / front.length;
    return [l, f];
  }

  private _dist(a: [number, number], b: [number, number]): number {
    return Math.hypot(a[0] - b[0], a[1] - b[1]);
  }

  private async _novelty(vec: [number, number], k = 5): Promise<number> {
    const runs = await this.list();
    if (!runs.length) return 0;
    const dists = runs.map((r) => this._dist(vec, this._vector(r.paretoFront)));
    dists.sort((a, b) => a - b);
    const n = Math.min(k, dists.length);
    return dists.slice(0, n).reduce((s, d) => s + d, 0) / n;
  }

  async add(seed: number, params: any, paretoFront: any[], parents: number[] = []): Promise<number> {
    await this.open();
    const vec = this._vector(paretoFront);
    const score = (vec[0] + vec[1]) / 2;
    const novelty = await this._novelty(vec);
    const id = Date.now();
    const run: InsightRun = {
      id,
      seed,
      params,
      paretoFront,
      parents,
      score,
      novelty,
      timestamp: Date.now(),
    };
    await set(id, run, this.store);
    await this.prune(500);
    return id;
  }

  async list(): Promise<InsightRun[]> {
    await this.open();
    const runs = (await values(this.store)) as InsightRun[];
    runs.sort((a, b) => a.timestamp - b.timestamp);
    return runs;
  }

  async prune(max = 500): Promise<void> {
    const runs = await this.list();
    if (runs.length <= max) return;
    runs.sort((a, b) => a.score + a.novelty - (b.score + b.novelty));
    const remove = runs.slice(0, runs.length - max);
    await Promise.all(remove.map((r) => del(r.id, this.store)));
  }

  async selectParents(count: number, beta = 1, gamma = 1): Promise<InsightRun[]> {
    const runs = await this.list();
    if (!runs.length) return [];
    const scoreW = runs.map((r) => Math.exp(beta * r.score));
    const novW = runs.map((r) => Math.exp(gamma * r.novelty));
    const sumS = scoreW.reduce((a, b) => a + b, 0);
    const sumN = novW.reduce((a, b) => a + b, 0);
    const weights = runs.map((_, i) => (scoreW[i] / sumS) * (novW[i] / sumN));
    const selected: InsightRun[] = [];
    for (let i = 0; i < Math.min(count, runs.length); i++) {
      let r = Math.random();
      let idx = 0;
      for (; idx < weights.length; idx++) {
        if (r < weights[idx]) break;
        r -= weights[idx];
      }
      selected.push(runs[idx]);
    }
    return selected;
  }
}
