// SPDX-License-Identifier: Apache-2.0
import { lcg } from './utils/rng.js';
import { mutate } from './evolve/mutate.js';
import { paretoFront } from './utils/pareto.js';

export interface SimulatorOptions {
  popSize: number;
  generations: number;
  mutations?: string[];
  seed?: number;
  workerUrl?: string;
}

export interface Generation {
  gen: number;
  pop: any[];
}

export class Simulator {
  public readonly opts: SimulatorOptions;
  private rand: ReturnType<typeof lcg>;
  private worker: Worker | null = null;
  private _cancelled = false;

  constructor(opts: SimulatorOptions) {
    this.opts = { mutations: ['gaussian'], seed: 1, ...opts };
    this.rand = lcg(this.opts.seed ?? 1);
  }

  get cancelled(): boolean {
    return this._cancelled;
  }

  cancel() {
    this._cancelled = true;
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
  }

  async *run(): AsyncGenerator<Generation> {
    let pop = Array.from({ length: this.opts.popSize }, () => ({
      logic: this.rand(),
      feasible: this.rand(),
      strategy: 'base',
    }));
    for (let gen = 0; gen < this.opts.generations && !this._cancelled; gen++) {
      if (this.opts.workerUrl && typeof Worker !== 'undefined') {
        if (!this.worker) {
          this.worker = new Worker(this.opts.workerUrl, { type: 'module' });
        }
        const result: any = await new Promise((resolve) => {
          if (!this.worker) return resolve({ pop, rngState: this.rand.state() });
          this.worker.onmessage = (ev) => resolve(ev.data);
          this.worker.postMessage({
            pop,
            rngState: this.rand.state(),
            mutations: this.opts.mutations,
            popSize: this.opts.popSize,
          });
        });
        pop = result.pop;
        this.rand.set(result.rngState);
      } else {
        pop = mutate(pop, this.rand, this.opts.mutations ?? ['gaussian']);
        const front = paretoFront(pop);
        pop.forEach((d) => (d.front = front.includes(d)));
        pop = front.concat(pop.slice(0, this.opts.popSize - 10));
      }
      yield { gen: gen + 1, pop };
    }
    this.cancel();
  }
}
