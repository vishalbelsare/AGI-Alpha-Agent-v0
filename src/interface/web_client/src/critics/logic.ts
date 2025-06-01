// SPDX-License-Identifier: Apache-2.0
export async function loadExamples(url = '/data/critics/innovations.txt'): Promise<string[]> {
  try {
    const res = await fetch(url);
    if (!res.ok) return [];
    const text = await res.text();
    return text
      .split(/\n/)
      .map((l) => l.trim())
      .filter((l) => l.length > 0);
  } catch {
    return [];
  }
}

export class LogicCritic {
  examples: string[];
  index: Record<string, number>;
  scale: number;

  constructor(examples: string[] = []) {
    this.examples = examples;
    this.index = {};
    this.examples.forEach((e, i) => {
      this.index[e.toLowerCase()] = i;
    });
    this.scale = Math.max(this.examples.length - 1, 1);
  }

  score(genome: string | number[]): number {
    const key = String(genome).toLowerCase();
    const pos = this.index[key] ?? -1;
    const base = pos >= 0 ? (pos + 1) / (this.scale + 1) : 0;
    const noise = Math.random() * 0.001;
    const val = base + noise;
    return Math.min(1, Math.max(0, val));
  }
}
