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

export class FeasibilityCritic {
  examples: string[];

  constructor(examples: string[] = []) {
    this.examples = examples;
  }

  private static jaccard(a: string[], b: string[]): number {
    const sa = new Set(a);
    const sb = new Set(b);
    if (!sa.size || !sb.size) return 0;
    let intersection = 0;
    for (const x of sa) if (sb.has(x)) intersection++;
    const union = new Set([...a, ...b]).size;
    return intersection / union;
  }

  score(genome: string | number[]): number {
    const tokens = String(genome).toLowerCase().split(/\s+/);
    let best = 0;
    for (const ex of this.examples) {
      const sim = FeasibilityCritic.jaccard(tokens, ex.toLowerCase().split(/\s+/));
      if (sim > best) best = sim;
    }
    const noise = Math.random() * 0.001;
    const val = best + noise;
    return Math.min(1, Math.max(0, val));
  }
}
