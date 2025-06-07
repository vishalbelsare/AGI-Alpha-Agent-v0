// SPDX-License-Identifier: Apache-2.0
export interface Point { logic: number; feasible: number; }

export function paretoEntropy(points: Point[], bins = 10): number {
  if (!points.length) return 0;
  const hist = new Array(bins * bins).fill(0);
  for (const p of points) {
    const x = Math.max(0, Math.min(bins - 1, Math.floor((p.logic ?? 0) * bins)));
    const y = Math.max(0, Math.min(bins - 1, Math.floor((p.feasible ?? 0) * bins)));
    hist[y * bins + x] += 1;
  }
  const total = points.length;
  return -hist.reduce((s, c) => {
    if (!c) return s;
    const prob = c / total;
    return s + prob * Math.log2(prob);
  }, 0);
}
