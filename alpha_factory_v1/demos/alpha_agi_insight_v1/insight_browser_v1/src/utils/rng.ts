// SPDX-License-Identifier: Apache-2.0
export function lcg(seed: number) {
  function rand(): number {
    seed = Math.imul(1664525, seed) + 1013904223 >>> 0;
    return seed / 2 ** 32;
  }
  rand.state = (): number => seed;
  rand.set = (s: number): void => { seed = s >>> 0; };
  return rand;
}
