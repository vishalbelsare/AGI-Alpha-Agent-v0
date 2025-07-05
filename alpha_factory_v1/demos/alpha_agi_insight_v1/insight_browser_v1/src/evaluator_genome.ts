// SPDX-License-Identifier: Apache-2.0
export interface EvaluatorGenome {
  weights: { logic: number; feasible: number };
  prompt: string;
}

function clamp(v: number): number {
  return Math.min(1, Math.max(0, v));
}

export function mutateEvaluator(
  base: EvaluatorGenome,
  rand: () => number = Math.random
): EvaluatorGenome {
  let l = clamp(base.weights.logic + (rand() - 0.5) * 0.1);
  let f = clamp(base.weights.feasible + (rand() - 0.5) * 0.1);
  const sum = l + f || 1;
  l /= sum;
  f /= sum;
  const words = ['innovative', 'efficient', 'robust', 'scalable'];
  let tokens = base.prompt.split(/\s+/);
  if (rand() < 0.5 && tokens.length > 1) {
    tokens.splice(Math.floor(rand() * tokens.length), 1);
  } else {
    const w = words[Math.floor(rand() * words.length)];
    tokens.splice(Math.floor(rand() * (tokens.length + 1)), 0, w);
  }
  const prompt = tokens.join(' ');
  return { weights: { logic: l, feasible: f }, prompt };
}
