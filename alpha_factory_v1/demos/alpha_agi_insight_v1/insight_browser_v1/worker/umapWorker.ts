// SPDX-License-Identifier: Apache-2.0
import { loadPyodide } from '../lib/pyodide.js';
import type { Individual } from '../src/state/serializer.ts';

let pyReady: any;
async function initPy(): Promise<any> {
  if (!pyReady) {
    pyReady = await loadPyodide({ indexURL: './wasm/' }).catch(() => null);
  }
  return pyReady;
}

async function embedTexts(texts: string[]): Promise<[number, number][]> {
  const py = await initPy();
  if (!py) return texts.map(() => [Math.random(), Math.random()]);
  try {
    py.globals.set('texts', texts);
    await py.runPythonAsync(`import json\nfrom sentence_transformers import SentenceTransformer\nfrom umap import UMAP\n_model = SentenceTransformer('all-MiniLM-L6-v2')\n_emb = _model.encode(texts, normalize_embeddings=True)\n_coords = UMAP(n_components=2).fit_transform(_emb)\nresult = json.dumps(_coords.tolist())`);
    const res = py.globals.get('result');
    return JSON.parse(res);
  } catch {
    return texts.map(() => [Math.random(), Math.random()]);
  }
}

interface UmapRequest {
  population: (Individual & { summary?: string })[];
}

self.onmessage = async (ev: MessageEvent<UmapRequest>) => {
  const { population } = ev.data;
  const texts = population.map((p) => p.summary || '');
  const coords = await embedTexts(texts);
  const out = population.map((p, i) => ({ ...p, umap: coords[i] }));
  self.postMessage(out);
};
