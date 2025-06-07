// SPDX-License-Identifier: Apache-2.0
import { loadPyodide } from '../lib/pyodide.js';
import type { Individual } from '../src/state/serializer.ts';
import { t } from '../src/ui/i18n.js';

self.onerror = ((e: ErrorEvent) => {
  self.postMessage({
    type: 'error',
    message: e.message,
    url: (e as ErrorEvent).filename,
    line: (e as ErrorEvent).lineno,
    column: (e as ErrorEvent).colno,
    stack: (e as ErrorEvent).error?.stack,
    ts: Date.now(),
  });
}) as any;
self.onunhandledrejection = ((ev: PromiseRejectionEvent) => {
  const reason: any = ev.reason || {};
  self.postMessage({
    type: 'error',
    message: reason.message ? String(reason.message) : String(reason),
    stack: reason.stack,
    ts: Date.now(),
  });
}) as any;

interface Pyodide {
  globals: { set(key: string, value: unknown): void; get(key: string): string };
  runPythonAsync(code: string): Promise<unknown>;
}

let pyReady: Pyodide | null = null;
async function initPy(): Promise<Pyodide | null> {
  if (!pyReady) {
    pyReady = await loadPyodide({ indexURL: './wasm/' }).catch(() => null);
    if (!pyReady) self.postMessage({ toast: t('pyodide_failed') });
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
