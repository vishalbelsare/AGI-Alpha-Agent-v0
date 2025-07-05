// SPDX-License-Identifier: Apache-2.0
const path = require('path');

jest.mock('../src/evolve/mutate.ts', () => ({
  mutate: jest.fn(() => [])
}));

const { mutate } = require('../src/evolve/mutate.ts');

function makeMsg(gen) {
  return { pop: [], rngState: 1, mutations: [], popSize: 1, critic: 'none', gen };
}

test('worker updates gpu flag before mutate calls', async () => {
  const selfObj = { navigator: {}, postMessage: jest.fn() };
  global.self = selfObj;
  await import('../worker/evolver.ts');
  const handler = selfObj.onmessage;

  handler({ data: { type: 'gpu', available: true } });
  await handler({ data: makeMsg(1) });
  expect(mutate.mock.calls[0][6]).toBe(true);

  handler({ data: { type: 'gpu', available: false } });
  await handler({ data: makeMsg(2) });
  expect(mutate.mock.calls[1][6]).toBe(false);
});
