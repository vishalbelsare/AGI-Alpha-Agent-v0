// SPDX-License-Identifier: Apache-2.0
/*
 * Simple debate arena executed in a Web Worker. The worker receives a
 * hypothesis string and runs a fixed exchange between four roles:
 * Proposer, Skeptic, Regulator and Investor. The outcome score is
 * returned to the caller along with the threaded messages.
 */
interface ArenaRequest {
  hypothesis?: string;
}

interface DebateMessage {
  role: string;
  text: string;
}

interface ArenaResult {
  messages: DebateMessage[];
  score: number;
}

self.onerror = (e) => {
  self.postMessage({
    type: 'error',
    message: e.message,
    url: (e as ErrorEvent).filename,
    line: (e as ErrorEvent).lineno,
    column: (e as ErrorEvent).colno,
    stack: (e as ErrorEvent).error?.stack,
    ts: Date.now(),
  });
};
self.onunhandledrejection = (ev) => {
  const reason: any = ev.reason || {};
  self.postMessage({
    type: 'error',
    message: reason.message ? String(reason.message) : String(reason),
    stack: reason.stack,
    ts: Date.now(),
  });
};

self.onmessage = (ev: MessageEvent<ArenaRequest>) => {
  const { hypothesis } = ev.data || {};
  if (!hypothesis) return;

  const messages = [
    { role: 'Proposer', text: `I propose that ${hypothesis}.` },
    { role: 'Skeptic', text: `I doubt that ${hypothesis} holds under scrutiny.` },
    { role: 'Regulator', text: `Any implementation of ${hypothesis} must be safe.` },
  ];

  const approved = Math.random() > 0.5;
  messages.push({
    role: 'Investor',
    text: approved
      ? `Funding approved for: ${hypothesis}.`
      : `Funding denied for: ${hypothesis}.`,
  });

  const score = approved ? 1 : 0;
  const result: ArenaResult = { messages, score };
  self.postMessage(result);
};
