// SPDX-License-Identifier: Apache-2.0
import { t } from '../ui/i18n.ts';
interface ErrorEntry {
  type: string;
  message: string;
  url?: string;
  line?: number;
  column?: number;
  stack?: string;
  ts: number;
}
let log: ErrorEntry[] = [];

export function initErrorBoundary() {
  try {
    log = JSON.parse(localStorage.getItem('errorLog') || '[]');
  } catch {
    log = [];
  }
  function record(entry: ErrorEntry): void {
    log.push(entry);
    try {
      localStorage.setItem('errorLog', JSON.stringify(log));
    } catch {}
    if (window.toast) {
      window.toast(entry.message ? String(entry.message) : t('error_unknown'));
    }
  }
  window.onerror = (msg, url, line, col, err) => {
    record({
      type: 'error',
      message: String(msg),
      url: url || '',
      line: line || 0,
      column: col || 0,
      stack: err && err.stack,
      ts: Date.now(),
    });
  };
  window.onunhandledrejection = (ev) => {
    const reason = ev.reason || {};
    record({
      type: 'unhandledrejection',
      message: reason.message ? String(reason.message) : String(reason),
      stack: reason.stack,
      ts: Date.now(),
    });
  };
}

export function getErrorLog(): ErrorEntry[] {
  return log.slice();
}

export function clearErrorLog(): void {
  log = [];
  try {
    localStorage.removeItem('errorLog');
  } catch {}
}
