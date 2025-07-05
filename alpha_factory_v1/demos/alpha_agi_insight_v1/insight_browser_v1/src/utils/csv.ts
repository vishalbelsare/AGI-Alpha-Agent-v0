// SPDX-License-Identifier: Apache-2.0
export function toCSV(rows: any[], headers?: string[]): string {
  if (!rows.length) return '';
  const keys = headers || Object.keys(rows[0]);
  const escape = (v: unknown): string => `"${String(v).replace(/"/g, '""')}"`;
  const lines = [keys.join(',')];
  for (const row of rows) {
    lines.push(keys.map((k) => escape(row[k])).join(','));
  }
  return lines.join('\n');
}
