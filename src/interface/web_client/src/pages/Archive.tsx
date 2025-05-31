// SPDX-License-Identifier: Apache-2.0
import React, { useEffect, useState } from 'react';

interface Agent {
  hash: string;
  parent: string | null;
  score: number;
}

interface TimelinePoint {
  tool: string;
  ts: number;
}

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');

export default function Archive() {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [diff, setDiff] = useState('');
  const [timeline, setTimeline] = useState<TimelinePoint[]>([]);
  const [parent, setParent] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/archive`)
      .then((res) => res.ok ? res.json() : [])
      .then(setAgents)
      .catch(() => setAgents([]));
  }, []);

  async function select(a: Agent) {
    setParent(a.parent);
    try {
      const res = await fetch(`${API_BASE}/archive/${a.hash}/diff`);
      if (res.ok) setDiff(await res.text());
    } catch {
      setDiff('');
    }
    try {
      const res = await fetch(`${API_BASE}/archive/${a.hash}/timeline`);
      if (res.ok) setTimeline(await res.json());
    } catch {
      setTimeline([]);
    }
  }

  return (
    <div>
      <h1>Agent Archive</h1>
      <ul>
        {agents.map((a) => (
          <li key={a.hash} className="agent-row">
            <button type="button" onClick={() => select(a)}>{a.hash}</button>
            <a href={`${API_BASE}/archive/${a.hash}/diff`} download>
              download
            </a>
          </li>
        ))}
      </ul>
      {diff && (
        <div>
          <pre className="diff">{diff}</pre>
          {parent && (
            <a href={`/archive/${parent}`} className="parent-link">Parent</a>
          )}
          <ul>
            {timeline.map((t) => (
              <li key={t.ts}>{t.tool} - {t.ts}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
