import React, { useEffect, useState } from 'react';
import Plotly from 'plotly.js-dist';

interface ProgressMsg {
  id: string;
  year: number;
  capability: number;
}

const API_TOKEN = import.meta.env.VITE_API_TOKEN || 'demo-token';

export default function App() {
  const [horizon, setHorizon] = useState(5);
  const [popSize, setPopSize] = useState(6);
  const [generations, setGenerations] = useState(3);
  const [runId, setRunId] = useState<string | null>(null);
  const [years, setYears] = useState<number[]>([]);
  const [caps, setCaps] = useState<number[]>([]);

  async function start(e: React.FormEvent) {
    e.preventDefault();
    const res = await fetch('/simulate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${API_TOKEN}`,
      },
      body: JSON.stringify({ horizon, pop_size: popSize, generations }),
    });
    const data = await res.json();
    setRunId(data.id);
    setYears([]);
    setCaps([]);
  }

  useEffect(() => {
    if (!runId) return;
    const ws = new WebSocket(`ws://${window.location.host}/ws/progress`);
    ws.onopen = () => ws.send('ready');
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data) as ProgressMsg;
        if (msg.id === runId) {
          setYears((y) => [...y, msg.year]);
          setCaps((c) => [...c, msg.capability]);
        }
      } catch {
        /* ignore */
      }
    };
    return () => ws.close();
  }, [runId]);

  useEffect(() => {
    if (years.length) {
      Plotly.react('chart', [
        { x: years, y: caps, mode: 'lines+markers', type: 'scatter' },
      ], { margin: { t: 20 } });
    }
  }, [years, caps]);

  return (
    <div>
      <h2>α‑AGI Insight</h2>
      <form onSubmit={start} style={{ marginBottom: '1em' }}>
        <label>
          Horizon
          <input type="number" value={horizon} onChange={(e) => setHorizon(Number(e.target.value))} />
        </label>
        <label>
          Population
          <input type="number" value={popSize} onChange={(e) => setPopSize(Number(e.target.value))} />
        </label>
        <label>
          Generations
          <input type="number" value={generations} onChange={(e) => setGenerations(Number(e.target.value))} />
        </label>
        <button type="submit">Run</button>
      </form>
      <div id="chart" style={{ width: '100%', height: 400 }} />
    </div>
  );
}
