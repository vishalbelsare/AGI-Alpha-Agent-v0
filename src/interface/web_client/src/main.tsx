// SPDX-License-Identifier: Apache-2.0
import React, { useEffect, useState, FormEvent } from 'react';
import ReactDOM from 'react-dom/client';
import Plotly from 'plotly.js-dist';

interface SectorData {
  name: string;
  energy: number;
  disrupted?: boolean;
}

interface ForecastPoint {
  year: number;
  capability: number;
  sectors?: SectorData[];
}

interface PopulationMember {
  effectiveness: number;
  risk: number;
  complexity: number;
  rank: number;
}

function Dashboard() {
  const [horizon, setHorizon] = useState(5);
  const [popSize, setPopSize] = useState(6);
  const [generations, setGenerations] = useState(3);
  const [curve, setCurve] = useState('logistic');
  const [progress, setProgress] = useState(0);
  const [runId, setRunId] = useState<string | null>(null);
  const [timeline, setTimeline] = useState<ForecastPoint[]>([]);
  const [population, setPopulation] = useState<PopulationMember[]>([]);

  const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');
  const TOKEN = import.meta.env.VITE_API_TOKEN ?? '';
  const HEADERS = TOKEN ? { Authorization: `Bearer ${TOKEN}` } : {};

  useEffect(() => {
    if (!timeline.length) return;
    const years = timeline.map((p) => p.year);
    const capability = timeline.map((p) => p.capability);
    Plotly.react('capability', [
      { x: years, y: capability, mode: 'lines', type: 'scatter' },
    ], { margin: { t: 20 } });

    const bySector: Record<string, { x: number[]; y: number[] }> = {};
    for (const pt of timeline) {
      for (const s of pt.sectors ?? []) {
        bySector[s.name] = bySector[s.name] || { x: [], y: [] };
        bySector[s.name].x.push(pt.year);
        bySector[s.name].y.push(s.energy);
      }
    }
    const traces = Object.entries(bySector).map(([name, v]) => ({
      name,
      x: v.x,
      y: v.y,
      mode: 'lines',
      type: 'scatter',
    }));
    Plotly.react('sectors', traces, { margin: { t: 20 } });
  }, [timeline]);

  useEffect(() => {
    if (!population.length) return;
    Plotly.react('pareto', [
      {
        x: population.map((p) => p.effectiveness),
        y: population.map((p) => p.risk),
        mode: 'markers',
        type: 'scatter',
        marker: { color: population.map((p) => p.rank) },
      },
    ], {
      margin: { t: 20 },
      xaxis: { title: 'Effectiveness' },
      yaxis: { title: 'Risk' },
    });
  }, [population]);

  async function fetchResults(id: string) {
    const res = await fetch(`${API_BASE}/results/${id}`, { headers: HEADERS });
    if (!res.ok) return;
    const body = await res.json();
    setTimeline(body.forecast ?? []);
    setPopulation(body.population ?? []);
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setProgress(0);
    setTimeline([]);
    setPopulation([]);

    const res = await fetch(`${API_BASE}/simulate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...HEADERS,
      },
      body: JSON.stringify({
        horizon: Number(horizon),
        pop_size: Number(popSize),
        generations: Number(generations),
        curve,
      }),
    });
    if (!res.ok) return;
    const body = await res.json();
    const id = body.id as string;
    setRunId(id);

    const wsBase = API_BASE.replace(/^http/, 'ws');
    const ws = new WebSocket(`${wsBase}/ws/progress?token=${TOKEN}`);
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.id === id && horizon) {
          setProgress(msg.year / horizon);
        }
      } catch {
        // ignore non-JSON messages
      }
    };
    ws.onclose = () => {
      fetchResults(id).catch(() => null);
    };
  }

  return (
    <div>
      <h1>AGI Simulation Dashboard</h1>
      <form onSubmit={onSubmit}>
        <label>
          Horizon
          <input
            type="number"
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}
          />
        </label>
        <label>
          Population size
          <input
            type="number"
            value={popSize}
            onChange={(e) => setPopSize(Number(e.target.value))}
          />
        </label>
        <label>
          Generations
          <input
            type="number"
            value={generations}
            onChange={(e) => setGenerations(Number(e.target.value))}
          />
        </label>
        <label>
          Curve
          <select value={curve} onChange={(e) => setCurve(e.target.value)}>
            <option value="logistic">logistic</option>
            <option value="linear">linear</option>
            <option value="exponential">exponential</option>
          </select>
        </label>
        <button type="submit">Run</button>
      </form>
      {runId && <p>Run ID: {runId}</p>}
      <progress value={progress} max={1} style={{ width: '100%' }} />
      <div id="sectors" style={{ width: '100%', height: 300 }} />
      <div id="capability" style={{ width: '100%', height: 300 }} />
      <div id="pareto" style={{ width: '100%', height: 400 }} />
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <Dashboard />,
);
