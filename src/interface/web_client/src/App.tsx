// SPDX-License-Identifier: Apache-2.0
import React, { useEffect, useState, FormEvent } from 'react';
import Plotly from 'plotly.js-dist';

interface SectorData {
  name: string;
  energy: number;
}

interface ForecastPoint {
  year: number;
  capability: number;
  sectors: SectorData[];
}

interface PopulationMember {
  effectiveness: number;
  risk: number;
  complexity: number;
  rank: number;
}

interface ResultsResponse {
  forecast: ForecastPoint[];
  population?: PopulationMember[];
}

interface RunsResponse {
  ids: string[];
}

export default function App() {
  const [horizon, setHorizon] = useState(5);
  const [popSize, setPopSize] = useState(6);
  const [generations, setGenerations] = useState(3);
  const [data, setData] = useState<ForecastPoint[]>([]);
  const [population, setPopulation] = useState<PopulationMember[]>([]);
  const [runs, setRuns] = useState<string[]>([]);
  const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');
  const TOKEN = import.meta.env.VITE_API_TOKEN ?? '';

  useEffect(() => {
    refreshRuns();
    fetchLatest();
  }, []);

  async function fetchLatest() {
    try {
      const res = await fetch(`${API_BASE}/results`, {
        headers: TOKEN ? { Authorization: `Bearer ${TOKEN}` } : {},
      });
      if (res.ok) {
        const body: ResultsResponse = await res.json();
        setData(body.forecast);
        setPopulation(body.population ?? []);
      }
    } catch {
      // ignore
    }
  }

  async function refreshRuns() {
    try {
      const res = await fetch(`${API_BASE}/runs`, {
        headers: TOKEN ? { Authorization: `Bearer ${TOKEN}` } : {},
      });
      if (res.ok) {
        const body: RunsResponse = await res.json();
        setRuns(body.ids.slice(-20).reverse());
      }
    } catch {
      // ignore
    }
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    try {
      await fetch(`${API_BASE}/simulate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(TOKEN ? { Authorization: `Bearer ${TOKEN}` } : {}),
        },
        body: JSON.stringify({
          horizon: Number(horizon),
          pop_size: Number(popSize),
          generations: Number(generations),
        }),
      });
      await fetchLatest();
      await refreshRuns();
    } catch {
      // ignore
    }
  }

  useEffect(() => {
    if (!data.length) return;
    const years = data.map((p) => p.year);
    const cap = data.map((p) => p.capability);
    Plotly.react('capability', [
      { x: years, y: cap, mode: 'lines', type: 'scatter' },
    ], { margin: { t: 20 } });

    const bySector: Record<string, { x: number[]; y: number[] }> = {};
    for (const pt of data) {
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
  }, [data]);

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
          Population
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
        <button type="submit">Run simulation</button>
      </form>
      <button type="button" onClick={refreshRuns}>Refresh summaries</button>
      <div id="sectors" style={{ width: '100%', height: 300 }} />
      <div id="capability" style={{ width: '100%', height: 300 }} />
      <div id="pareto" style={{ width: '100%', height: 400 }} />
      <h2>Last 20 simulations</h2>
      <ul>
        {runs.map((r) => (
          <li key={r}>{r}</li>
        ))}
      </ul>
    </div>
  );
}
