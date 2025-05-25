import React, { useEffect, useState } from 'react';
import Plotly from 'plotly.js-dist';

interface ForecastPoint {
  year: number;
  capability: number;
}

interface ResultsResponse {
  forecast: ForecastPoint[];
}

export default function App() {
  const [data, setData] = useState<ForecastPoint[]>([]);

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch('/results');
        if (res.ok) {
          const body: ResultsResponse = await res.json();
          setData(body.forecast);
        }
      } catch {
        // ignore network errors in demo
      }
    }
    load();
  }, []);

  useEffect(() => {
    if (data.length) {
      Plotly.react('timeline', [
        {
          x: data.map(p => p.year),
          y: data.map(p => p.capability),
          mode: 'lines+markers',
          type: 'scatter',
        },
      ], {});
    }
  }, [data]);

  return (
    <div>
      <h1>Disruption Timeline</h1>
      <div id="timeline" style={{ width: '100%', height: 300 }} />
    </div>
  );
}
