import React, { useEffect, useState } from 'react';
import Plotly from 'plotly.js-dist';

interface ForecastPoint {
  year: number;
  capability: number;
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

export default function App() {
  const [data, setData] = useState<ForecastPoint[]>([]);
  const [population, setPopulation] = useState<PopulationMember[]>([]);

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch('/results');
        if (res.ok) {
          const body: ResultsResponse = await res.json();
          setData(body.forecast);
          setPopulation(body.population ?? []);
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

  useEffect(() => {
    if (population.length) {
      Plotly.react('population', [
        {
          x: population.map(p => p.effectiveness),
          y: population.map(p => p.risk),
          z: population.map(p => p.complexity),
          mode: 'markers',
          type: 'scatter3d',
          marker: { color: population.map(p => p.rank) },
        },
      ], { scene: { xaxis: { title: 'Effectiveness' }, yaxis: { title: 'Risk' }, zaxis: { title: 'Complexity' } } });
    }
  }, [population]);

  return (
    <div>
      <h1>Disruption Timeline</h1>
      <div id="timeline" style={{ width: '100%', height: 300 }} />
      <div id="population" style={{ width: '100%', height: 400 }} />
    </div>
  );
}
