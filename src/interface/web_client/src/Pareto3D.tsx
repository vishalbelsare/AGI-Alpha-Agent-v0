// SPDX-License-Identifier: Apache-2.0
import React, { useEffect } from 'react';
import Plotly from 'plotly.js-dist';

export interface PopulationMember {
  effectiveness: number;
  risk: number;
  complexity: number;
  rank: number;
  impact: number;
}

interface Props {
  data: PopulationMember[];
}

export default function Pareto3D({ data }: Props) {
  useEffect(() => {
    if (!data.length) return;
    Plotly.react(
      'pareto3d',
      [
        {
          x: data.map((p) => p.effectiveness),
          y: data.map((p) => p.risk),
          z: data.map((p) => p.complexity),
          mode: 'markers',
          type: 'scatter3d',
          marker: { color: data.map((p) => p.impact), size: 3 },
        },
      ],
      {
        margin: { t: 0 },
        scene: {
          xaxis: { title: 'Effectiveness' },
          yaxis: { title: 'Risk' },
          zaxis: { title: 'Complexity' },
        },
      },
    );
  }, [data]);

  return <div id="pareto3d" className="w-full h-[400px]" />;
}
