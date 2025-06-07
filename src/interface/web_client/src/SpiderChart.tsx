// SPDX-License-Identifier: Apache-2.0
import React from 'react';

interface Props {
  labels: string[];
  values: number[];
  size?: number;
}

export default function SpiderChart({ labels, values, size = 200 }: Props) {
  const center = size / 2;
  const radius = center - 20;
  const step = (Math.PI * 2) / labels.length;

  const points = labels
    .map((_, i) => {
      const angle = i * step - Math.PI / 2;
      const r = radius * (values[i] ?? 0);
      const x = center + r * Math.cos(angle);
      const y = center + r * Math.sin(angle);
      return `${x},${y}`;
    })
    .join(' ');

  return (
    <svg width={size} height={size}>
      {labels.map((label, i) => {
        const angle = i * step - Math.PI / 2;
        const x = center + radius * Math.cos(angle);
        const y = center + radius * Math.sin(angle);
        const tx = center + (radius + 12) * Math.cos(angle);
        const ty = center + (radius + 12) * Math.sin(angle);
        return (
          <g key={label}>
            <line x1={center} y1={center} x2={x} y2={y} stroke="#ccc" />
            <text x={tx} y={ty} textAnchor="middle" dominantBaseline="middle" fontSize="10">
              {label}
            </text>
          </g>
        );
      })}
      <polygon points={points} fill="rgba(0,100,250,0.3)" stroke="blue" />
    </svg>
  );
}
