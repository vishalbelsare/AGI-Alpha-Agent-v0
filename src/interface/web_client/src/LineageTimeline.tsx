// SPDX-License-Identifier: Apache-2.0
import React, { useEffect } from 'react';
import Plotly from 'plotly.js-dist';

export interface LineageNode {
  id: number;
  parent?: number | null;
  diff?: string | null;
  pass_rate: number;
}

interface Props {
  data: LineageNode[];
}

function computeDepths(nodes: LineageNode[]): Map<number, number> {
  const map = new Map<number, LineageNode>();
  for (const n of nodes) map.set(n.id, n);
  const depths = new Map<number, number>();
  function depth(id: number): number {
    const cached = depths.get(id);
    if (cached !== undefined) return cached;
    const node = map.get(id);
    if (!node || !node.parent) {
      depths.set(id, 0);
      return 0;
    }
    const d = depth(node.parent) + 1;
    depths.set(id, d);
    return d;
  }
  for (const n of nodes) depth(n.id);
  return depths;
}

export default function LineageTimeline({ data }: Props) {
  useEffect(() => {
    if (!data.length) return;
    const nodes = [...data].sort((a, b) => a.id - b.id);
    const index = new Map<number, number>();
    nodes.forEach((n, i) => index.set(n.id, i));
    const depths = computeDepths(nodes);

    const lines: Plotly.Data[] = [];
    for (const n of nodes) {
      if (!n.parent) continue;
      const px = index.get(n.parent);
      const py = depths.get(n.parent);
      const cx = index.get(n.id);
      const cy = depths.get(n.id);
      if (px === undefined || py === undefined || cx === undefined || cy === undefined) continue;
      lines.push({
        x: [px, cx],
        y: [py, cy],
        mode: 'lines',
        type: 'scatter',
        line: { color: '#888' },
        hoverinfo: 'skip',
        showlegend: false,
      });
    }

    lines.push({
      x: nodes.map((_, i) => i),
      y: nodes.map((n) => depths.get(n.id) ?? 0),
      text: nodes.map((n) => String(n.id)),
      mode: 'markers+text',
      type: 'scatter',
      marker: { color: nodes.map((n) => n.pass_rate), colorscale: 'Blues', size: 8 },
    });

    Plotly.react('lineage-timeline', lines, {
      margin: { t: 20 },
      xaxis: { title: 'Generation' },
      yaxis: { title: 'Depth' },
      hovermode: 'closest',
    });
  }, [data]);

  return <div id="lineage-timeline" style={{ width: '100%', height: 400 }} />;
}
