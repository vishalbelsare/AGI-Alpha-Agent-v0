// SPDX-License-Identifier: Apache-2.0
import React, { useEffect } from 'react';
import Plotly from 'plotly.js-dist';

export interface LineageNode {
  id: number;
  parent?: number | null;
  patch?: string | null;
  pass_rate: number;
}

interface Props {
  data: LineageNode[];
}

export default function LineageTree({ data }: Props) {
  useEffect(() => {
    if (!data.length) return;
    const idMap: Record<number, LineageNode> = {};
    for (const n of data) idMap[n.id] = n;

    const ids = data.map((n) => String(n.id));
    const parents = data.map((n) => (n.parent ? String(n.parent) : ''));
    const passRates = data.map((n) => n.pass_rate);
    const custom = data.map((n) => {
      const parent = n.parent ? idMap[n.parent] : undefined;
      const delta = parent ? n.pass_rate - parent.pass_rate : n.pass_rate;
      return [n.patch ?? '', delta];
    });

    Plotly.react(
      'lineage-tree',
      [
        {
          type: 'treemap',
          ids,
          parents,
          values: ids.map(() => 1),
          customdata: custom,
          text: ids,
          hovertemplate: 'patch=%{customdata[0]}<br>\u0394pass-rate=%{customdata[1]:.2f}<extra></extra>',
          marker: { colors: passRates, colorscale: 'Blues' },
        },
      ],
      { margin: { t: 0 } },
    );
  }, [data]);

  return <div id="lineage-tree" className="w-full h-[400px]" />;
}
