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

export default function LineageTree({ data }: Props) {
  useEffect(() => {
    if (!data.length) return;
    const ids = data.map((n) => String(n.id));
    const parents = data.map((n) => (n.parent ? String(n.parent) : ''));
    const diffs = data.map((n) => n.diff ?? '');
    const passRates = data.map((n) => n.pass_rate);
    Plotly.react(
      'lineage-tree',
      [
        {
          type: 'treemap',
          ids,
          parents,
          values: ids.map(() => 1),
          customdata: diffs,
          text: ids.map((id, i) => (diffs[i] ? `<a href='${diffs[i]}'>${id}</a>` : id)),
          hovertemplate: 'hash=%{id}<br>diff=%{customdata}<extra></extra>',
          marker: { colors: passRates, colorscale: 'Blues' },
        },
      ],
      { margin: { t: 0 } },
    );
  }, [data]);

  return <div id="lineage-tree" style={{ width: '100%', height: 400 }} />;
}
