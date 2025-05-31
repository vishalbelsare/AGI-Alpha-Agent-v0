// SPDX-License-Identifier: Apache-2.0
import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

export interface LineageNode {
  id: number;
  parent?: number | null;
  diff?: string | null;
  pass_rate: number;
}

interface Props {
  data: LineageNode[];
}

export default function D3LineageTree({ data }: Props) {
  const ref = useRef<SVGSVGElement | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();
    if (!data.length) return;

    const stratify = d3
      .stratify<LineageNode>()
      .id((d) => String(d.id))
      .parentId((d) => (d.parent ? String(d.parent) : null));

    const root = stratify(data);
    const tree = d3.tree<typeof root>().size([800, 400]);
    const nodes = tree(root);

    const g = svg.append('g').attr('transform', 'translate(40,20)');

    g.selectAll('line')
      .data(nodes.links())
      .enter()
      .append('line')
      .attr('x1', (d) => d.source.x)
      .attr('y1', (d) => d.source.y)
      .attr('x2', (d) => d.target.x)
      .attr('y2', (d) => d.target.y)
      .attr('stroke', '#999');

    const node = g
      .selectAll('circle')
      .data(nodes.descendants())
      .enter()
      .append('circle')
      .attr('cx', (d) => d.x)
      .attr('cy', (d) => d.y)
      .attr('r', 4)
      .attr('fill', (d) => d3.interpolateBlues(d.data.pass_rate))
      .style('cursor', 'pointer')
      .on('click', (event, d) => setSelected(d.data.diff ?? null));

    node.append('title').text((d) => `pass=${d.data.pass_rate}`);
  }, [data]);

  return (
    <div>
      <svg ref={ref} width={840} height={440} />
      {selected && <pre className="diff">{selected}</pre>}
    </div>
  );
}
