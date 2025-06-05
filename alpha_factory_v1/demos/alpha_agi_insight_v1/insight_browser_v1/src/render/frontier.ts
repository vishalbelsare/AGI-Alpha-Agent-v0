// @ts-nocheck
// SPDX-License-Identifier: Apache-2.0
import * as Plot from '@observablehq/plot';
import { plotCanvas } from './plotCanvas.ts';
import { paretoFront } from '../utils/pareto.ts';
import { depthColor } from './colors.ts';
import { drawHeatmap } from './canvasLayer.ts';

export function renderFrontier(
  container: HTMLElement,
  pop: any[],
  onSelect?: (d: any, el: SVGCircleElement) => void,
): void {
  const front = paretoFront(pop).sort((a, b) => a.logic - b.logic);

  const maxDepth = pop.reduce((m, d) => Math.max(m, d.depth ?? 0), 0);
  const dotOptions = {
    x: 'logic',
    y: 'feasible',
    r: 3,
    fill: (d) => depthColor(d.depth ?? 0, maxDepth),
    title: (d) => `${d.summary ?? ''}\n${d.critic ?? ''}`,
  };

  const marks = [
    Plot.areaY(front, {
      x: 'logic',
      y: 'feasible',
      fill: 'rgba(0,175,255,0.2)',
      stroke: null,
    }),
  ];

  marks.push(
    pop.length > 10000 ? plotCanvas(Plot.dot(pop, dotOptions)) : Plot.dot(pop, dotOptions),
  );

  const plot = Plot.plot({
    width: 500,
    height: 500,
    x: { domain: [0, 1] },
    y: { domain: [0, 1] },
    marks,
  });

  container.innerHTML = '';
  container.append(plot);
  const svg = plot.querySelector('svg') || plot;
  drawHeatmap(svg, pop, (d) => d.logic * 500, (d) => (1 - d.feasible) * 500);
  if (onSelect) {
    d3.select(plot).selectAll('circle').on('click', function (_, d) {
      onSelect(d, this);
    });
  }
}
