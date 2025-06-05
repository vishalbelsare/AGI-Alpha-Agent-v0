// SPDX-License-Identifier: Apache-2.0
import createREGL from 'regl';

let ctxCache: WeakMap<HTMLCanvasElement, ReturnType<typeof createREGL>> = new WeakMap();

function parseColor(color: string): [number, number, number, number] {
  const c = document.createElement('canvas');
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = color;
  const computed = ctx.fillStyle;
  if (computed.startsWith('#')) {
    const n = parseInt(computed.slice(1), 16);
    const r = (n >> 16) & 255;
    const g = (n >> 8) & 255;
    const b = n & 255;
    return [r / 255, g / 255, b / 255, 1];
  }
  const m = computed.match(/\d+(\.\d+)?/g);
  if (m) {
    return [
      Number(m[0]) / 255,
      Number(m[1]) / 255,
      Number(m[2]) / 255,
      m[3] ? Number(m[3]) : 1,
    ];
  }
  return [0, 0, 0, 1];
}

function ensureGL(parent: any): [HTMLCanvasElement, ReturnType<typeof createREGL>] {
  const node = parent.node ? parent.node() : parent;
  let canvas = node.querySelector<HTMLCanvasElement>('canvas.webgl-layer');
  if (!canvas) {
    const svg = (node.ownerSVGElement || node) as SVGSVGElement;
    const vb = svg.viewBox?.baseVal;
    const width = vb && vb.width ? vb.width : svg.clientWidth;
    const height = vb && vb.height ? vb.height : svg.clientHeight;
    canvas = document.createElement('canvas');
    canvas.className = 'webgl-layer';
    canvas.width = width;
    canvas.height = height;
    canvas.style.position = 'absolute';
    canvas.style.left = '0';
    canvas.style.top = '0';
    canvas.style.pointerEvents = 'none';
    node.appendChild(canvas);
  }
  let regl = ctxCache.get(canvas);
  if (!regl) {
    regl = createREGL({ canvas });
    ctxCache.set(canvas, regl);
  }
  return [canvas, regl];
}

export function plotCanvas(
  parent: HTMLElement | SVGSVGElement,
  pop: any[],
  x: (d: any) => number,
  y: (d: any) => number,
  colorFn: (d: any) => string,
): void {
  const [canvas, regl] = ensureGL(parent);
  const positions = pop.map((d) => [x(d), y(d)]);
  const colors = pop.map((d) => parseColor(colorFn(d)));
  const draw = regl({
    attributes: {
      position: positions,
      color: colors,
    },
    uniforms: {
      pointSize: 6,
    },
    vert: `
    precision mediump float;
    attribute vec2 position;
    attribute vec4 color;
    uniform float pointSize;
    varying vec4 vColor;
    void main() {
      vColor = color;
      gl_PointSize = pointSize;
      gl_Position = vec4(
        position.x / ${canvas.width}.0 * 2.0 - 1.0,
        1.0 - position.y / ${canvas.height}.0 * 2.0,
        0.0,
        1.0);
    }`,
    frag: `
    precision mediump float;
    varying vec4 vColor;
    void main() {
      gl_FragColor = vColor;
    }`,
    count: positions.length,
  });
  function frame() {
    regl.clear({ color: [0, 0, 0, 0], depth: 1 });
    draw();
    requestAnimationFrame(frame);
  }
  frame();
}
