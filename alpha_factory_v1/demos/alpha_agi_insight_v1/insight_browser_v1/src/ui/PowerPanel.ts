// SPDX-License-Identifier: Apache-2.0
import { setUseGpu, gpuAvailable } from '../utils/llm.ts';

export function initPowerPanel(): { update: (e: any) => void; gpuToggle: HTMLInputElement } {
  const panel = document.createElement('div');
  panel.id = 'power-panel';
  panel.setAttribute('role', 'region');
  panel.setAttribute('aria-label', 'Power');
  Object.assign(panel.style, {
    position: 'fixed',
    top: '10px',
    left: '10px',
    background: 'rgba(0,0,0,0.7)',
    color: '#fff',
    padding: '8px',
    fontSize: '12px',
    zIndex: 1000,
    whiteSpace: 'pre',
  });
  const pre = document.createElement('pre');
  const gpuLabel = document.createElement('label');
  const gpuToggle = document.createElement('input');
  gpuToggle.type = 'checkbox';
  gpuToggle.id = 'gpu-toggle';
  gpuToggle.setAttribute('aria-label', 'Use GPU');
  gpuLabel.appendChild(gpuToggle);
  gpuLabel.append(' GPU ');
  const gpuStatus = document.createElement('span');
  gpuStatus.id = 'gpu-status';
  gpuStatus.textContent = gpuAvailable ? '(available)' : '(unavailable)';
  gpuLabel.appendChild(gpuStatus);
  panel.appendChild(gpuLabel);
  panel.appendChild(pre);
  try {
    const saved = localStorage.getItem('USE_GPU');
    gpuToggle.checked = saved !== '0';
  } catch {
    gpuToggle.checked = true;
  }
  window.USE_GPU = gpuToggle.checked && !!(navigator as any).gpu;
  setUseGpu(window.USE_GPU);
  gpuToggle.addEventListener('change', () => {
    window.USE_GPU = gpuToggle.checked && !!(navigator as any).gpu;
    setUseGpu(window.USE_GPU);
  });
  document.body.appendChild(panel);
  function update(e: unknown): void {
    pre.textContent = JSON.stringify(e, null, 2);
  }
  return { update, gpuToggle };
}
