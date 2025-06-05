// SPDX-License-Identifier: Apache-2.0

import type { Individual } from '../state/serializer.ts';

/** GPU availability and usage toggle */
export interface GpuToggleEvent {
  /** Whether WebGPU is available */
  gpu: boolean;
  /** User preference to use the GPU */
  use: boolean;
}

/** Summary of a simulator generation */
export interface SimulatorStatus {
  /** Current generation number */
  gen: number;
  /** Number of individuals on the current front */
  frontSize: number;
}

/** A stored population frame used for replay */
export interface PopulationFrame {
  population: Individual[];
  gen: number;
}
