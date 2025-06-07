// SPDX-License-Identifier: Apache-2.0
import { initTelemetry } from '../../../telemetry.js';

interface Recorder {
  recordRun: (n: number) => void;
  recordShare: () => void;
}

class Telemetry {
  private rec: Recorder = { recordRun() {}, recordShare() {} };

  requestConsent(msg: string): void {
    this.rec = initTelemetry(() => msg);
  }

  recordRun(generations: number): void {
    this.rec.recordRun(generations);
  }

  recordShare(): void {
    this.rec.recordShare();
  }
}

export default new Telemetry();
