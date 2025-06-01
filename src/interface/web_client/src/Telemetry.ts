// SPDX-License-Identifier: Apache-2.0
import { WebTracerProvider } from '@opentelemetry/sdk-trace-web';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http';
import { SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { trace } from '@opentelemetry/api';

interface TelemetryEvent {
  name: string;
  attributes?: Record<string, number | string>;
}

const BUFFER_KEY = 'telemetryBuffer';
const CONSENT_KEY = 'telemetryConsent';

class Telemetry {
  private enabled = false;
  private provider: WebTracerProvider | null = null;
  private tracer = trace.getTracer('web-client');

  requestConsent(): void {
    const stored = localStorage.getItem(CONSENT_KEY);
    if (stored === null) {
      const allow = window.confirm('Allow anonymous telemetry?');
      localStorage.setItem(CONSENT_KEY, String(allow));
      this.enabled = allow;
    } else {
      this.enabled = stored === 'true';
    }
    if (this.enabled) {
      this.start();
    }
    window.addEventListener('online', () => this.flush());
  }

  private start(): void {
    const url = import.meta.env.VITE_OTEL_EXPORTER_OTLP_ENDPOINT;
    if (!url) return;
    const exporter = new OTLPTraceExporter({ url });
    this.provider = new WebTracerProvider();
    this.provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
    this.provider.register();
    this.flush();
  }

  recordRun(generations: number): void {
    this.send({ name: 'generation_run', attributes: { generations } });
  }

  recordShare(): void {
    this.send({ name: 'share_click' });
  }

  private queue(evt: TelemetryEvent): void {
    const buf = JSON.parse(localStorage.getItem(BUFFER_KEY) ?? '[]');
    buf.push(evt);
    localStorage.setItem(BUFFER_KEY, JSON.stringify(buf));
  }

  private send(evt: TelemetryEvent): void {
    if (!this.enabled) return;
    const attempt = () => {
      if (!this.provider) return;
      const span = this.tracer.startSpan(evt.name);
      for (const [k, v] of Object.entries(evt.attributes ?? {})) {
        span.setAttribute(k, v);
      }
      span.end();
      this.provider.forceFlush().catch(() => this.queue(evt));
    };

    if (navigator.onLine) {
      try {
        attempt();
      } catch {
        this.queue(evt);
      }
    } else {
      this.queue(evt);
    }
  }

  flush(): void {
    if (!this.enabled || !navigator.onLine) return;
    const buf = JSON.parse(localStorage.getItem(BUFFER_KEY) ?? '[]');
    localStorage.removeItem(BUFFER_KEY);
    for (const evt of buf) {
      this.send(evt);
    }
  }
}

export default new Telemetry();
