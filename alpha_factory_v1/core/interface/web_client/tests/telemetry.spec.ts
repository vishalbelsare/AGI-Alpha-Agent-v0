import { test, expect } from '@playwright/test';

test('no telemetry when disabled', async ({ page }) => {
  const requests: string[] = [];
  await page.route('**/telemetry', route => {
    requests.push(route.request().url());
    route.fulfill({ status: 200, body: '' });
  });
  await page.addInitScript(() => {
    // provide endpoint so requests would fire if consented
    (window as any).OTEL_ENDPOINT = 'https://example.com/telemetry';
    localStorage.setItem('telemetryConsent', 'false');
  });
  await page.goto('/');
  await page.click('text=Run');
  await page.waitForTimeout(1000);
  expect(requests.length).toBe(0);
});
