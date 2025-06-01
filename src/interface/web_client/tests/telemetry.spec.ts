import { test, expect } from '@playwright/test';

test('no telemetry when disabled', async ({ page }) => {
  const requests: string[] = [];
  await page.route('**/v1/traces', route => {
    requests.push(route.request().url());
    route.fulfill({ status: 200, body: '' });
  });
  await page.addInitScript(() => {
    localStorage.setItem('telemetryConsent', 'false');
  });
  await page.goto('/');
  await page.click('text=Run');
  await page.waitForTimeout(1000);
  expect(requests.length).toBe(0);
});
