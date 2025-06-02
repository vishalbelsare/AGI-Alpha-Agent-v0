import { test, expect } from '@playwright/test';

test('renders pareto front and timeline', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('#lineage-tree');
  await expect(page.locator('#pareto3d')).toBeVisible();
  await expect(page.locator('#lineage-timeline')).toBeVisible();
});

test('rationale panel fits mobile viewport', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 667 });
  await page.goto('/');
  await page.click('text=Show details');
  await page.click('text=Learn more');
  const box = await page.locator('.modal-content').boundingBox();
  expect(box?.width).toBeLessThanOrEqual(375);
});
