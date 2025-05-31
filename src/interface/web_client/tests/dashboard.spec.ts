import { test, expect } from '@playwright/test';

test('renders pareto front and timeline', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('#lineage-tree');
  await expect(page.locator('#pareto3d')).toBeVisible();
  await expect(page.locator('#lineage-timeline')).toBeVisible();
});
