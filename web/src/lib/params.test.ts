import { expect, test } from 'bun:test';
import { breakdown, defaultShape } from './params';

test('default shape matches the PyTorch module', () => {
  const report = breakdown(defaultShape);
  expect(report.valid).toBe(true);
  expect(report.total).toBe(1_035_008);
  expect(report.tokenEmbedding).toBe(256 * 128);
  expect(report.positionEmbedding).toBe(128 * 128);
});

test('rejects a width that the heads cannot split', () => {
  const report = breakdown({ ...defaultShape, nEmbd: 100, nHead: 6 });
  expect(report.valid).toBe(false);
});
