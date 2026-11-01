import { expect, test } from 'bun:test';
import { rankNext } from './distribution.ts';

test('temperature zero keeps only the largest logit', () => {
  const ranked = rankNext([0.2, 3.5, 1.1, -4], 0, 0, 8);
  expect(ranked.pieces).toHaveLength(1);
  expect(ranked.pieces[0].id).toBe(1);
  expect(ranked.pieces[0].prob).toBe(1);
  expect(ranked.rest).toBe(0);
});

test('top-k drops the tail and the listed mass plus rest is one', () => {
  const logits = [1, 2, 3, 0, -1];
  const ranked = rankNext(logits, 1, 2, 1);
  expect(ranked.pieces).toHaveLength(1);
  expect(ranked.pieces[0].id).toBe(2);
  expect(ranked.pieces[0].prob + ranked.rest).toBeCloseTo(1, 6);
  expect(ranked.pieces[0].prob).toBeGreaterThan(0.5);
});
