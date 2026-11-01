import { expect, test } from 'bun:test';
import { defaultShape } from './params.ts';
import { sanitizeBench, sanitizeLab, sanitizeSample, sanitizeStage, sanitizeTokenize } from './persist.ts';

test('lab snapshot keeps a valid shape and drops a broken record', () => {
  const saved = sanitizeLab({
    shape: { ...defaultShape, nLayer: 3, nEmbd: 64 },
    prompt: 'Question: What is the bell?\nAnswer:',
    completion: ' bronze\n',
    temperature: 0.2
  });
  expect(saved?.shape.nLayer).toBe(3);
  expect(saved?.shape.vocabSize).toBe(8192);
  expect(saved?.temperature).toBe(0.2);
  expect(sanitizeLab({ shape: defaultShape, prompt: 'x' })).toBeNull();
});

test('bench, sample, and tokenize snapshots clamp to the form limits', () => {
  const bench = sanitizeBench({
    stepCount: 99,
    ftCount: 4,
    rate: 0.001,
    batchSize: 8,
    seed: 11,
    deviceChoice: 'cpu',
    gfxOverride: true
  });
  expect(bench).toEqual({
    stepCount: 99,
    ftCount: 4,
    rate: 0.001,
    batchSize: 8,
    seed: 11,
    deviceChoice: 'cpu',
    gfxOverride: true
  });
  expect(sanitizeBench(null)).toBeNull();
  expect(sanitizeSample({ prompt: 'lamp', maxNewTokens: 12, topK: 4 })?.maxNewTokens).toBe(12);
  expect(sanitizeTokenize({ text: 'abc', block: 8, cursor: 2 })?.cursor).toBe(2);
});

test('stage restore keeps the clock and mode, and an unknown mode falls back', () => {
  const stage = sanitizeStage('/train', { time: 4.5, speed: 2, mode: 'loss', filter: 'all', layer: 0, head: 0, query: 0, exploded: false });
  expect(stage.time).toBe(4.5);
  expect(stage.speed).toBe(2);
  expect(stage.mode).toBe('loss');
  expect(sanitizeStage('/train', { mode: 'forward', speed: 9, time: -3 }).mode).toBe('step');
  expect(sanitizeStage('/architecture', null).mode).toBe('forward');
});
