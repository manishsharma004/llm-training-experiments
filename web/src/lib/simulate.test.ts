import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { encodeText, glyphFor, installTokenizer, mergeFilm } from './bytes.ts';
import { pipelineScene, sampleAt, sampleScene, spaceScene, unigramStep, windowScene } from './simulate.ts';

const table = JSON.parse(readFileSync(path.join(import.meta.dir, '../../../tinyllm/bpe.json'), 'utf8')) as {
  pieces: string[];
  merges: [string, string][];
};
installTokenizer(table.pieces, table.merges);

test('merge film ends on the same pieces the encoder emits', () => {
  const text = 'The Old Testament';
  const film = mergeFilm(text);
  expect(film.length).toBeGreaterThan(2);
  expect(film[0].note).toContain('Bytes');
  expect(film.at(-1)?.pieces.join('')).toBe(encodeText(text).map((token) => token.glyph).join(''));
  expect(film[1].pieces.length).toBeLessThan(film[0].pieces.length);
});

test('the pipeline packet moves when time moves', () => {
  const early = pipelineScene(0.2);
  const later = pipelineScene(3.2);
  const a = early.points[0];
  const b = later.points[0];
  expect(a.x).not.toBe(b.x);
  expect(later.readout).not.toBe(early.readout);
});

test('a training window advances along the stream', () => {
  const ids = encodeText('The lamp is copper. The bell is bronze. The map is linen.');
  const first = windowScene(ids.map((token) => token.id), 0.1, 100, 10);
  const next = windowScene(ids.map((token) => token.id), 1.2, 100, 10);
  expect(first.readout).not.toBe(next.readout);
  expect(first.boxes.length).toBe(8);
});

test('sharpening the unigram lowers the loss', () => {
  const ids = encodeText('The lamp is copper. The lamp is copper. The lamp is bronze.').map((token) => token.id);
  const start = unigramStep(ids, 0.05);
  const later = unigramStep(ids, 1.4);
  expect(later.loss).toBeLessThan(start.loss);
  expect(later.sharpen).toBeGreaterThan(start.sharpen);
});

test('space scene connects a row’s start to its end and to a full-width neighbor', () => {
  const scene = spaceScene(
    {
      source: 'checkpoints/pretrain.pt',
      width: 128,
      layers: 1,
      heads: 1,
      startKind: 'seed',
      seed: 7,
      meanTravel: 1.5,
      meanCosineToStart: 0.2,
      points: [
        {
          id: 1,
          glyph: 'lamp',
          x: 0.2,
          y: 0.1,
          z: 0,
          sx: -0.4,
          sy: 0.2,
          sz: 0.1,
          travel: 2.2,
          cosineToStart: 0.15,
          neighbors: [2],
          neighborCos: [0.66],
          preview: [0.4, -0.2],
          startHead: [0.02, -0.01]
        },
        {
          id: 2,
          glyph: 'light',
          x: 0.5,
          y: -0.2,
          z: 0.1,
          sx: 0.1,
          sy: -0.1,
          sz: 0,
          travel: 0.4,
          cosineToStart: 0.8,
          neighbors: [1],
          neighborCos: [0.66]
        }
      ],
      attention: { text: 'lamp', ids: [1], glyphs: ['lamp'], layers: [] }
    },
    0.1
  );
  expect(scene.readout).toContain('lamp');
  expect(scene.readout).toContain('0.66');
  expect(scene.segments.length).toBe(2);
  expect(scene.points.some((point) => point.label.includes('start'))).toBe(true);
  expect(scene.caption).toContain('128');
  expect(scene.boxes.length).toBe(4);
  const stepped = spaceScene(
    {
      source: 'checkpoints/pretrain.pt',
      width: 128,
      layers: 1,
      heads: 1,
      startKind: 'seed',
      seed: 7,
      points: [
        {
          id: 1,
          glyph: 'lamp',
          x: 0,
          y: 0,
          z: 0,
          neighbors: [],
          preview: [0.1],
          startHead: [0.01]
        },
        {
          id: 2,
          glyph: 'light',
          x: 1,
          y: 0,
          z: 0,
          neighbors: [],
          preview: [0.2],
          startHead: [0.02]
        }
      ],
      attention: { text: 'lamp', ids: [1, 2], glyphs: ['lamp', 'light'], layers: [] }
    },
    0.45
  );
  expect(stepped.readout).toContain('light');
});

test('temperature zero always samples the tallest logit', () => {
  const logits = [0.2, 2.4, 0.4];
  expect(sampleAt(0, logits, 0)).toBe(1);
  expect(sampleAt(5, logits, 0)).toBe(1);
});

test('the sample stage draws the supplied checkpoint logits', () => {
  const logits = new Array(32).fill(-2);
  logits[4] = 5;
  const scene = sampleScene(0, 0.2, { logits, checkpoint: 'checkpoints/pretrain.pt', topK: 0 });
  expect(scene.readout).toContain('100.0%');
  expect(scene.readout).toContain('pretrain.pt');
  expect(scene.boxes[0]?.label.startsWith(glyphFor(4))).toBe(true);
  expect(scene.caption).toContain('live model');
  expect(sampleScene(0.4, 0, null).readout).toBe('Loading the live logits');
});
