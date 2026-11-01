import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { encodeText, glyphFor, installTokenizer, mergeFilm } from './bytes.ts';
import { pipelineScene, sampleAt, sampleScene, unigramStep, windowScene } from './simulate.ts';

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
