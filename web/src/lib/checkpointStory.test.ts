import { expect, test } from 'bun:test';
import { checkpointStory, tokensAt } from './checkpointStory.ts';

test('a long byte pretrain is the book pass', () => {
  const row = { path: 'checkpoints/corpus.pt', phase: 'pretrain', step: 11531, loss: 1.1957, vocabSize: 256, blockSize: 128, batchSize: 16 };
  expect(tokensAt(row)).toBe(11531 * 128 * 16);
  expect(checkpointStory(row)).toContain('book bytes');
});

test('a short byte pretrain is letter soup', () => {
  const text = checkpointStory({ path: 'checkpoints/pretrain.pt', phase: 'pretrain', step: 40, loss: 2.23, vocabSize: 256, blockSize: 128, batchSize: 16 });
  expect(text).toContain('letter soup');
  expect(text).toContain('5.55');
});

test('finetune names the answer-only loss', () => {
  const text = checkpointStory({ path: 'checkpoints/sft.pt', phase: 'finetune', step: 80, loss: 0.002, vocabSize: 256, blockSize: 128, batchSize: 16 });
  expect(text).toContain('answer tokens');
  expect(text).toContain('memorized');
});

test('a short book-vocabulary pretrain names its share of the stream', () => {
  const text = checkpointStory(
    { path: 'checkpoints/pretrain.pt', phase: 'pretrain', step: 100, loss: 6.39, vocabSize: 8192, blockSize: 128, batchSize: 16 },
    6_398_495
  );
  expect(text).toContain('3.2%');
  expect(text).toContain('9.01');
});
