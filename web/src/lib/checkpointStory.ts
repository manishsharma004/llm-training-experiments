export type CheckpointFact = {
  path: string;
  phase?: string;
  step?: number;
  loss?: number;
  vocabSize?: number;
  blockSize?: number;
  batchSize?: number;
};

export function tokensAt(row: CheckpointFact): number | null {
  if (row.step == null || row.blockSize == null || row.batchSize == null) return null;
  return row.step * row.blockSize * row.batchSize;
}

export function checkpointStory(row: CheckpointFact, trainTokens?: number | null): string {
  const loss = row.loss == null ? '—' : row.loss.toFixed(2);
  const steps = (row.step ?? 0).toLocaleString();
  const seen = tokensAt(row);
  const seenText = seen == null ? '—' : seen.toLocaleString();
  const vocab = row.vocabSize ?? 0;
  const share = trainTokens && seen ? seen / trainTokens : null;
  const shareText = share == null ? '' : share >= 1 ? ' That is at least one pass over the training stream.' : ` That is about ${(share * 100).toFixed(1)}% of one pass over the training stream.`;

  if (row.phase === 'finetune') {
    const settled = row.loss != null && row.loss < 0.1;
    const outcome = settled
      ? 'The loss is near zero, so those answers were memorized. A prose prompt is outside that format, and the sample spills pieces of them.'
      : 'The loss is still high, so those answers have not stuck. A prose prompt has even less to stand on.';
    return `Finetune for ${steps} steps, loss ${loss}. The question stayed in context and the loss trained only the answer tokens of the harbor notes. ${outcome}`;
  }
  if (vocab > 0 && vocab <= 256 && (seen ?? 0) >= 1_000_000) {
    return `Pretrain on the book bytes: ${steps} steps, about ${seenText} tokens, loss ${loss}. That covers the text once. The vocabulary is 256, one id per byte, so this file stays separate from the 8,192-piece model.`;
  }
  if (vocab > 0 && vocab <= 256) {
    return `Pretrain on byte ids, stopped after ${steps} steps, about ${seenText} tokens, loss ${loss}. A uniform guess over 256 bytes is 5.55 nats, so this file knows local letter patterns and the sample is letter soup.`;
  }
  if (vocab >= 8192 && row.phase === 'pretrain') {
    return `Pretrain with the 8,192-piece vocabulary: ${steps} steps, about ${seenText} tokens, loss ${loss}. Chance over 8,192 pieces is 9.01 nats. The windows were drawn from the whole book stream.${shareText} A loss still far above 1 nat leaves the continuation unsteady.`;
  }
  return `${row.phase ?? 'Checkpoint'}, ${steps} steps, loss ${loss}.`;
}
