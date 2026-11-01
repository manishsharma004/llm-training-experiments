import type { RunStatus } from '#lib/types.ts';

/** Prefer a live run, then the longest book-corpus run, then the newest match. */
export function pickRun(runs: RunStatus[], phase: string): RunStatus | null {
  const matching = runs.filter((item) => item.phase === phase);
  const running = matching.find((item) => item.state === 'running');
  if (running) return running;
  const corpus = matching
    .filter((item) => item.id === 'corpus' || String(item.dataSource ?? '').includes('corpus'))
    .sort((left, right) => (right.step ?? 0) - (left.step ?? 0));
  return corpus[0] ?? matching[0] ?? null;
}
