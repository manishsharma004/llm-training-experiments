import { glyphFor, softmax } from './bytes.ts';

export type RankedPiece = {
  id: number;
  glyph: string;
  logit: number;
  prob: number;
};

export type RankedNext = {
  pieces: RankedPiece[];
  rest: number;
};

/** Highest probabilities after temperature and top-k, plus the mass of everything not listed. */
export function rankNext(logits: number[], temperature: number, topK: number, limit = 8): RankedNext {
  if (!logits.length) return { pieces: [], rest: 0 };
  if (temperature <= 0) {
    let best = 0;
    for (let index = 1; index < logits.length; index += 1) {
      if (logits[index] > logits[best]) best = index;
    }
    return { pieces: [{ id: best, glyph: glyphFor(best), logit: logits[best], prob: 1 }], rest: 0 };
  }

  const scored = logits.map((logit, id) => ({ id, logit }));
  if (topK > 0 && topK < logits.length) {
    const keep = new Set(
      scored
        .slice()
        .sort((left, right) => right.logit - left.logit)
        .slice(0, topK)
        .map((row) => row.id)
    );
    for (const row of scored) {
      if (!keep.has(row.id)) row.logit = Number.NEGATIVE_INFINITY;
    }
  }
  const probs = softmax(
    scored.map((row) => row.logit),
    temperature
  );
  const ranked = scored
    .map((row, index) => ({ id: row.id, glyph: glyphFor(row.id), logit: logits[row.id], prob: probs[index] }))
    .filter((row) => row.prob > 0 && Number.isFinite(row.prob))
    .sort((left, right) => right.prob - left.prob);
  const pieces = ranked.slice(0, Math.max(1, limit));
  const shown = pieces.reduce((sum, row) => sum + row.prob, 0);
  return { pieces, rest: Math.max(0, 1 - shown) };
}
