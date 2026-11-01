import { installTokenizer } from './bytes.ts';

export const tokenizerState = $state({
  ready: false,
  vocabSize: 8192,
  merges: 7936
});

let loading: Promise<void> | null = null;

export function ensureTokenizer(): Promise<void> {
  if (tokenizerState.ready) return Promise.resolve();
  loading ??= fetch('/api/tokenizer')
    .then((response) => response.json())
    .then((body: { pieces: string[]; merges: [string, string][]; vocab_size?: number }) => {
      installTokenizer(body.pieces, body.merges);
      tokenizerState.vocabSize = body.vocab_size ?? body.pieces.length;
      tokenizerState.merges = body.merges.length;
      tokenizerState.ready = true;
    });
  return loading;
}
