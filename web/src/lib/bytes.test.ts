import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { decodeIds, encodeText, installTokenizer } from './bytes.ts';

const table = JSON.parse(readFileSync(path.join(import.meta.dir, '../../../tinyllm/bpe.json'), 'utf8')) as {
  pieces: string[];
  merges: [string, string][];
};

installTokenizer(table.pieces, table.merges);

const fixtures: Record<string, number[]> = {
  'The lamp is copper.\n': [627, 5507, 351, 1001, 7670, 13, 198],
  '  leading': [220, 5486],
  'Question: What is the lighthouse lamp?\nAnswer:': [5488, 418, 334, 25, 1166, 351, 260, 1034, 1714, 5507, 30, 198, 2969, 82, 1368, 25],
  ' copper\n': [1001, 7670, 198],
  "It's a well-known 19th-century café.\n": [945, 6, 82, 258, 792, 12, 2654, 5464, 372, 12, 2292, 2397, 1144, 69, 1003, 13, 198],
  "don't": [4303, 6, 83],
  'a\n\nb': [64, 198, 198, 65],
  '😀': [172, 253, 246, 222]
};

test('the browser encoder matches the Python BPE table', () => {
  for (const [text, ids] of Object.entries(fixtures)) {
    const encoded = encodeText(text);
    expect(encoded.map((token) => token.id)).toEqual(ids);
    expect(decodeIds(encoded.map((token) => token.id))).toBe(text);
  }
});
