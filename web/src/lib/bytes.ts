export type ByteToken = {
  index: number;
  id: number;
  glyph: string;
  piece: string;
};

type TokenizerTable = {
  pieces: string[];
  vocab: Map<string, number>;
  ranks: Map<string, number>;
  byteToUnicode: string[];
  unicodeToByte: Map<string, number>;
};

let table: TokenizerTable | null = null;

function bytesToUnicode(): { byteToUnicode: string[]; unicodeToByte: Map<string, number> } {
  const visible = [
    ...range(0x21, 0x7e),
    ...range(0xa1, 0xac),
    ...range(0xae, 0xff)
  ];
  const mapped = [...visible];
  let extra = 0;
  for (let byte = 0; byte < 256; byte += 1) {
    if (!visible.includes(byte)) {
      visible.push(byte);
      mapped.push(256 + extra);
      extra += 1;
    }
  }
  const byteToUnicode = new Array<string>(256);
  const unicodeToByte = new Map<string, number>();
  visible.forEach((byte, index) => {
    const char = String.fromCharCode(mapped[index]);
    byteToUnicode[byte] = char;
    unicodeToByte.set(char, byte);
  });
  return { byteToUnicode, unicodeToByte };
}

function range(start: number, end: number): number[] {
  const values: number[] = [];
  for (let value = start; value <= end; value += 1) values.push(value);
  return values;
}

export function installTokenizer(pieces: string[], merges: [string, string][]): void {
  const ranks = new Map<string, number>();
  merges.forEach((pair, index) => ranks.set(`${pair[0]}\0${pair[1]}`, index));
  const vocab = new Map<string, number>();
  pieces.forEach((piece, index) => vocab.set(piece, index));
  const maps = bytesToUnicode();
  wordCache.clear();
  table = { pieces, vocab, ranks, ...maps };
}

export function tokenizerInstalled(): boolean {
  return table !== null;
}

export function vocabSize(): number {
  return table?.pieces.length ?? 8192;
}

function isLetter(char: string): boolean {
  return /^\p{L}$/u.test(char);
}

function isNumber(char: string): boolean {
  return /^\p{N}$/u.test(char);
}

function isSpace(char: string): boolean {
  return /^\s$/u.test(char);
}

function isOther(char: string): boolean {
  return !isSpace(char) && !isLetter(char) && !isNumber(char);
}

function takeSpaced(text: string, index: number, kind: (char: string) => boolean): number | null {
  const start = index;
  if (text[index] === ' ' && index + 1 < text.length && !isSpace(text[index + 1])) index += 1;
  if (index >= text.length || !kind(text[index])) return null;
  index += 1;
  while (index < text.length && kind(text[index])) index += 1;
  return index > start ? index : null;
}

function splitWords(text: string): string[] {
  const parts: string[] = [];
  let index = 0;
  while (index < text.length) {
    if (text[index] === "'") {
      const rest = text.slice(index + 1);
      let suffix = '';
      if (rest.startsWith('ll') || rest.startsWith('ve') || rest.startsWith('re')) suffix = rest.slice(0, 2);
      else if ('sdmt'.includes(rest[0] ?? '')) suffix = rest[0];
      if (suffix) {
        parts.push(`'${suffix}`);
        index += 1 + suffix.length;
        continue;
      }
    }
    let matched = false;
    for (const kind of [isLetter, isNumber, isOther]) {
      const end = takeSpaced(text, index, kind);
      if (end !== null) {
        parts.push(text.slice(index, end));
        index = end;
        matched = true;
        break;
      }
    }
    if (matched) continue;
    if (!isSpace(text[index])) {
      parts.push(text[index]);
      index += 1;
      continue;
    }
    let end = index + 1;
    while (end < text.length && isSpace(text[end])) end += 1;
    if (end < text.length && end - index >= 2) end -= 1;
    parts.push(text.slice(index, end));
    index = end;
  }
  return parts;
}

const wordCache = new Map<string, number[]>();

function bpeWord(mapped: string): number[] {
  const cached = wordCache.get(mapped);
  if (cached) return cached;
  if (!table) return [];
  let symbols = [...mapped];
  while (symbols.length > 1) {
    let bestRank = 1e12;
    let best: [string, string] | null = null;
    const seen = new Set<string>();
    for (let index = 0; index < symbols.length - 1; index += 1) {
      const key = `${symbols[index]}\0${symbols[index + 1]}`;
      if (seen.has(key)) continue;
      seen.add(key);
      const rank = table.ranks.get(key) ?? 1e12;
      if (rank < bestRank) {
        bestRank = rank;
        best = [symbols[index], symbols[index + 1]];
      }
    }
    if (!best || bestRank >= 1e12) break;
    const [first, second] = best;
    const merged: string[] = [];
    let index = 0;
    while (index < symbols.length) {
      const found = symbols.indexOf(first, index);
      if (found < 0) {
        merged.push(...symbols.slice(index));
        break;
      }
      merged.push(...symbols.slice(index, found));
      index = found;
      if (index < symbols.length - 1 && symbols[index] === first && symbols[index + 1] === second) {
        merged.push(first + second);
        index += 2;
      } else {
        merged.push(symbols[index]);
        index += 1;
      }
    }
    symbols = merged;
  }
  const ids = symbols.map((symbol) => {
    const id = table!.vocab.get(symbol);
    if (id === undefined) throw new Error('Missing BPE piece');
    return id;
  });
  wordCache.set(mapped, ids);
  return ids;
}

function symbolText(symbol: string): string {
  if (!table) return symbol;
  const bytes: number[] = [];
  for (const char of symbol) {
    const byte = table.unicodeToByte.get(char);
    if (byte !== undefined) bytes.push(byte);
  }
  const text = new TextDecoder().decode(new Uint8Array(bytes));
  if (text === '\n') return '↵';
  if (text === ' ') return '␣';
  return text.replaceAll('\n', '↵').replaceAll(' ', '␣') || '·';
}

export type MergeFrame = { pieces: string[]; note: string };

const films = new Map<string, MergeFrame[]>();

/** Step through the real merges for a short span of text. The last frame is the BPE encoding. */
export function mergeFilm(text: string): MergeFrame[] {
  const key = text.slice(0, 96);
  const cached = films.get(key);
  if (cached) return cached;
  if (!table || !key) return [];
  const words = splitWords(key).slice(0, 8);
  const states = words.map((word) => {
    const mapped = [...new TextEncoder().encode(word)].map((byte) => table!.byteToUnicode[byte]);
    return mapped;
  });
  const snapshot = (note: string): MergeFrame => ({
    pieces: states.flatMap((symbols) => symbols.map((symbol) => symbolText(symbol))),
    note
  });
  const frames = [snapshot('Bytes, before any merge.')];
  for (let step = 0; step < 48; step += 1) {
    let bestRank = 1e12;
    let wordIndex = -1;
    let first = '';
    let second = '';
    states.forEach((symbols, index) => {
      const seen = new Set<string>();
      for (let cursor = 0; cursor < symbols.length - 1; cursor += 1) {
        const rankKey = `${symbols[cursor]}\0${symbols[cursor + 1]}`;
        if (seen.has(rankKey)) continue;
        seen.add(rankKey);
        const rank = table!.ranks.get(rankKey) ?? 1e12;
        if (rank < bestRank) {
          bestRank = rank;
          wordIndex = index;
          first = symbols[cursor];
          second = symbols[cursor + 1];
        }
      }
    });
    if (wordIndex < 0 || bestRank >= 1e12) break;
    const symbols = states[wordIndex];
    const merged: string[] = [];
    let cursor = 0;
    while (cursor < symbols.length) {
      const found = symbols.indexOf(first, cursor);
      if (found < 0) {
        merged.push(...symbols.slice(cursor));
        break;
      }
      merged.push(...symbols.slice(cursor, found));
      cursor = found;
      if (cursor < symbols.length - 1 && symbols[cursor] === first && symbols[cursor + 1] === second) {
        merged.push(first + second);
        cursor += 2;
      } else {
        merged.push(symbols[cursor]);
        cursor += 1;
      }
    }
    states[wordIndex] = merged;
    frames.push(snapshot(`Merge ${step + 1}: “${symbolText(first)}” + “${symbolText(second)}”.`));
  }
  films.set(key, frames);
  return frames;
}

export function decodeIds(ids: number[]): string {
  if (!table) return '';
  const bytes: number[] = [];
  for (const id of ids) {
    const piece = table.pieces[id];
    if (!piece) continue;
    for (const char of piece) {
      const byte = table.unicodeToByte.get(char);
      if (byte !== undefined) bytes.push(byte);
    }
  }
  return new TextDecoder().decode(new Uint8Array(bytes));
}

export function pieceText(id: number): string {
  if (!table || id < 0 || id >= table.pieces.length) return '';
  const bytes: number[] = [];
  for (const char of table.pieces[id]) {
    const byte = table.unicodeToByte.get(char);
    if (byte !== undefined) bytes.push(byte);
  }
  return new TextDecoder().decode(new Uint8Array(bytes));
}

export function glyphForToken(id: number): string {
  const text = pieceText(id);
  if (!text) return '·';
  if (text === '\n') return '↵';
  if (text === ' ') return '␣';
  return text.replaceAll('\n', '↵').replaceAll(' ', '␣');
}

export function glyphFor(id: number): string {
  if (table) return glyphForToken(id);
  if (id === 10) return '↵';
  if (id === 32) return '␣';
  if (id >= 32 && id < 127) return String.fromCharCode(id);
  return `\\x${id.toString(16).padStart(2, '0')}`;
}

export function encodeText(text: string): ByteToken[] {
  if (!table) return [];
  const ids: number[] = [];
  for (const part of splitWords(text)) {
    const mapped = [...new TextEncoder().encode(part)].map((byte) => table!.byteToUnicode[byte]).join('');
    ids.push(...bpeWord(mapped));
  }
  return ids.map((id, index) => ({ index, id, glyph: glyphForToken(id), piece: pieceText(id) }));
}

/** @deprecated Byte ids. The lab encodes with `encodeText` once the BPE table is loaded. */
export function encodeBytes(text: string): ByteToken[] {
  if (table) return encodeText(text);
  const bytes = new TextEncoder().encode(text);
  return [...bytes].map((id, index) => ({ index, id, glyph: glyphFor(id), piece: glyphFor(id) }));
}

export function softmax(logits: number[], temperature: number): number[] {
  const scale = temperature <= 0 ? 1 : temperature;
  const scaled = temperature <= 0 ? logits : logits.map((value) => value / scale);
  const peak = Math.max(...scaled);
  const exps = scaled.map((value) => Math.exp(value - peak));
  const sum = exps.reduce((total, value) => total + value, 0);
  return exps.map((value) => value / sum);
}
