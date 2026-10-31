export type ByteToken = {
  index: number;
  id: number;
  glyph: string;
};

export function encodeBytes(text: string): ByteToken[] {
  const bytes = new TextEncoder().encode(text);
  return [...bytes].map((id, index) => ({ index, id, glyph: glyphFor(id) }));
}

export function glyphFor(id: number): string {
  if (id === 10) return '↵';
  if (id === 32) return '␣';
  if (id >= 32 && id < 127) return String.fromCharCode(id);
  return `\\x${id.toString(16).padStart(2, '0')}`;
}

export function softmax(logits: number[], temperature: number): number[] {
  const scale = temperature <= 0 ? 1 : temperature;
  const scaled = temperature <= 0 ? logits : logits.map((value) => value / scale);
  const peak = Math.max(...scaled);
  const exps = scaled.map((value) => Math.exp(value - peak));
  const sum = exps.reduce((total, value) => total + value, 0);
  return exps.map((value) => value / sum);
}
