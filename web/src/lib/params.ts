/** Parameter breakdown. Keep this aligned with `tinyllm/params.py`. */

export type ModelShape = {
  vocabSize: number;
  blockSize: number;
  nLayer: number;
  nHead: number;
  nEmbd: number;
  mlpRatio: number;
};

export const defaultShape: ModelShape = {
  vocabSize: 8192,
  blockSize: 128,
  nLayer: 5,
  nHead: 4,
  nEmbd: 128,
  mlpRatio: 4
};

export type ParamReport = {
  tokenEmbedding: number;
  positionEmbedding: number;
  attention: number;
  mlp: number;
  layerNorm: number;
  perBlock: number;
  total: number;
  valid: boolean;
  reason: string;
};

export function breakdown(shape: ModelShape = defaultShape): ParamReport {
  const reason = validate(shape);
  const emb = shape.nEmbd;
  const hidden = shape.nEmbd * shape.mlpRatio;
  const tokenEmbedding = shape.vocabSize * emb;
  const positionEmbedding = shape.blockSize * emb;
  const attention = shape.nLayer * 4 * emb * emb;
  const mlp = shape.nLayer * 2 * emb * hidden;
  const blockNorm = 4 * emb;
  const perBlock = 4 * emb * emb + 2 * emb * hidden + blockNorm;
  const layerNorm = shape.nLayer * blockNorm + 2 * emb;
  const total = tokenEmbedding + positionEmbedding + attention + mlp + layerNorm;
  return {
    tokenEmbedding,
    positionEmbedding,
    attention,
    mlp,
    layerNorm,
    perBlock,
    total,
    valid: reason === '',
    reason
  };
}

export function validate(shape: ModelShape): string {
  if (shape.nEmbd % shape.nHead !== 0) {
    return 'Width must divide evenly across the heads.';
  }
  if (shape.nLayer < 1 || shape.nHead < 1 || shape.nEmbd < 1) {
    return 'Layers, heads, and width must be positive.';
  }
  return '';
}

export function formatCount(value: number): string {
  return new Intl.NumberFormat('en-US').format(value);
}

export const CHANCE_NATS = Math.log(8192);
