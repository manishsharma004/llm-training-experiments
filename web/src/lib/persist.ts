import { defaultShape, validate, type ModelShape } from './params.ts';

export type LabSnapshot = {
  shape: ModelShape;
  prompt: string;
  completion: string;
  temperature: number;
};

export type BenchSnapshot = {
  stepCount: number;
  ftCount: number;
  rate: number;
  batchSize: number;
  seed: number;
  deviceChoice: 'auto' | 'cpu' | 'gpu';
  gfxOverride: boolean;
};

export type SampleSnapshot = {
  prompt: string;
  maxNewTokens: number;
  topK: number;
  checkpoint: string;
};

export type TokenizeSnapshot = {
  text: string;
  block: number;
  cursor: number;
};

export type StageSnapshot = {
  time: number;
  speed: number;
  mode: string;
  filter: 'all' | 'letter' | 'digit' | 'space' | 'other';
  layer: number;
  head: number;
  query: number;
  exploded: boolean;
};

const MODES: Record<string, string[]> = {
  '/architecture': ['space', 'forward', 'attention', 'cloud', 'stack'],
  '/train': ['step', 'loss', 'machine'],
  '/finetune': ['mask', 'loss']
};

const FILTERS = ['all', 'letter', 'digit', 'space', 'other'] as const;

function whole(value: unknown, fallback: number, min: number, max: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return fallback;
  const rounded = Math.round(value);
  if (rounded < min || rounded > max) return fallback;
  return rounded;
}

function decimal(value: unknown, fallback: number, min: number, max: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return fallback;
  if (value < min || value > max) return fallback;
  return value;
}

export function sanitizeLab(raw: unknown): LabSnapshot | null {
  if (!raw || typeof raw !== 'object') return null;
  const row = raw as Record<string, unknown>;
  const shapeIn = row.shape && typeof row.shape === 'object' ? (row.shape as Record<string, unknown>) : {};
  const shape: ModelShape = {
    vocabSize: 8192,
    blockSize: whole(shapeIn.blockSize, defaultShape.blockSize, 8, 512),
    nLayer: whole(shapeIn.nLayer, defaultShape.nLayer, 1, 12),
    nHead: whole(shapeIn.nHead, defaultShape.nHead, 1, 8),
    nEmbd: whole(shapeIn.nEmbd, defaultShape.nEmbd, 8, 512),
    mlpRatio: whole(shapeIn.mlpRatio, defaultShape.mlpRatio, 1, 8)
  };
  if (validate(shape)) {
    shape.nEmbd = defaultShape.nEmbd;
    shape.nHead = defaultShape.nHead;
  }
  if (typeof row.prompt !== 'string' || typeof row.completion !== 'string') return null;
  const temperature = decimal(row.temperature, -1, 0, 2);
  if (temperature < 0) return null;
  return {
    shape,
    prompt: row.prompt.slice(0, 2000),
    completion: row.completion.slice(0, 2000),
    temperature
  };
}

export function sanitizeBench(raw: unknown): BenchSnapshot | null {
  if (!raw || typeof raw !== 'object') return null;
  const row = raw as Record<string, unknown>;
  const deviceChoice = row.deviceChoice === 'cpu' || row.deviceChoice === 'gpu' || row.deviceChoice === 'auto' ? row.deviceChoice : 'auto';
  return {
    stepCount: whole(row.stepCount, 20, 1, 20_000),
    ftCount: whole(row.ftCount, 8, 1, 20_000),
    rate: decimal(row.rate, 0.003, 1e-5, 0.1),
    batchSize: whole(row.batchSize, 16, 1, 64),
    seed: whole(row.seed, 7, 0, 1_000_000),
    deviceChoice,
    gfxOverride: row.gfxOverride === true
  };
}

export function sanitizeSample(raw: unknown): SampleSnapshot | null {
  if (!raw || typeof raw !== 'object') return null;
  const row = raw as Record<string, unknown>;
  if (typeof row.prompt !== 'string') return null;
  const checkpoint = typeof row.checkpoint === 'string' && row.checkpoint.endsWith('.pt') && !row.checkpoint.includes('..') ? row.checkpoint.slice(0, 180) : '';
  return {
    prompt: row.prompt.slice(0, 500),
    maxNewTokens: whole(row.maxNewTokens, 24, 1, 80),
    topK: whole(row.topK, 0, 0, 64),
    checkpoint
  };
}

export function sanitizeTokenize(raw: unknown): TokenizeSnapshot | null {
  if (!raw || typeof raw !== 'object') return null;
  const row = raw as Record<string, unknown>;
  if (typeof row.text !== 'string') return null;
  return {
    text: row.text.slice(0, 4000),
    block: whole(row.block, 16, 4, 48),
    cursor: whole(row.cursor, 0, 0, 100_000)
  };
}

export function defaultStage(pathname: string): StageSnapshot {
  const modes = MODES[pathname] ?? [];
  return {
    time: 0,
    speed: 1,
    mode: modes[0] ?? 'forward',
    filter: 'all',
    layer: 0,
    head: 0,
    query: 0,
    exploded: false
  };
}

export function sanitizeStage(pathname: string, raw: unknown): StageSnapshot {
  const base = defaultStage(pathname);
  if (!raw || typeof raw !== 'object') return base;
  const row = raw as Record<string, unknown>;
  const modes = MODES[pathname] ?? [];
  const mode = typeof row.mode === 'string' && (modes.length === 0 || modes.includes(row.mode)) ? row.mode : base.mode;
  const filter = FILTERS.includes(row.filter as (typeof FILTERS)[number]) ? (row.filter as (typeof FILTERS)[number]) : 'all';
  const speed = row.speed === 0.5 || row.speed === 1 || row.speed === 2 ? row.speed : 1;
  const time = typeof row.time === 'number' && Number.isFinite(row.time) && row.time >= 0 ? Math.min(row.time, 86_400) : 0;
  return {
    time,
    speed,
    mode,
    filter,
    layer: whole(row.layer, 0, 0, 32),
    head: whole(row.head, 0, 0, 32),
    query: whole(row.query, 0, 0, 256),
    exploded: row.exploded === true
  };
}

export function stageKey(pathname: string): string {
  return `stage:${pathname}`;
}

const DB_NAME = 'book-lab';
const STORE = 'kv';

let opening: Promise<IDBDatabase> | null = null;
let installed = false;
const queued = new Map<string, { timer: ReturnType<typeof setTimeout>; value: unknown }>();

function canStore(): boolean {
  return typeof indexedDB !== 'undefined';
}

function openDb(): Promise<IDBDatabase> {
  if (!opening) {
    opening = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, 1);
      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE);
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => {
        opening = null;
        reject(request.error);
      };
    });
  }
  return opening;
}

function installFlush(): void {
  if (installed || typeof window === 'undefined') return;
  installed = true;
  window.addEventListener('pagehide', () => {
    for (const [key, entry] of queued) {
      clearTimeout(entry.timer);
      void writeStore(key, entry.value);
    }
    queued.clear();
  });
}

export async function readStore(key: string): Promise<unknown> {
  if (!canStore()) return null;
  try {
    const db = await openDb();
    return await new Promise((resolve, reject) => {
      const request = db.transaction(STORE, 'readonly').objectStore(STORE).get(key);
      request.onsuccess = () => resolve(request.result ?? null);
      request.onerror = () => reject(request.error);
    });
  } catch {
    return null;
  }
}

export async function writeStore(key: string, value: unknown): Promise<void> {
  if (!canStore()) return;
  try {
    const db = await openDb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, 'readwrite');
      tx.objectStore(STORE).put(value, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch {
    // The page still works when the browser rejects the write.
  }
}

export function writeStoreSoon(key: string, value: unknown, wait = 200): void {
  if (!canStore()) return;
  installFlush();
  const snapshot = JSON.parse(JSON.stringify(value)) as unknown;
  const existing = queued.get(key);
  if (existing) clearTimeout(existing.timer);
  const timer = setTimeout(() => {
    queued.delete(key);
    void writeStore(key, snapshot);
  }, wait);
  queued.set(key, { timer, value: snapshot });
}
