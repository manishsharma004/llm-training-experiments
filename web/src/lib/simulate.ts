import { encodeText, glyphForToken, mergeFilm, softmax, tokenizerInstalled } from './bytes.ts';
import { rankNext } from './distribution.ts';

export type SimPoint = {
  id: string;
  x: number;
  y: number;
  z: number;
  color: string;
  label: string;
  detail: string;
  size?: number;
};

export type SimBox = {
  id: string;
  x: number;
  y: number;
  z: number;
  dx: number;
  dy: number;
  dz: number;
  color: string;
  label: string;
  detail: string;
};

export type SimSegment = {
  x1: number;
  y1: number;
  z1: number;
  x2: number;
  y2: number;
  z2: number;
  color: string;
};

export type Scene = {
  points: SimPoint[];
  boxes: SimBox[];
  segments: SimSegment[];
  caption: string;
  duration: number;
  readout: string;
};

type FileRow = { source: string; bytes: number };
type CleanRow = { source: string; reasons: string[]; excerpt: string };
type KeptRow = { source: string; chars: number; after: string };
type MetricRow = { step: number; loss: number; lr: number; phase: string };
type GeoPoint = {
  id: number;
  glyph: string;
  x: number;
  y: number;
  z: number;
  sx?: number;
  sy?: number;
  sz?: number;
  travel?: number;
  cosineToStart?: number;
  neighbors: number[];
  neighborCos?: number[];
  preview?: number[];
  startHead?: number[];
};
type Geo = {
  source: string;
  width: number;
  layers: number;
  heads: number;
  startKind?: string;
  seed?: number | null;
  meanTravel?: number;
  meanCosineToStart?: number;
  points: GeoPoint[];
  attention: { text: string; ids: number[]; glyphs: string[]; layers: number[][][][] };
};

export type SimInput = {
  path: string;
  time: number;
  playing: boolean;
  mode: string;
  filter: string;
  layer: number;
  head: number;
  query: number;
  exploded: boolean;
  files: FileRow[];
  clean: {
    kept?: number;
    dropped?: number;
    reasons?: Record<string, number>;
    dropped_examples?: CleanRow[];
    kept_preview?: KeptRow[];
  } | null;
  trainTokens: number | null;
  valTokens: number | null;
  preview: string;
  excerptIds: number[];
  metrics: MetricRow[];
  prompt: string;
  completion: string;
  temperature: number;
  shape: { nLayer: number; nHead: number; nEmbd: number; blockSize: number; mlpRatio: number };
  geometry: Geo | null;
  powerLast: number | null;
  powerPeak: number | null;
  vramLast: number | null;
  vramPeak: number | null;
  vramTotal: number | null;
  junctionLast: number | null;
  logits: number[];
  topK: number;
  checkpoint: string;
};

const PINE = '#1e4642';
const COPPER = '#c4622d';
const SAND = '#d89a55';
const INK = '#8a8175';

function wrap(time: number, duration: number): number {
  if (duration <= 0) return 0;
  const value = time % duration;
  return value < 0 ? value + duration : value;
}

function fileName(source: string): string {
  return source.split('/').pop() ?? source;
}

export function pipelineScene(time: number): Scene {
  const stations = ['Gather', 'Clean', 'Prepare', 'Tokens', 'Shape', 'Train', 'Sample'];
  const notes = [
    'Files are copied in and counted. Nothing is removed.',
    'HTML, duplicates, and junk fall out. The rest continue.',
    'Documents split, then the BPE writes token ids.',
    'Merges collapse bytes into the pieces the model sees.',
    'Those ids pick rows of the embedding.',
    'A window predicts the next id and the loss updates the weights.',
    'Temperature turns the last logits into one sampled piece.'
  ];
  const duration = 9;
  const travel = wrap(time, duration) / duration * (stations.length - 1);
  const index = Math.min(stations.length - 1, Math.floor(travel));
  const frac = travel - index;
  const boxes: SimBox[] = stations.map((label, station) => ({
    id: label,
    x: (station - 3) * 0.92,
    y: 0.28,
    z: 0,
    dx: 0.3,
    dy: station === index ? 0.4 : 0.24,
    dz: 0.3,
    color: station === index ? COPPER : PINE,
    label,
    detail: notes[station]
  }));
  const x = (index - 3) * 0.92 + frac * 0.92;
  const points: SimPoint[] = [
    {
      id: 'packet',
      x,
      y: 0.95 + Math.sin(time * 8) * 0.05,
      z: 0.05,
      color: SAND,
      size: 11,
      label: 'Document',
      detail: `Moving through ${stations[index]}. ${notes[index]}`
    }
  ];
  return {
    points,
    boxes,
    segments: [{ x1: -3, y1: 0.02, z1: 0, x2: 3, y2: 0.02, z2: 0, color: INK }],
    caption: 'A document moving through the lab. Play runs it. Step and the scrubber rewind the same path.',
    duration,
    readout: `${stations[index]} · ${notes[index]}`
  };
}

export function gatherScene(files: FileRow[], time: number): Scene {
  const shown = files.slice(0, 18);
  const duration = Math.max(4, shown.length * 0.55);
  const local = wrap(time, duration);
  const arrived = shown.length === 0 ? 0 : Math.min(shown.length, 1 + Math.floor((local / duration) * shown.length));
  const boxes: SimBox[] = [];
  shown.forEach((file, index) => {
    if (index >= arrived) return;
    const grow = index === arrived - 1 ? Math.min(1, (local - (index * duration) / shown.length) / 0.35) : 1;
    const height = Math.max(0.18, Math.min(1.5, Math.log10(file.bytes + 10) / 3)) * Math.max(0.15, grow);
    boxes.push({
      id: file.source,
      x: (index % 6) * 0.72 - 1.6,
      y: height / 2,
      z: Math.floor(index / 6) * 0.72 - 0.4,
      dx: 0.22,
      dy: height / 2,
      dz: 0.22,
      color: index === arrived - 1 ? COPPER : SAND,
      label: fileName(file.source),
      detail: `${file.bytes.toLocaleString()} bytes. The reader is on this file. Earlier files are already in the raw set.`
    });
  });
  const current = shown[arrived - 1];
  return {
    points: [],
    boxes,
    segments: [],
    caption: 'Gather, as a pass over the book files. Each box lands when the reader reaches it. Height is the log of the file size.',
    duration,
    readout: current
      ? `Reading ${arrived} / ${shown.length} · ${fileName(current.source)} · ${current.bytes.toLocaleString()} bytes`
      : 'No book files in this report yet.'
  };
}

export function cleanScene(clean: SimInput['clean'], time: number): Scene {
  const dropped = clean?.dropped_examples ?? [];
  const kept = clean?.kept_preview ?? [];
  const items = [
    ...dropped.slice(0, 6).map((row) => ({ name: fileName(row.source), fate: row.reasons[0] ?? 'dropped', kept: false })),
    ...kept.slice(0, 4).map((row) => ({ name: fileName(row.source), fate: 'kept', kept: true }))
  ];
  if (items.length === 0) {
    items.push(
      { name: 'kept document', fate: 'kept', kept: true },
      { name: 'duplicate', fate: 'duplicate', kept: false },
      { name: 'too short', fate: 'too_short', kept: false }
    );
  }
  const duration = items.length * 1.35;
  const local = wrap(time, duration);
  const focus = Math.min(items.length - 1, Math.floor(local / 1.35));
  const frac = (local - focus * 1.35) / 1.35;
  const boxes: SimBox[] = [];
  const reasons = ['duplicate', 'too_short', 'low_letter_ratio', 'too_long'];
  reasons.forEach((reason, index) => {
    const count = clean?.reasons?.[reason] ?? 0;
    boxes.push({
      id: `bin-${reason}`,
      x: (index - 1.5) * 0.9,
      y: 0.12,
      z: 1.35,
      dx: 0.32,
      dy: 0.1,
      dz: 0.28,
      color: '#8d2b22',
      label: reason,
      detail: `${count.toLocaleString()} documents in the book corpus were dropped for ${reason}.`
    });
  });
  boxes.push({
    id: 'kept-bin',
    x: 0,
    y: 0.12,
    z: -1.45,
    dx: 0.7,
    dy: 0.1,
    dz: 0.28,
    color: PINE,
    label: 'Kept',
    detail: `${(clean?.kept ?? 0).toLocaleString()} documents stayed. ${(clean?.dropped ?? 0).toLocaleString()} were dropped.`
  });
  items.forEach((item, index) => {
    let z = -0.2;
    let y = 0.45;
    let x = (index - (items.length - 1) / 2) * 0.15;
    if (index < focus) {
      z = item.kept ? -1.45 : 1.35;
      y = 0.35;
      x = item.kept ? (index - 2) * 0.28 : (reasons.indexOf(item.fate) - 1.5) * 0.9;
    } else if (index === focus) {
      z = -1.2 + frac * 2.4;
      if (!item.kept && frac > 0.55) y = 0.45 - (frac - 0.55) * 0.7;
      x = item.kept || frac < 0.55 ? 0 : (Math.max(0, reasons.indexOf(item.fate)) - 1.5) * 0.9 * (frac - 0.55) * 2;
    } else {
      z = -1.6 - (index - focus) * 0.35;
      y = 0.45;
    }
    boxes.push({
      id: `doc-${index}`,
      x,
      y,
      z,
      dx: 0.16,
      dy: 0.14,
      dz: 0.16,
      color: index === focus ? COPPER : item.kept ? PINE : '#8d2b22',
      label: item.name,
      detail: item.kept ? `${item.name} passes the gate and stays in the corpus.` : `${item.name} is dropped: ${item.fate}.`
    });
  });
  const current = items[focus];
  return {
    points: [],
    boxes,
    segments: [],
    caption: 'The clean gate. A document approaches, then either falls into a drop bin or lands in the kept pile. The bins use the book corpus counts.',
    duration,
    readout: current.kept ? `Keeping ${current.name}` : `Dropping ${current.name} · ${current.fate}`
  };
}

export function windowScene(ids: number[], time: number, trainTokens: number | null, valTokens: number | null): Scene {
  const block = 8;
  const span = Math.min(ids.length, 48);
  const usable = ids.slice(0, span);
  if (usable.length < 2) {
    return {
      points: [],
      boxes: [],
      segments: [],
      caption: 'The token stream has not loaded yet.',
      duration: 4,
      readout: 'Waiting for train.bin'
    };
  }
  const hops = Math.max(1, usable.length - block);
  const duration = hops * 0.45;
  const start = Math.min(hops - 1, Math.floor(wrap(time, duration) / 0.45));
  const boxes: SimBox[] = [];
  const points: SimPoint[] = [];
  for (let index = 0; index < block; index += 1) {
    const id = usable[start + index];
    const glyph = glyphForToken(id);
    const predicts = usable[start + index + 1];
    boxes.push({
      id: `w${index}`,
      x: (index - (block - 1) / 2) * 0.78,
      y: 0.4,
      z: 0,
      dx: 0.22,
      dy: 0.36,
      dz: 0.22,
      color: PINE,
      label: glyph,
      detail: `Context id ${id} (${glyph}) at stream index ${start + index}. It is trained to predict id ${predicts}.`
    });
  }
  const target = usable[start + block];
  if (target !== undefined) {
    points.push({
      id: 'target',
      x: (block - (block - 1) / 2) * 0.78,
      y: 1.05 + Math.sin(time * 6) * 0.06,
      z: 0.4,
      color: COPPER,
      size: 12,
      label: glyphForToken(target),
      detail: `Label id ${target}. The window does not contain it. The loss compares the model's scores with this id.`
    });
  }
  return {
    points,
    boxes,
    segments: [],
    caption: 'One training window sliding along the real book stream. Copper is the next token, the label. Play moves the cut.',
    duration,
    readout: `Window at token ${start.toLocaleString()} of ${span}. Stream ${trainTokens?.toLocaleString() ?? '—'} train / ${valTokens?.toLocaleString() ?? '—'} validation. Next id ${target ?? '—'}.`
  };
}

export function bpeScene(preview: string, time: number): Scene {
  const text = (preview || 'The Old Testament of the King James Version').slice(0, 96);
  const film = tokenizerInstalled() ? mergeFilm(text) : [];
  if (film.length === 0) {
    return {
      points: [],
      boxes: [],
      segments: [],
      caption: 'Loading the merge table.',
      duration: 4,
      readout: 'Waiting for the vocabulary'
    };
  }
  const duration = film.length * 0.7;
  const frame = Math.min(film.length - 1, Math.floor(wrap(time, duration) / 0.7));
  const pieces = film[frame].pieces.slice(0, 12);
  const boxes: SimBox[] = pieces.map((piece, index) => ({
    id: `p${index}`,
    x: (index - (pieces.length - 1) / 2) * 0.58,
    y: 0.42,
    z: 0,
    dx: Math.min(0.28, 0.16 + piece.length * 0.03),
    dy: 0.38,
    dz: 0.22,
    color: frame === 0 ? INK : COPPER,
    label: piece,
    detail: `${film[frame].note} Piece “${piece}” is on the tape at this step.`
  }));
  return {
    points: [],
    boxes,
    segments: [],
    caption: 'The byte-level merges, in the order the vocabulary learned them, applied to the start of the book stream. Rewind to see the bytes again.',
    duration,
    readout: `Step ${frame + 1} / ${film.length} · ${film[frame].note}`
  };
}

function attentionFrame(geometry: Geo, layer: number, head: number, query: number, time: number): Scene {
  const layers = geometry.attention.layers;
  const safeLayer = Math.min(layer, Math.max(0, layers.length - 1));
  const heads = layers[safeLayer] ?? [];
  const grid = heads[Math.min(head, Math.max(0, heads.length - 1))] ?? [];
  const glyphs = geometry.attention.glyphs;
  const q = Math.min(query, Math.max(0, glyphs.length - 1));
  const row = grid[q] ?? [];
  const boxes: SimBox[] = glyphs.map((glyph, index) => ({
    id: `tok${index}`,
    x: (index - (glyphs.length - 1) / 2) * 0.85,
    y: index === q ? 0.22 : 0.4,
    z: index === q ? 0.55 : -0.35,
    dx: 0.28,
    dy: index === q ? 0.22 : 0.36,
    dz: 0.24,
    color: index === q ? COPPER : PINE,
    label: glyph,
    detail: index === q ? `Query “${glyph}”. The bars are how strongly it reads each earlier key.` : `Key “${glyph}”.`
  }));
  const points: SimPoint[] = [];
  const segments: SimSegment[] = [];
  row.forEach((weight, key) => {
    if (key > q) return;
    const x = (key - (glyphs.length - 1) / 2) * 0.85;
    const height = 0.35 + weight * 2.2;
    boxes.push({
      id: `bar${key}`,
      x,
      y: 1.15 + height / 2,
      z: 1.15,
      dx: 0.2,
      dy: height / 2,
      dz: 0.2,
      color: key === q ? COPPER : SAND,
      label: weight.toFixed(2),
      detail: `Weight ${weight.toFixed(3)} at layer ${safeLayer + 1}, head ${Math.min(head, heads.length - 1) + 1}. Query “${glyphs[q]}” reading “${glyphs[key]}”. Keys after the query stay at zero.`
    });
  });
  return {
    points,
    boxes,
    segments,
    caption: `Causal attention from the real forward of “${geometry.attention.text}”. Play walks the query, then the layer.`,
    duration: Math.max(4, glyphs.length * layers.length * 0.55),
    readout: `Layer ${safeLayer + 1} / ${layers.length} · head ${Math.min(head, heads.length - 1) + 1} · query “${glyphs[q] ?? '—'}” (${q + 1}/${glyphs.length})`
  };
}

export function forwardScene(geometry: Geo | null, time: number, playing: boolean, layer: number, head: number, query: number): Scene {
  if (!geometry || geometry.attention.glyphs.length === 0) {
    return {
      points: [],
      boxes: [],
      segments: [],
      caption: 'Waiting for the attention pass from the model.',
      duration: 4,
      readout: 'Loading the forward'
    };
  }
  const glyphs = geometry.attention.glyphs;
  const layers = Math.max(1, geometry.attention.layers.length);
  const duration = glyphs.length * layers * 0.55;
  const local = wrap(time, duration);
  const beat = Math.floor(local / 0.55);
  const q = playing ? beat % glyphs.length : Math.min(query, glyphs.length - 1);
  const layerIndex = playing ? Math.floor(beat / glyphs.length) % layers : layer;
  return attentionFrame(geometry, layerIndex, head, q, time);
}

export function lookupScene(geometry: Geo | null, ids: number[], time: number, filter: string): Scene {
  if (!geometry) {
    return { points: [], boxes: [], segments: [], caption: 'Waiting for the embedding table.', duration: 4, readout: 'Loading embeddings' };
  }
  const stream = ids.length > 0 ? ids : geometry.attention.ids;
  const duration = Math.max(4, stream.length * 0.6);
  const cursor = stream[Math.floor(wrap(time, duration) / 0.6) % stream.length] ?? 0;
  const byId = new Map(geometry.points.map((point) => [point.id, point]));
  const active = byId.get(cursor);
  const keep = new Set<number>([cursor, ...(active?.neighbors ?? [])]);
  const stride = Math.max(1, Math.floor(geometry.points.length / 280));
  const points: SimPoint[] = [];
  const segments: SimSegment[] = [];
  geometry.points.forEach((point, index) => {
    const neighbor = keep.has(point.id);
    if (!neighbor && index % stride !== 0) return;
    const glyph = point.glyph;
    const bare = glyph.replaceAll('␣', '').replaceAll('↵', '').trim();
    const kind = !bare ? 'space' : /^\p{N}+$/u.test(bare) ? 'digit' : /^\p{L}+$/u.test(bare) ? 'letter' : 'other';
    if (!neighbor && filter !== 'all' && kind !== filter) return;
    points.push({
      id: `e${point.id}`,
      x: point.x * 2.2,
      y: point.y * 2.2 + (point.id === cursor ? Math.sin(time * 5) * 0.08 : 0),
      z: point.z * 2.2,
      color: point.id === cursor ? COPPER : neighbor ? SAND : INK,
      size: point.id === cursor ? 12 : neighbor ? 8 : 3.5,
      label: `${glyph} · ${point.id}`,
      detail: point.id === cursor
        ? `Lookup of stream id ${cursor}. The row is copied out of the ${geometry.width}-d table from ${geometry.source}.`
        : `Neighbor of the looked-up row, in the full width, drawn in the same 3D projection.`
    });
  });
  if (active) {
    for (const neighbor of active.neighbors) {
      const other = byId.get(neighbor);
      if (!other) continue;
      segments.push({
        x1: active.x * 2.2,
        y1: active.y * 2.2,
        z1: active.z * 2.2,
        x2: other.x * 2.2,
        y2: other.y * 2.2,
        z2: other.z * 2.2,
        color: COPPER
      });
    }
  }
  return {
    points,
    boxes: [],
    segments,
    caption: 'Embedding lookup along the book stream. Copper is the id being read. Lines are its nearest rows in the full width, not in this picture.',
    duration,
    readout: `Reading id ${cursor} ${active ? `(${active.glyph})` : ''} from the ${geometry.points.length.toLocaleString()}-row table`
  };
}

function spaceTour(geometry: Geo): number[] {
  const byTravel = geometry.points
    .slice()
    .sort((left, right) => (right.travel ?? 0) - (left.travel ?? 0))
    .slice(0, 8)
    .map((point) => point.id);
  const seen = new Set<number>();
  const tour: number[] = [];
  for (const id of [...geometry.attention.ids, ...byTravel]) {
    if (seen.has(id) || !geometry.points.some((point) => point.id === id)) continue;
    seen.add(id);
    tour.push(id);
    if (tour.length >= 16) break;
  }
  return tour.length ? tour : [geometry.points[0]?.id ?? 0];
}

export function spaceScene(geometry: Geo | null, time: number): Scene {
  if (!geometry || geometry.points.length === 0) {
    return {
      points: [],
      boxes: [],
      segments: [],
      caption: 'Waiting for the token table.',
      duration: 4,
      readout: 'Loading the width'
    };
  }
  const byId = new Map(geometry.points.map((point) => [point.id, point]));
  const tour = spaceTour(geometry);
  const station = 0.4;
  const duration = Math.max(4, tour.length * station);
  const cursor = tour[Math.floor(wrap(time, duration) / station) % tour.length] ?? tour[0];
  const active = byId.get(cursor);
  const keep = new Set<number>([cursor, ...(active?.neighbors ?? [])]);
  const stride = Math.max(1, Math.floor(geometry.points.length / 220));
  const points: SimPoint[] = [];
  const segments: SimSegment[] = [];
  const place = (point: GeoPoint, which: 'start' | 'end') => ({
    x: (which === 'start' ? (point.sx ?? point.x) : point.x) * 2.2,
    y: (which === 'start' ? (point.sy ?? point.y) : point.y) * 2.2,
    z: (which === 'start' ? (point.sz ?? point.z) : point.z) * 2.2
  });
  geometry.points.forEach((point, index) => {
    const focus = keep.has(point.id);
    if (!focus && index % stride !== 0) return;
    const end = place(point, 'end');
    const neighborIndex = active?.neighbors.indexOf(point.id) ?? -1;
    const cosine = neighborIndex >= 0 ? active?.neighborCos?.[neighborIndex] : undefined;
    points.push({
      id: `end${point.id}`,
      ...end,
      color: point.id === cursor ? COPPER : neighborIndex >= 0 ? SAND : INK,
      size: point.id === cursor ? 12 : neighborIndex >= 0 ? 8 : 3.2,
      label:
        point.id === cursor
          ? `${point.glyph} end`
          : neighborIndex === 0 && cosine != null
            ? `${point.glyph} ${cosine.toFixed(2)}`
            : '',
      detail:
        point.id === cursor
          ? `End state of “${point.glyph}” in the ${geometry.width}-d table from ${geometry.source}. Travel from its start row is ${(point.travel ?? 0).toFixed(2)}. Cosine with that start row is ${(point.cosineToStart ?? 0).toFixed(2)}.`
          : cosine == null
            ? `Another trained row, drawn in the same three axes.`
            : `Cosine with “${active?.glyph ?? ''}” in the full ${geometry.width}-d width is ${cosine.toFixed(2)}. The line is that relation, drawn after the table was flattened to three axes.`
    });
    if (geometry.startKind === 'seed') {
      const start = place(point, 'start');
      points.push({
        id: `start${point.id}`,
        ...start,
        color: point.id === cursor ? '#3f7a74' : '#8a8175',
        size: point.id === cursor ? 9 : focus ? 5 : 2.2,
        label: point.id === cursor ? `${point.glyph} start` : '',
        detail: `Start row of “${point.glyph}” before training, rebuilt with seed ${geometry.seed ?? '—'}.`
      });
    }
  });
  if (active && geometry.startKind === 'seed') {
    const start = place(active, 'start');
    const end = place(active, 'end');
    segments.push({ ...startTo(start), ...endTo(end), color: '#3f7a74' });
    active.neighbors.forEach((neighbor) => {
      const other = byId.get(neighbor);
      if (!other) return;
      const otherEnd = place(other, 'end');
      segments.push({
        x1: end.x,
        y1: end.y,
        z1: end.z,
        x2: otherEnd.x,
        y2: otherEnd.y,
        z2: otherEnd.z,
        color: COPPER
      });
    });
  }
  const nearest = active?.neighbors?.[0];
  const nearestPoint = nearest == null ? undefined : byId.get(nearest);
  const nearestCos = active?.neighborCos?.[0];
  const file = geometry.source.split('/').pop() ?? geometry.source;
  const shown = Math.min(active?.preview?.length ?? 0, active?.startHead?.length ?? 0);
  return {
    points,
    boxes: active && shown ? coordinateStrip(active, shown, geometry.width) : [],
    segments,
    caption: `Token rows of ${file} in their ${geometry.width}-number width. Three shared axes are only a shadow of that table. A green point is the row at initialization. Copper is the same row after training. Copper lines are nearest neighbors by cosine in the full width. The bars under the cloud are the first ${shown || 16} coordinates of the active row, green at the start and copper after training. Mean travel ${geometry.meanTravel ?? 0}, mean cosine with the start row ${geometry.meanCosineToStart ?? 0}.`,
    duration,
    readout: active
      ? `“${active.glyph}” · travel ${(active.travel ?? 0).toFixed(2)} in ${geometry.width}-d · cosine to start ${(active.cosineToStart ?? 0).toFixed(2)} · nearest “${nearestPoint?.glyph ?? '—'}” ${nearestCos == null ? '' : nearestCos.toFixed(2)}`
      : 'No row'
  };
}

function coordinateStrip(active: GeoPoint, count: number, width: number): SimBox[] {
  const endHead = active.preview ?? [];
  const startHead = active.startHead ?? [];
  const boxes: SimBox[] = [];
  const origin = -((count - 1) * 0.28) / 2;
  for (let index = 0; index < count; index += 1) {
    const pair = [
      ['start', startHead[index] ?? 0, -0.07, '#3f7a74'],
      ['end', endHead[index] ?? 0, 0.07, '#c4622d']
    ] as const;
    for (const [kind, value, shift, color] of pair) {
      const magnitude = Math.max(0.04, Math.abs(value) * 2.6);
      const direction = value < 0 ? -1 : 1;
      boxes.push({
        id: `${kind}${active.id}d${index}`,
        x: origin + index * 0.28 + shift,
        y: -2.85 + direction * magnitude,
        z: 0.15,
        dx: 0.045,
        dy: magnitude,
        dz: 0.045,
        color,
        label: '',
        detail: `${kind === 'start' ? 'Start' : 'End'} coordinate ${index + 1} of the ${width}-d row “${active.glyph}” is ${value.toFixed(3)}. The strip shows ${count} of those coordinates.`
      });
    }
  }
  return boxes;
}

function startTo(point: { x: number; y: number; z: number }) {
  return { x1: point.x, y1: point.y, z1: point.z };
}

function endTo(point: { x: number; y: number; z: number }) {
  return { x2: point.x, y2: point.y, z2: point.z };
}

export function stackScene(shape: SimInput['shape'], time: number, playing: boolean, exploded: boolean, query: number): Scene {
  const plates = ['Embed', ...Array.from({ length: shape.nLayer }, (_, index) => `Block ${index + 1}`), 'Head'];
  const duration = plates.length * 1.4;
  const local = wrap(time, duration);
  const cursor = playing ? Math.min(plates.length - 1, Math.floor(local / 1.4)) : Math.min(query, plates.length - 1);
  const frac = playing ? (local - cursor * 1.4) / 1.4 : 0;
  const open = (playing && plates[cursor].startsWith('Block')) || (exploded && plates[cursor]?.startsWith('Block'));
  const boxes: SimBox[] = [];
  const points: SimPoint[] = [];
  plates.forEach((label, index) => {
    if (open && index === cursor) {
      const parts = [
        ['Norm', '#3f7a74'],
        ['QKV', COPPER],
        ['Scores', SAND],
        ['MLP', '#8d3f16']
      ] as const;
      const part = Math.min(3, Math.floor(frac * 4));
      parts.forEach(([name, color], piece) => {
        boxes.push({
          id: `${label}-${name}`,
          x: (piece - 1.5) * 0.72,
          y: 0.28,
          z: 0,
          dx: 0.24,
          dy: piece === part ? 0.34 : 0.18,
          dz: 0.24,
          color,
          label: `${label} ${name}`,
          detail: `${label} is open. The packet is in ${parts[part][0]}. The stream here is [batch, ${shape.blockSize}, ${shape.nEmbd}].`
        });
      });
      points.push({
        id: 'packet',
        x: (part - 1.5) * 0.72,
        y: 0.85,
        z: 0.2,
        color: SAND,
        size: 10,
        label: 'Residual',
        detail: `Inside ${label}, at ${parts[part][0]}.`
      });
      return;
    }
    boxes.push({
      id: label,
      x: 0,
      y: index * 0.48,
      z: 0,
      dx: 0.8,
      dy: index === cursor ? 0.16 : 0.1,
      dz: 0.5,
      color: index === cursor ? COPPER : PINE,
      label,
      detail: `${label} reads and writes [batch, ${shape.blockSize}, ${shape.nEmbd}]. ${shape.nLayer} blocks, ${shape.nHead} heads.`
    });
  });
  if (!open) {
    points.push({
      id: 'packet',
      x: 0,
      y: cursor * 0.48 + 0.35 + frac * 0.2,
      z: 0.35,
      color: SAND,
      size: 10,
      label: 'Residual',
      detail: `The residual stream is on ${plates[cursor]}.`
    });
  }
  return {
    points,
    boxes,
    segments: [],
    caption: 'A residual packet climbing the stack. On a block it opens norm, QKV, scores, and the MLP, then moves on.',
    duration,
    readout: open ? `${plates[cursor]} open` : `Passing ${plates[cursor]}`
  };
}

export function unigramStep(ids: number[], time: number): { start: number; focus: number; target: number; loss: number; sharpen: number; bars: { id: number; p: number }[] } {
  const block = 8;
  const hops = Math.max(1, ids.length - block - 1);
  const duration = hops * 1.6;
  const local = wrap(time, duration);
  const start = Math.min(hops - 1, Math.floor(local / 1.6));
  const sharpen = Math.min(1, Math.max(0, (local - start * 1.6) / 1.6));
  const counts = new Map<number, number>();
  for (const id of ids) counts.set(id, (counts.get(id) ?? 0) + 1);
  const focus = start + block - 1;
  const target = ids[focus + 1] ?? ids[focus];
  const ranked = [...counts.entries()].sort((left, right) => right[1] - left[1]).slice(0, 5);
  if (!ranked.some(([id]) => id === target)) ranked.push([target, counts.get(target) ?? 1]);
  const raw = ranked.map(([id, count]) => ({ id, p: count / ids.length }));
  const mass = raw.reduce((sum, row) => sum + row.p, 0) || 1;
  const bars = raw.map((row) => {
    const base = row.p / mass;
    const aimed = row.id === target ? 1 : 0;
    return { id: row.id, p: (1 - sharpen) * base + sharpen * aimed };
  });
  const targetP = Math.max(1e-8, bars.find((row) => row.id === target)?.p ?? 1e-8);
  return { start, focus, target, loss: -Math.log(targetP), sharpen, bars };
}

export function trainStepScene(ids: number[], time: number): Scene {
  if (ids.length < 4) {
    return { points: [], boxes: [], segments: [], caption: 'Need the book stream for a step.', duration: 4, readout: 'Waiting for tokens' };
  }
  const step = unigramStep(ids, time);
  const block = 8;
  const boxes: SimBox[] = [];
  for (let index = 0; index < block; index += 1) {
    const id = ids[step.start + index];
    boxes.push({
      id: `c${index}`,
      x: (index - 3.5) * 0.62,
      y: 0.36,
      z: -1.1,
      dx: 0.18,
      dy: index === block - 1 ? 0.42 : 0.28,
      dz: 0.18,
      color: index === block - 1 ? COPPER : PINE,
      label: glyphForToken(id),
      detail: `Context id ${id}. The copper token is the one about to predict.`
    });
  }
  step.bars.forEach((row, index) => {
    const height = 0.15 + row.p * 1.8;
    boxes.push({
      id: `p${row.id}`,
      x: (index - (step.bars.length - 1) / 2) * 0.7,
      y: height / 2,
      z: 0.85,
      dx: 0.22,
      dy: height / 2,
      dz: 0.22,
      color: row.id === step.target ? COPPER : PINE,
      label: glyphForToken(row.id),
      detail: `Score mass ${(row.p * 100).toFixed(1)}% on id ${row.id}. Sharpen ${(step.sharpen * 100).toFixed(0)}% of the way from the excerpt unigram toward the true next token.`
    });
  });
  return {
    points: [],
    boxes,
    segments: [],
    caption: 'One training beat on the book window. The bars start as this excerpt’s unigram, then probability moves onto the true next token and the loss falls. A real step does that with the network’s logits.',
    duration: Math.max(4, (ids.length - 9) * 1.6),
    readout: `Predict ${glyphForToken(step.target)} · loss ${step.loss.toFixed(2)} nats · update ${(step.sharpen * 100).toFixed(0)}%`
  };
}

export function lossScene(metrics: MetricRow[], time: number): Scene {
  if (metrics.length === 0) return trainStepScene([], time);
  const duration = Math.max(6, metrics.length * 0.35);
  const shown = Math.max(1, Math.min(metrics.length, Math.ceil((wrap(time, duration) / duration) * metrics.length)));
  const points: SimPoint[] = [];
  const segments: SimSegment[] = [];
  metrics.slice(0, shown).forEach((row, index) => {
    const x = metrics.length < 2 ? 0 : (index / (metrics.length - 1)) * 4 - 2;
    const y = Math.max(0, 2.4 - row.loss / 2.2);
    points.push({
      id: `m${row.step}`,
      x,
      y,
      z: row.lr * 400,
      color: row.phase === 'finetune' ? PINE : COPPER,
      size: index === shown - 1 ? 10 : 5,
      label: `step ${row.step}`,
      detail: `Recorded loss ${row.loss.toFixed(3)} at learning rate ${row.lr}. The ribbon grows as play advances through the logged steps.`
    });
    if (index > 0) {
      const previous = metrics[index - 1];
      segments.push({
        x1: ((index - 1) / (metrics.length - 1)) * 4 - 2,
        y1: Math.max(0, 2.4 - previous.loss / 2.2),
        z1: previous.lr * 400,
        x2: x,
        y2: y,
        z2: row.lr * 400,
        color: '#8d3f16'
      });
    }
  });
  const tip = metrics[shown - 1];
  return {
    points,
    boxes: [],
    segments,
    caption: 'Replay of the logged loss. Play draws the run forward. Scrub to an earlier step. Depth is the learning rate.',
    duration,
    readout: `Step ${tip.step} · loss ${tip.loss.toFixed(3)} · ${shown} / ${metrics.length} logged points`
  };
}

export function machineScene(input: SimInput): Scene {
  const cap = 65;
  const bars = [
    ['Power', (input.powerLast ?? 0) / cap, COPPER, input.powerLast != null ? `${input.powerLast} W now, peak ${input.powerPeak ?? '—'} W` : 'No power sample in this run.'],
    ['Card', input.vramTotal ? (input.vramLast ?? 0) / input.vramTotal : 0, PINE, input.vramLast != null ? `${Math.round(input.vramLast / 1024 / 1024)} MB now, peak ${Math.round((input.vramPeak ?? 0) / 1024 / 1024)} MB` : 'No memory sample in this run.'],
    ['Junction', Math.min(1, (input.junctionLast ?? 0) / 100), '#8d3f16', input.junctionLast != null ? `${input.junctionLast} °C` : 'No junction temperature in this run.']
  ] as const;
  const boxes: SimBox[] = bars.map(([label, fraction, color, detail], index) => {
    const height = 0.25 + fraction * 2;
    return {
      id: label,
      x: (index - 1) * 1.15,
      y: height / 2,
      z: 0,
      dx: 0.3,
      dy: height / 2,
      dz: 0.3,
      color,
      label,
      detail
    };
  });
  return {
    points: [],
    boxes,
    segments: [],
    caption: 'The last power, memory, and temperature written by a run. These bars follow the sampler. They do not invent watts between steps.',
    duration: 4,
    readout: bars.map(([label, , , detail]) => `${label}: ${detail}`).join(' · ')
  };
}

export function maskScene(prompt: string, completion: string, time: number): Scene {
  const promptTokens = encodeText(prompt);
  const answerTokens = encodeText(completion);
  const tokens = [...promptTokens, ...answerTokens].slice(0, 24);
  const promptLen = promptTokens.length;
  if (tokens.length < 2) {
    return { points: [], boxes: [], segments: [], caption: 'Loading the tokenizer for the mask.', duration: 4, readout: 'Waiting' };
  }
  const duration = (tokens.length + 1) * 0.38;
  const cursor = Math.min(tokens.length, Math.floor(wrap(time, duration) / 0.38));
  const boxes: SimBox[] = tokens.map((token, index) => {
    const seen = index < cursor;
    const supervised = index + 1 >= promptLen;
    const raised = seen && supervised;
    return {
      id: `t${index}`,
      x: (index - (tokens.length - 1) / 2) * 0.4,
      y: raised ? 0.55 : 0.2,
      z: index < promptLen ? -0.15 : 0.35,
      dx: 0.14,
      dy: raised ? 0.32 : 0.12,
      dz: 0.14,
      color: !seen ? INK : supervised ? COPPER : PINE,
      label: token.glyph,
      detail: supervised
        ? `Position ${index} predicts ${tokens[index + 1]?.glyph ?? 'end'} and the label is kept.`
        : `Position ${index} still points at the question, so the label is −100.`
    };
  });
  const here = tokens[Math.min(cursor, tokens.length - 1)];
  const supervised = cursor >= promptLen;
  return {
    points: [],
    boxes,
    segments: [],
    caption: 'The loss mask sweeping the question and the answer. Pine has been seen and ignored. Copper has been seen and kept. Gray has not been visited yet.',
    duration,
    readout: cursor >= tokens.length ? 'Mask complete' : `At ${cursor} “${here?.glyph ?? ''}” · ${supervised ? 'loss on' : 'label −100'}`
  };
}

export function sampleFromProbs(step: number, probs: number[]): number {
  let state = (step + 1) * 1103515245 + 12345;
  state = Math.imul(state ^ (state >>> 16), 2246822519);
  const roll = ((state >>> 0) % 10000) / 10000;
  let cursor = 0;
  for (let index = 0; index < probs.length; index += 1) {
    cursor += probs[index];
    if (roll <= cursor) return index;
  }
  return Math.max(0, probs.length - 1);
}

export function sampleAt(step: number, logits: number[], temperature: number): number {
  const probs = softmax(logits, temperature <= 0 ? 0.0001 : temperature);
  if (temperature <= 0) return probs.indexOf(Math.max(...probs));
  return sampleFromProbs(step, probs);
}

export function sampleScene(
  temperature: number,
  time: number,
  live: { logits: number[]; checkpoint: string; topK: number } | null
): Scene {
  if (!live || live.logits.length === 0) {
    return {
      points: [],
      boxes: [],
      segments: [],
      caption: 'The sample stage runs one forward pass of the selected checkpoint and raises a bar for each likely next piece.',
      duration: 4,
      readout: 'Loading the live logits'
    };
  }
  const ranked = rankNext(live.logits, temperature, live.topK, 8);
  const bins = ranked.pieces.map((piece) => ({ ...piece }));
  if (ranked.rest > 0.005) bins.push({ id: -1, glyph: 'rest', logit: Number.NaN, prob: ranked.rest });
  const probs = bins.map((bin) => bin.prob);
  const draws = Math.max(1, Math.floor(wrap(time, 16) / 0.35) + 1);
  const counts = new Array(bins.length).fill(0);
  for (let step = 0; step < draws; step += 1) counts[sampleFromProbs(step, probs)] += 1;
  const latest = sampleFromProbs(draws - 1, probs);
  const tallest = Math.max(...probs, 1e-9);
  const file = live.checkpoint.split('/').pop() ?? live.checkpoint;
  const boxes: SimBox[] = bins.map((bin, index) => {
    const height = 0.18 + (bin.prob / tallest) * 1.7;
    return {
      id: `p${bin.id}`,
      x: (index - (bins.length - 1) / 2) * 0.58,
      y: height / 2,
      z: index === latest ? 0.25 : 0,
      dx: 0.18,
      dy: height / 2,
      dz: 0.18,
      color: index === latest ? COPPER : PINE,
      label: `${bin.glyph} ${(bin.prob * 100).toFixed(1)}%`,
      detail: `${file} gives “${bin.glyph}” probability ${(bin.prob * 100).toFixed(2)}% at temperature ${temperature.toFixed(2)}${Number.isFinite(bin.logit) ? `, logit ${bin.logit.toFixed(2)}` : ''}. Drawn ${counts[index]} of ${draws} times. Heights are scaled to the tallest bar.`
    };
  });
  const points: SimPoint[] = bins.map((bin, index) => ({
    id: `e${index}`,
    x: (index - (bins.length - 1) / 2) * 0.58,
    y: 0.15 + (counts[index] / draws) * 2.1,
    z: 0.85,
    color: SAND,
    size: 7,
    label: `${bin.glyph} seen`,
    detail: `Drawn ${counts[index]} times in ${draws} samples from ${file}.`
  }));
  const chosen = bins[latest];
  return {
    points,
    boxes,
    segments: chosen
      ? [
          {
            x1: (latest - (bins.length - 1) / 2) * 0.58,
            y1: 0.15 + (counts[latest] / draws) * 2.1,
            z1: 0.85,
            x2: (latest - (bins.length - 1) / 2) * 0.58,
            y2: 0.18 + (chosen.prob / tallest) * 1.7,
            z2: 0.25,
            color: COPPER
          }
        ]
      : [],
    caption: `Next-token distribution from ${file}. Pine bars are the live model. The copper bar is the piece drawn at this moment. Sand dots count repeated draws. Percentages are the real probabilities; heights are scaled to the tallest bar so a flat model is still readable.`,
    duration: 16,
    readout: `Draw ${draws}: “${chosen?.glyph ?? ''}” · ${((chosen?.prob ?? 0) * 100).toFixed(1)}% · ${file}`
  };
}

export function buildScene(input: SimInput): Scene {
  const path = input.path;
  if (path === '/') return pipelineScene(input.time);
  if (path === '/gather') return gatherScene(input.files, input.time);
  if (path === '/clean') return cleanScene(input.clean, input.time);
  if (path === '/prepare') return windowScene(input.excerptIds, input.time, input.trainTokens, input.valTokens);
  if (path === '/tokenize') return bpeScene(input.preview, input.time);
  if (path === '/architecture' && input.mode === 'space') return spaceScene(input.geometry, input.time);
  if (path === '/architecture' && input.mode === 'attention') {
    return forwardScene(input.geometry, input.time, input.playing, input.layer, input.head, input.query);
  }
  if (path === '/architecture' && input.mode === 'cloud') return lookupScene(input.geometry, input.excerptIds, input.time, input.filter);
  if (path === '/architecture' && input.mode === 'stack') {
    return stackScene(input.shape, input.time, input.playing, input.exploded, input.query);
  }
  if (path === '/architecture') return forwardScene(input.geometry, input.time, input.playing, input.layer, input.head, input.query);
  if (path === '/train' && input.mode === 'loss') {
    return input.metrics.length ? lossScene(input.metrics, input.time) : trainStepScene(input.excerptIds, input.time);
  }
  if (path === '/train' && input.mode === 'machine') return machineScene(input);
  if (path === '/train') return trainStepScene(input.excerptIds, input.time);
  if (path === '/finetune' && input.mode === 'loss') {
    return input.metrics.length ? lossScene(input.metrics, input.time) : maskScene(input.prompt, input.completion, input.time);
  }
  if (path === '/finetune') return maskScene(input.prompt, input.completion, input.time);
  return sampleScene(input.temperature, input.time, {
    logits: input.logits,
    checkpoint: input.checkpoint,
    topK: input.topK
  });
}
