<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/state';
  import { encodeText, glyphForToken, softmax } from '#lib/bytes.ts';
  import { ensureTokenizer, tokenizerState } from '#lib/tokenizer.svelte.ts';
  import { lab } from '#lib/lab.svelte.ts';
  import { pickRun } from '#lib/pickRun.ts';
  import type { RunStatus } from '#lib/types.ts';
  import Viewport, { type StageBox, type StagePoint, type StageSegment } from './Viewport.svelte';

  type GeoPoint = {
    id: number;
    glyph: string;
    x: number;
    y: number;
    z: number;
    neighbors: number[];
    preview: number[];
  };
  type Geometry = {
    source: string;
    width: number;
    layers: number;
    heads: number;
    points: GeoPoint[];
    attention: { text: string; ids: number[]; glyphs: string[]; layers: number[][][][] };
  };
  type DataSnap = {
    raw?: { source: string; bytes: number }[];
    clean?: { kept?: number; dropped?: number; reasons?: Record<string, number> } | null;
    prepare?: { train_tokens?: number; val_tokens?: number; assignment?: { source: string; split: string }[] } | null;
    excerpt?: { path: string; total: number; bytes: number[] } | null;
  };

  let geometry = $state<Geometry | null>(null);
  let geometryError = $state('');
  let data = $state<DataSnap | null>(null);
  let run = $state<RunStatus | null>(null);
  let filter = $state<'all' | 'letter' | 'digit' | 'space' | 'other'>('all');
  let layer = $state(0);
  let head = $state(0);
  let query = $state(12);
  let exploded = $state(false);
  let mode = $state('cloud');

  const path = $derived(page.url.pathname);

  let routed = '';
  $effect(() => {
    if (routed === path) return;
    routed = path;
    if (path === '/architecture') mode = 'cloud';
    else if (path === '/train') mode = 'loss';
    else if (path === '/finetune') mode = 'mask';
  });
  const demo = [
    { glyph: 'c', logit: 2.4 },
    { glyph: '␣', logit: 1.1 },
    { glyph: 'b', logit: 0.4 },
    { glyph: 'd', logit: 0.2 },
    { glyph: 't', logit: -0.4 },
    { glyph: 'x', logit: -1.6 }
  ];

  function kindOfGlyph(glyph: string) {
    const bare = glyph.replaceAll('␣', '').replaceAll('↵', '').trim();
    if (!bare) return 'space';
    if (/^\p{N}+$/u.test(bare)) return 'digit';
    if (/^\p{L}+$/u.test(bare)) return 'letter';
    return 'other';
  }

  function colorOfGlyph(glyph: string) {
    const kind = kindOfGlyph(glyph);
    if (kind === 'letter') return '#c4622d';
    if (kind === 'digit') return '#8d3f16';
    if (kind === 'space') return '#1e4642';
    return '#8a8175';
  }

  function count(value: number | undefined | null) {
    return value == null || Number.isNaN(value) ? '—' : value.toLocaleString();
  }

  onMount(() => {
    void ensureTokenizer();
    let stop = false;
    const pullGeometry = () => {
      fetch('/api/geometry')
        .then((response) => response.json())
        .then((body) => {
          if (body.error) geometryError = body.error;
          else geometry = body as Geometry;
        })
        .catch(() => {
          geometryError = 'The embedding projection did not load.';
        });
    };
    const pullData = () => {
      fetch('/api/data')
        .then((response) => response.json())
        .then((body) => {
          if (!body.error) data = body as DataSnap;
        })
        .catch(() => undefined);
    };
    const pullRun = () => {
      fetch('/api/runs')
        .then((response) => response.json())
        .then((body) => {
          const runs = (body.runs ?? []) as RunStatus[];
          const wanted = path === '/finetune' ? 'finetune' : 'pretrain';
          run = pickRun(runs, wanted) ?? (path === '/train' || path === '/finetune' ? null : runs.find((item) => item.state === 'running') ?? null);
        })
        .catch(() => undefined);
    };
    pullGeometry();
    pullData();
    pullRun();
    const timer = setInterval(() => {
      if (stop) return;
      pullRun();
      if (path === '/gather' || path === '/clean' || path === '/prepare') pullData();
    }, 800);
    return () => {
      stop = true;
      clearInterval(timer);
    };
  });

  const scene = $derived.by(() => {
    void tokenizerState.ready;
    const points: StagePoint[] = [];
    const boxes: StageBox[] = [];
    const segments: StageSegment[] = [];
    const route = path;
    if (route === '/') {
      const stations = ['Gather', 'Clean', 'Prepare', 'Shape', 'Train', 'Finetune', 'Sample'];
      stations.forEach((label, index) => {
        boxes.push({
          id: label,
          x: (index - 3) * 0.85,
          y: 0.35,
          z: 0,
          dx: 0.32,
          dy: 0.28,
          dz: 0.32,
          color: index === 4 ? '#c4622d' : '#1e4642',
          label,
          detail: 'One stage of the lab. Orbit the row, then open that page. The work moves from documents to bytes to a residual stream.'
        });
      });
      return { points, boxes, segments, caption: 'The pipeline as a row you can turn. Click a stage.' };
    }
    if (route === '/gather') {
      const files = data?.raw ?? [];
      files.slice(0, 36).forEach((file, index) => {
        const height = Math.max(0.2, Math.min(1.6, Math.log10(file.bytes + 10) / 3));
        boxes.push({
          id: file.source,
          x: (index % 6) * 0.7 - 1.5,
          y: height / 2,
          z: Math.floor(index / 6) * 0.7 - 0.6,
          dx: 0.24,
          dy: height / 2,
          dz: 0.24,
          color: '#d89a55',
          label: file.source.split('/').pop() ?? file.source,
          detail: `${file.bytes.toLocaleString()} bytes on disk. Height is the log of that size. Nothing has been filtered yet.`
        });
      });
      return { points, boxes, segments, caption: files.length ? 'Each box is one raw file.' : 'Gather a document and these boxes appear.' };
    }
    if (route === '/clean') {
      const reasons = Object.entries(data?.clean?.reasons ?? {});
      reasons.forEach(([reason, amount], index) => {
        const height = Math.max(0.25, Math.min(2, amount / 4));
        boxes.push({
          id: reason,
          x: index * 0.9 - (reasons.length - 1) * 0.45,
          y: height / 2,
          z: 0,
          dx: 0.28,
          dy: height / 2,
          dz: 0.28,
          color: '#8d2b22',
          label: reason,
          detail: `${amount} documents dropped for this reason. Kept ${data?.clean?.kept ?? 0}, dropped ${data?.clean?.dropped ?? 0}.`
        });
      });
      boxes.push({
        id: 'kept',
        x: 0,
        y: 0.2,
        z: 1.3,
        dx: 0.7,
        dy: 0.16,
        dz: 0.4,
        color: '#1e4642',
        label: 'Kept',
        detail: `${data?.clean?.kept ?? 0} documents survived cleaning and deduplication.`
      });
      return { points, boxes, segments, caption: 'Red boxes are drop reasons. The pine slab is what stayed.' };
    }
    if (route === '/prepare' || route === '/tokenize') {
      const bytes = data?.excerpt?.bytes ?? [];
      bytes.forEach((id, index) => {
        const angle = index * 0.33;
        const glyph = glyphForToken(id);
        points.push({
          id: `b${index}`,
          x: Math.cos(angle) * 1.35,
          y: ((id % 256) - 128) / 90,
          z: Math.sin(angle) * 1.35,
          color: colorOfGlyph(glyph),
          size: 7,
          label: glyph,
          detail: `Token ${index} of ${data?.excerpt?.path ?? 'the stream'} is id ${id} (${glyph}). The helix is the start of the book stream, ${data?.excerpt?.total?.toLocaleString() ?? '0'} tokens long. Train ${data?.prepare?.train_tokens?.toLocaleString() ?? '—'}, validation ${data?.prepare?.val_tokens?.toLocaleString() ?? '—'}.`
        });
      });
      return {
        points,
        boxes,
        segments,
        caption: route === '/tokenize'
          ? 'The first pieces of the book stream. Click one to read the id from the 8192-piece vocabulary.'
          : 'A helix of the first training tokens. Click one to read its piece.'
      };
    }
    if (route === '/architecture' && mode === 'cloud') {
      if (!geometry) return { points, boxes, segments, caption: 'Projecting the embedding. This can take a moment the first time PyTorch loads.' };
      const glyphs = new Map(geometry.points.map((item) => [item.id, item.glyph]));
      for (const point of geometry.points) {
        if (filter !== 'all' && kindOfGlyph(point.glyph) !== filter) continue;
        const names = point.neighbors.map((id) => glyphs.get(id) ?? String(id)).join(' ');
        points.push({
          id: `e${point.id}`,
          x: point.x * 2.4,
          y: point.y * 2.4,
          z: point.z * 2.4,
          color: colorOfGlyph(point.glyph),
          size: 6,
          label: `${point.glyph} · ${point.id}`,
          detail: `One of ${geometry.points.length.toLocaleString()} token vectors. This dot is 3 principal components of a ${geometry.width}-d embedding from ${geometry.source}. Nearest in the full ${geometry.width}-d space: ${names}. First 8 coordinates: ${point.preview.join(', ')}.`
        });
      }
      return { points, boxes, segments, caption: `Principal components of the real ${geometry.width}-d table (${geometry.source}). Neighbors are from the full vector, not from this picture.` };
    }
    if (route === '/architecture' && mode === 'attention') {
      if (!geometry) return { points, boxes, segments, caption: 'Loading attention from the checkpoint.' };
      const model = geometry;
      const layers = model.attention.layers;
      const safeLayer = Math.min(layer, layers.length - 1);
      const heads = layers[safeLayer] ?? [];
      const grid = heads[Math.min(head, heads.length - 1)] ?? [];
      const glyphs = model.attention.glyphs;
      const mid = (glyphs.length - 1) / 2;
      grid.forEach((row, q) => {
        row.forEach((weight, k) => {
          if (k > q || weight < 0.004) return;
          const x = (k - mid) * 0.34;
          const y = weight * 2.6;
          const z = (q - mid) * 0.34;
          const selected = q === Math.min(query, glyphs.length - 1);
          points.push({
            id: `a${q}-${k}`,
            x,
            y,
            z,
            color: selected ? '#c4622d' : '#1e4642',
            size: selected ? 7 + weight * 10 : 4,
            label: `${glyphs[q]} ← ${glyphs[k]}`,
            detail: `Attention ${weight.toFixed(3)} at layer ${safeLayer + 1}, head ${Math.min(head, heads.length - 1) + 1}. Query “${glyphs[q]}” reading key “${glyphs[k]}” in “${model.attention.text}”. Height is the weight. Keys after the query are masked, so they are not drawn.`
          });
          if (selected) {
            segments.push({ x1: x, y1: 0, z1: z, x2: x, y2: y, z2: z, color: '#c4622d' });
          }
        });
      });
      return { points, boxes, segments, caption: `Causal attention for “${model.attention.text}”. Copper is the query you selected. Change layer, head, or query.` };
    }
    if (route === '/architecture') {
      const shape = lab.shape;
      const plates = ['Bytes', 'Embed', ...Array.from({ length: shape.nLayer }, (_, index) => `Block ${index + 1}`), 'Tied head'];
      plates.forEach((label, index) => {
        const isBlock = label.startsWith('Block');
        const open = exploded && isBlock && index === Math.min(query, plates.length - 1);
        if (open) {
          const parts = [
            ['Norm', '#3f7a74'],
            ['QKV', '#c4622d'],
            ['Scores', '#d89a55'],
            ['MLP', '#8d3f16']
          ] as const;
          parts.forEach(([name, color], part) => {
            boxes.push({
              id: `${label}-${name}`,
              x: (part - 1.5) * 0.7,
              y: 0.3,
              z: 0.2,
              dx: 0.26,
              dy: 0.22,
              dz: 0.26,
              color,
              label: `${label} ${name}`,
              detail: `Inside ${label}. The stream at this point is [batch, ${shape.blockSize}, ${shape.nEmbd}]. QKV is one matrix ${shape.nEmbd} → ${shape.nEmbd * 3}. The MLP widens by ${shape.mlpRatio}. This is the shape on the sliders, drawn as volumes you can pick apart.`
            });
          });
          return;
        }
        boxes.push({
          id: label,
          x: 0,
          y: index * 0.55,
          z: 0,
          dx: label === 'Bytes' ? 0.7 : 0.85,
          dy: 0.12,
          dz: 0.55,
          color: isBlock ? '#c4622d' : '#1e4642',
          label,
          detail: `${label} reads and writes [batch, ${shape.blockSize}, ${shape.nEmbd}]. ${shape.nLayer} blocks, ${shape.nHead} heads, MLP ratio ${shape.mlpRatio}. Click a block, then Explode, to open it. Drag and scroll to move around the stack.`
        });
      });
      return { points, boxes, segments, caption: 'The residual stack for the sliders. Explode opens the selected block into norm, QKV, scores, and MLP.' };
    }
    if (route === '/train' && mode === 'machine') {
      const power = run?.compute?.power_w;
      const cap = run?.compute?.power_cap_w || 65;
      const vram = run?.compute?.vram_bytes;
      const total = run?.compute?.vram_total_bytes || 1;
      const junction = run?.compute?.temperature_c?.junction;
      const bars = [
        ['Power', (power?.last ?? 0) / cap, '#c4622d', power ? `${power.last} W now. Min ${power.min}, average ${power.avg}, peak ${power.max}. Cap ${cap} W.` : 'Waiting for the first step.'],
        ['Card memory', (vram?.last ?? 0) / total, '#1e4642', vram ? `${Math.round(vram.last / 1024 / 1024)} MB now. Min ${Math.round(vram.min / 1024 / 1024)}, average ${Math.round(vram.avg / 1024 / 1024)}, peak ${Math.round(vram.max / 1024 / 1024)}.` : 'Waiting for the first step.'],
        ['Junction', Math.min(1, (junction?.last ?? 0) / 100), '#8d3f16', junction ? `${junction.last} °C now. Min ${junction.min}, average ${junction.avg}, peak ${junction.max}.` : 'Waiting for the first step.']
      ] as const;
      bars.forEach(([label, fraction, color, detail], index) => {
        const height = 0.25 + fraction * 2;
        boxes.push({
          id: label,
          x: (index - 1) * 1.1,
          y: height / 2,
          z: 0,
          dx: 0.32,
          dy: height / 2,
          dz: 0.32,
          color,
          label,
          detail
        });
      });
      return { points, boxes, segments, caption: 'Live machine meters. They grow while the step loop is running.' };
    }
    if (route === '/train' || (route === '/finetune' && mode === 'loss')) {
      const metrics = run?.metrics ?? [];
      metrics.forEach((row, index) => {
        const x = metrics.length < 2 ? 0 : (index / (metrics.length - 1)) * 4 - 2;
        points.push({
          id: `m${row.step}`,
          x,
          y: Math.max(0, 2.2 - row.loss / 2.5),
          z: row.lr * 400,
          color: row.phase === 'finetune' ? '#1e4642' : '#c4622d',
          size: 6,
          label: `step ${row.step}`,
          detail: `Loss ${row.loss.toFixed(3)}, learning rate ${row.lr}. Height falls as the loss falls. Depth is the learning rate.`
        });
        if (index > 0) {
          const previous = metrics[index - 1];
          const x0 = ((index - 1) / (metrics.length - 1)) * 4 - 2;
          segments.push({
            x1: x0,
            y1: Math.max(0, 2.2 - previous.loss / 2.5),
            z1: previous.lr * 400,
            x2: x,
            y2: Math.max(0, 2.2 - row.loss / 2.5),
            z2: row.lr * 400,
            color: '#8d3f16'
          });
        }
      });
      return { points, boxes, segments, caption: metrics.length ? 'The loss ribbon updates on every logged step, including while the run is still going.' : 'The ribbon appears on the first logged step.' };
    }
    if (route === '/finetune') {
      const ids = [...encodeText(lab.prompt), ...encodeText(lab.completion)];
      const promptLen = encodeText(lab.prompt).length;
      const shown = ids.slice(0, 64);
      shown.forEach((token, index) => {
        const supervised = index + 1 >= promptLen && index + 1 < ids.length;
        boxes.push({
          id: `t${index}`,
          x: (index - shown.length / 2) * 0.42,
          y: supervised ? 0.55 : 0.16,
          z: index < promptLen ? -0.2 : 0.35,
          dx: 0.16,
          dy: supervised ? 0.34 : 0.12,
          dz: 0.16,
          color: supervised ? '#c4622d' : '#1e4642',
          label: token.glyph,
          detail: supervised
            ? `Position ${index} predicts token ${ids[index + 1]?.id ?? '—'} (${ids[index + 1]?.glyph ?? ''}). That label is in the loss.`
            : `Position ${index} is still inside the question, so its label is −100 and Adam does not see it. The last question token is the exception: it predicts the first answer token.`
        });
      });
      return { points, boxes, segments, caption: 'Copper positions enter the loss. Pine positions are context. Edit the prompt and this row changes. Drag to read along it.' };
    }
    const probs = softmax(demo.map((row) => row.logit), lab.temperature <= 0 ? 0.0001 : lab.temperature);
    const shown = lab.temperature <= 0 ? demo.map((_, index) => (index === 0 ? 1 : 0)) : probs;
    demo.forEach((row, index) => {
      const height = 0.2 + shown[index] * 2.2;
      boxes.push({
        id: row.glyph,
        x: (index - 2.5) * 0.7,
        y: height / 2,
        z: 0,
        dx: 0.22,
        dy: height / 2,
        dz: 0.22,
        color: shown[index] === Math.max(...shown) ? '#c4622d' : '#1e4642',
        label: row.glyph,
        detail: `Logit ${row.logit}. Probability ${(shown[index] * 100).toFixed(1)}% at temperature ${lab.temperature.toFixed(2)}. The temperature slider on the page drives these bars.`
      });
    });
    return { points, boxes, segments, caption: 'The same six toy logits, as volumes. Temperature 0 keeps only the tallest.' };
  });

  const showLive = $derived(path === '/train' || path === '/finetune' || run?.state === 'running');
  const modes = $derived(
    path === '/architecture'
      ? [
          ['cloud', 'Embedding'],
          ['attention', 'Attention'],
          ['stack', 'Stack']
        ]
      : path === '/train'
        ? [
            ['loss', 'Loss'],
            ['machine', 'Machine']
          ]
        : path === '/finetune'
          ? [
              ['mask', 'Mask'],
              ['loss', 'Loss']
            ]
          : []
  );
</script>

<aside class="dock">
  <p class="kicker">Stage</p>
  {#if showLive}
    <div class="live">
      <div><span>training stream</span><strong>{count(data?.prepare?.train_tokens)}</strong></div>
      <div><span>validation</span><strong>{count(data?.prepare?.val_tokens)}</strong></div>
      <div><span>tokens trained</span><strong>{count(run?.tokensTrained)}</strong></div>
      <div><span>tokens seen</span><strong>{count(run?.tokensSeen)}</strong></div>
      <div><span>step</span><strong>{run ? `${run.step ?? 0} / ${run.totalSteps ?? '?'}` : '—'}</strong></div>
      <div><span>loss</span><strong>{run?.loss ? run.loss.toFixed(3) : '—'}</strong></div>
      <div><span>tokens/s</span><strong>{run?.tokensPerSec ? Math.round(run.tokensPerSec).toLocaleString() : '—'}</strong></div>
      <div><span>elapsed</span><strong>{run?.elapsedS != null ? `${run.elapsedS}s` : '—'}</strong></div>
      <div><span>power now</span><strong>{run?.compute?.power_w ? `${run.compute.power_w.last} W` : '—'}</strong></div>
      <div><span>power peak</span><strong>{run?.compute?.power_w ? `${run.compute.power_w.max} W` : '—'}</strong></div>
      <div><span>energy</span><strong>{run?.compute ? `${run.compute.energy_wh ?? 0} Wh` : '—'}</strong></div>
      <div><span>card peak</span><strong>{run?.compute?.vram_bytes ? `${Math.round(run.compute.vram_bytes.max / 1024 / 1024)} MB` : '—'}</strong></div>
      <div><span>process peak</span><strong>{run?.compute?.process_vram_bytes ? `${Math.round(run.compute.process_vram_bytes.max / 1024 / 1024)} MB` : '—'}</strong></div>
      <div><span>edge / junction</span><strong>{run?.compute?.temperature_c?.edge ? `${run.compute.temperature_c.edge.last}° / ${run.compute.temperature_c.junction?.last ?? '—'}°` : '—'}</strong></div>
    </div>
    <p class="note">
      {#if run?.state === 'running'}
        These update on every step, while training is still running.
      {:else if run?.id === 'corpus'}
        This is the finished byte-vocabulary pass over the books, {run.step?.toLocaleString()} steps. The stream on disk is now the 8192-piece encoding. A new pretrain starts at 20 steps and writes checkpoints/pretrain.pt.
      {:else if run}
        Last {run.phase} run. {run.state}.
      {:else}
        Start a run and the totals count here from step 1.
      {/if}
      {#if run?.tokensSeen != null && run.tokensTrained != null && run.tokensSeen !== run.tokensTrained}
        Tokens seen includes the question. Tokens trained are the positions whose labels were not −100.
      {/if}
    </p>
  {/if}

  {#if modes.length}
    <div class="modes">
      {#each modes as [id, label]}
        <button type="button" class="btn ghost" class:on={mode === id} onclick={() => (mode = id)}>{label}</button>
      {/each}
    </div>
  {/if}
  {#if path === '/architecture' && mode === 'cloud'}
    <div class="modes">
      {#each ['all', 'letter', 'digit', 'space', 'other'] as name}
        <button type="button" class="btn ghost" class:on={filter === name} onclick={() => (filter = name as typeof filter)}>{name}</button>
      {/each}
    </div>
  {/if}
  {#if path === '/architecture' && mode === 'attention' && geometry}
    <div class="modes">
      <label>Layer {Math.min(layer, geometry.layers - 1) + 1}
        <input type="range" min="0" max={geometry.layers - 1} bind:value={layer} />
      </label>
      <label>Head {Math.min(head, geometry.heads - 1) + 1}
        <input type="range" min="0" max={geometry.heads - 1} bind:value={head} />
      </label>
      <label>Query {query + 1}
        <input type="range" min="0" max={Math.max(0, geometry.attention.glyphs.length - 1)} bind:value={query} />
      </label>
    </div>
  {/if}
  {#if path === '/architecture' && mode === 'stack'}
    <div class="modes">
      <button type="button" class="btn ghost" class:on={exploded} onclick={() => (exploded = !exploded)}>{exploded ? 'Close block' : 'Explode block'}</button>
      <label>Plate {query + 1}
        <input type="range" min="0" max={lab.shape.nLayer + 1} bind:value={query} />
      </label>
    </div>
  {/if}

  {#if geometryError && path === '/architecture' && mode !== 'stack'}
    <p class="warn">{geometryError}</p>
  {:else if !geometry && path === '/architecture' && mode !== 'stack'}
    <p class="note">Projecting the {tokenizerState.vocabSize.toLocaleString()} embedding vectors…</p>
  {/if}

  <Viewport points={scene.points} boxes={scene.boxes} segments={scene.segments} ariaLabel={scene.caption} />
  <p class="note">{scene.caption}</p>
</aside>

<style>
  .dock { display: grid; align-content: start; gap: 10px; }
  .live {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px 12px;
  }
  .live div {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 8px 10px;
  }
  .live span {
    display: block;
    color: var(--ink-soft);
    font-size: 0.75rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  .live strong { font-family: var(--serif); font-size: 1.15rem; }
  .modes { display: flex; flex-wrap: wrap; gap: 8px; align-items: end; }
  .modes label { display: grid; gap: 4px; font-size: 0.82rem; color: var(--ink-soft); }
  .btn.on { background: var(--pine); color: #f6f3ec; border-color: var(--pine); }
</style>
