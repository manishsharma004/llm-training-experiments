<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/state';
  import { ensureTokenizer, tokenizerState } from '#lib/tokenizer.svelte.ts';
  import { lab } from '#lib/lab.svelte.ts';
  import { pickRun } from '#lib/pickRun.ts';
  import { pauseSim, setSpeed, sim, toggleSim } from '#lib/sim.svelte.ts';
  import { buildScene } from '#lib/simulate.ts';
  import type { RunStatus } from '#lib/types.ts';
  import Viewport from './Viewport.svelte';

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
    clean?: {
      kept?: number;
      dropped?: number;
      reasons?: Record<string, number>;
      dropped_examples?: { source: string; reasons: string[]; excerpt: string }[];
      kept_preview?: { source: string; chars: number; after: string }[];
    } | null;
    prepare?: { train_tokens?: number; val_tokens?: number; preview?: string } | null;
    excerpt?: { path: string; total: number; bytes: number[] } | null;
  };

  let geometry = $state<Geometry | null>(null);
  let geometryError = $state('');
  let data = $state<DataSnap | null>(null);
  let run = $state<RunStatus | null>(null);
  let filter = $state<'all' | 'letter' | 'digit' | 'space' | 'other'>('all');
  let layer = $state(0);
  let head = $state(0);
  let query = $state(0);
  let exploded = $state(false);
  let mode = $state('forward');

  const path = $derived(page.url.pathname);

  let routed = '';
  $effect(() => {
    if (routed === path) return;
    routed = path;
    sim.time = 0;
    sim.playing = true;
    if (path === '/architecture') mode = 'forward';
    else if (path === '/train') mode = 'step';
    else if (path === '/finetune') mode = 'mask';
  });

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
      if (path === '/gather' || path === '/clean' || path === '/prepare' || path === '/tokenize' || path === '/') pullData();
    }, 1500);
    let last = performance.now();
    let frame = 0;
    const loop = (now: number) => {
      if (stop) return;
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      if (sim.playing) sim.time += dt * sim.speed;
      frame = requestAnimationFrame(loop);
    };
    frame = requestAnimationFrame(loop);
    return () => {
      stop = true;
      clearInterval(timer);
      cancelAnimationFrame(frame);
    };
  });

  const scene = $derived.by(() => {
    void tokenizerState.ready;
    return buildScene({
      path,
      time: sim.time,
      playing: sim.playing,
      mode,
      filter,
      layer,
      head,
      query,
      exploded,
      files: data?.raw ?? [],
      clean: data?.clean ?? null,
      trainTokens: data?.prepare?.train_tokens ?? null,
      valTokens: data?.prepare?.val_tokens ?? null,
      preview: data?.prepare?.preview ?? '',
      excerptIds: data?.excerpt?.bytes ?? [],
      metrics: run?.metrics ?? [],
      prompt: lab.prompt,
      completion: lab.completion,
      temperature: lab.temperature,
      shape: {
        nLayer: lab.shape.nLayer,
        nHead: lab.shape.nHead,
        nEmbd: lab.shape.nEmbd,
        blockSize: lab.shape.blockSize,
        mlpRatio: lab.shape.mlpRatio
      },
      geometry,
      powerLast: run?.compute?.power_w?.last ?? null,
      powerPeak: run?.compute?.power_w?.max ?? null,
      vramLast: run?.compute?.vram_bytes?.last ?? null,
      vramPeak: run?.compute?.vram_bytes?.max ?? null,
      vramTotal: run?.compute?.vram_total_bytes ?? null,
      junctionLast: run?.compute?.temperature_c?.junction?.last ?? null
    });
  });

  const showLive = $derived(path === '/train' || path === '/finetune' || run?.state === 'running');
  const modes = $derived(
    path === '/architecture'
      ? [
          ['forward', 'Forward'],
          ['attention', 'Attention'],
          ['cloud', 'Lookup'],
          ['stack', 'Stack']
        ]
      : path === '/train'
        ? [
            ['step', 'Step'],
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

  function scrub(event: Event) {
    pauseSim();
    sim.time = Number((event.currentTarget as HTMLInputElement).value);
  }

  function beat() {
    pauseSim();
    sim.time += 0.45;
  }
</script>

<aside class="dock">
  <p class="kicker">Simulation</p>
  <div class="modes">
    <button type="button" class="btn ghost" class:on={sim.playing} onclick={toggleSim}>{sim.playing ? 'Pause' : 'Play'}</button>
    <button type="button" class="btn ghost" onclick={beat}>Step</button>
    {#each [0.5, 1, 2] as speed}
      <button type="button" class="btn ghost" class:on={sim.speed === speed} onclick={() => setSpeed(speed)}>{speed}×</button>
    {/each}
  </div>
  <label class="scrub">
    <span>{scene.readout}</span>
    <input type="range" min="0" max={scene.duration} step="0.01" value={sim.time % scene.duration} oninput={scrub} />
  </label>
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
        <input type="range" min="0" max={geometry.layers - 1} bind:value={layer} oninput={pauseSim} />
      </label>
      <label>Head {Math.min(head, geometry.heads - 1) + 1}
        <input type="range" min="0" max={geometry.heads - 1} bind:value={head} oninput={pauseSim} />
      </label>
      <label>Query {query + 1}
        <input type="range" min="0" max={Math.max(0, geometry.attention.glyphs.length - 1)} bind:value={query} oninput={pauseSim} />
      </label>
    </div>
  {/if}
  {#if path === '/architecture' && mode === 'stack'}
    <div class="modes">
      <button type="button" class="btn ghost" class:on={exploded} onclick={() => (exploded = !exploded)}>{exploded ? 'Close block' : 'Explode block'}</button>
      <label>Plate {query + 1}
        <input type="range" min="0" max={lab.shape.nLayer + 1} bind:value={query} oninput={pauseSim} />
      </label>
    </div>
  {/if}

  {#if geometryError && path === '/architecture' && (mode === 'forward' || mode === 'attention' || mode === 'cloud')}
    <p class="warn">{geometryError}</p>
  {:else if !geometry && path === '/architecture' && mode !== 'stack'}
    <p class="note">Running the forward once so the simulation can use the real attention weights…</p>
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
  .scrub { display: grid; gap: 4px; font-size: 0.86rem; color: var(--ink-soft); }
  .scrub input { width: 100%; accent-color: var(--copper); }
</style>
