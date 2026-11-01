<script lang="ts">
  import { untrack } from 'svelte';
  import { browser } from '$app/env';
  import LossChart from './LossChart.svelte';
  import type { RunStatus } from '#lib/types.ts';

  let {
    phase,
    steps = 200,
    finetuneSteps = 80,
    learningRate = 0.003
  }: {
    phase: 'pretrain' | 'finetune' | 'pipeline';
    steps?: number;
    finetuneSteps?: number;
    learningRate?: number;
  } = $props();

  let stepCount = $state(untrack(() => steps));
  let ftCount = $state(untrack(() => finetuneSteps));
  let rate = $state(untrack(() => learningRate));
  let batchSize = $state(16);
  let seed = $state(7);
  let deviceChoice = $state<'auto' | 'cpu' | 'gpu'>('auto');
  let gfxOverride = $state(false);
  let hardware = $state<Record<string, unknown> | null>(null);
  let run = $state<RunStatus | null>(null);
  let error = $state('');
  let polling = $state(false);

  const running = $derived(run?.state === 'running');

  function megabytes(bytes: number) {
    return (bytes / (1024 * 1024)).toFixed(0);
  }
  const latestLoss = $derived(run?.loss);
  const perplexity = $derived(latestLoss && latestLoss > 0 ? Math.exp(latestLoss) : null);

  async function refresh(id: string) {
    const response = await fetch(`/api/runs/${id}`);
    const body = await response.json();
    if (body.run) run = body.run as RunStatus;
    if (run?.state === 'running') {
      polling = true;
      setTimeout(() => refresh(id), 800);
    } else {
      polling = false;
    }
  }

  async function start() {
    error = '';
    const response = await fetch('/api/runs', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        phase,
        steps: stepCount,
        finetuneSteps: ftCount,
        learningRate: rate,
        batchSize,
        seed,
        device: deviceChoice,
        gfxOverride
      })
    });
    const body = await response.json();
    if (!response.ok) {
      error = body.error ?? 'Could not start.';
      if (body.run) run = body.run;
      return;
    }
    run = {
      state: 'running',
      phase,
      step: 0,
      totalSteps: phase === 'finetune' ? ftCount : stepCount,
      metrics: [],
      tokensTrained: 0,
      tokensSeen: 0,
      elapsedS: 0,
      message: 'Starting. Totals update on each step.'
    };
    refresh(body.id);
  }

  $effect(() => {
    if (!browser) return;
    const gfx = gfxOverride;
    fetch(`/api/device?gfx=${gfx ? '1' : '0'}`)
      .then((response) => response.json())
      .then((body) => {
        if (!body.error) hardware = body;
      })
      .catch(() => undefined);
  });

  $effect(() => {
    if (!browser || polling || run) return;
    fetch('/api/runs')
      .then((response) => response.json())
      .then((body) => {
        const runs = (body.runs ?? []) as RunStatus[];
        const match = runs.find((candidate) => candidate.phase === phase);
        if (!match?.id) return;
        run = match;
        if (match.state === 'running') refresh(match.id);
      })
      .catch(() => undefined);
  });
</script>

<div class="card stack">
  {#if hardware}
    <p class="note">
      Device now: {hardware.torch_gpu ? `${hardware.backend} · ${hardware.name}` : hardware.name}
      {#if hardware.arch}({hardware.arch}){/if}.
      {#if hardware.rx5500_present}RX 5500 is on the PCI bus.{/if}
      {#if hardware.gpu_blocker}{hardware.gpu_blocker}{/if}
    </p>
  {/if}
  <div class="row">
    <label class="field">Device
      <select bind:value={deviceChoice}>
        <option value="auto">Auto</option>
        <option value="gpu">GPU</option>
        <option value="cpu">CPU</option>
      </select>
    </label>
    <label class="field">RX 5500 compatibility
      <input type="checkbox" bind:checked={gfxOverride} />
    </label>
    {#if phase !== 'finetune'}
      <label class="field">Pretrain steps
        <input type="number" min="1" max="2000" bind:value={stepCount} />
      </label>
    {/if}
    {#if phase !== 'pretrain'}
      <label class="field">Finetune steps
        <input type="number" min="1" max="2000" bind:value={ftCount} />
      </label>
    {/if}
    <label class="field">Batch
      <input type="number" min="1" max="64" bind:value={batchSize} />
    </label>
    <label class="field">Learning rate
      <input type="number" min="0.00001" max="0.1" step="0.0001" bind:value={rate} />
    </label>
    <label class="field">Seed
      <input type="number" min="0" bind:value={seed} />
    </label>
    <button class="btn" onclick={start} disabled={running}>
      {running ? 'Training' : phase === 'pipeline' ? 'Run both phases' : phase === 'finetune' ? 'Finetune' : 'Pretrain'}
    </button>
  </div>
  {#if error}<p class="warn">{error}</p>{/if}
  {#if run}
    <div class="stats">
      <div class="stat"><span class="note">{run.phase ?? phase}</span><strong>{run.state}</strong></div>
      <div class="stat"><span class="note">step</span><strong>{run.step ?? 0} / {run.totalSteps ?? '?'}</strong></div>
      <div class="stat"><span class="note">tokens trained</span><strong>{run.tokensTrained != null ? run.tokensTrained.toLocaleString() : '—'}</strong></div>
      <div class="stat"><span class="note">tokens/s</span><strong>{run.tokensPerSec ? Math.round(run.tokensPerSec).toLocaleString() : '—'}</strong></div>
      <div class="stat"><span class="note">loss</span><strong>{run.loss ? run.loss.toFixed(3) : '—'}</strong></div>
      <div class="stat"><span class="note">perplexity</span><strong>{perplexity ? perplexity.toFixed(1) : '—'}</strong></div>
      <div class="stat"><span class="note">ran on</span><strong>{run.device ?? '—'}{run.arch ? ` · ${run.arch}` : ''}</strong></div>
      {#if run.parameters}<div class="stat"><span class="note">parameters</span><strong>{run.parameters.toLocaleString()}</strong></div>{/if}
    </div>
    {#if run.compute}
      <table>
        <tbody>
          <tr>
            <td>Tokens trained</td>
            <td>{run.tokensTrained != null ? run.tokensTrained.toLocaleString() : '—'}{#if run.tokensSeen != null && run.tokensSeen !== run.tokensTrained} of {run.tokensSeen.toLocaleString()} seen{/if}{#if run.elapsedS != null} · {run.elapsedS}s{/if}</td>
          </tr>
          {#if run.compute.power_w}
            <tr>
              <td>Power</td>
              <td>{run.compute.power_w.min} / {run.compute.power_w.avg} / {run.compute.power_w.max} W min, average, peak{#if run.compute.power_cap_w} · cap {run.compute.power_cap_w} W{/if}</td>
            </tr>
          {/if}
          <tr><td>Energy</td><td>{run.compute.energy_wh ?? 0} Wh</td></tr>
          {#if run.compute.vram_bytes}
            <tr>
              <td>Card memory</td>
              <td>{megabytes(run.compute.vram_bytes.min)} / {megabytes(run.compute.vram_bytes.avg)} / {megabytes(run.compute.vram_bytes.max)} MB min, average, peak{#if run.compute.vram_total_bytes} of {megabytes(run.compute.vram_total_bytes)} MB{/if}</td>
            </tr>
          {/if}
          {#if run.compute.process_vram_bytes}
            <tr>
              <td>This process</td>
              <td>{megabytes(run.compute.process_vram_bytes.min)} / {megabytes(run.compute.process_vram_bytes.avg)} / {megabytes(run.compute.process_vram_bytes.max)} MB allocated min, average, peak</td>
            </tr>
          {/if}
          {#each Object.entries(run.compute.temperature_c ?? {}) as [label, stat]}
            <tr><td>{label}</td><td>{stat.last} °C now · {stat.min} min · {stat.max} peak</td></tr>
          {/each}
        </tbody>
      </table>
    {/if}
    {#if run.dataSource}<p class="note">Data: {run.dataSource}. Precision fp32.{#if run.tokensPerSec} {run.tokensPerSec} tokens/s.{/if}{#if run.valLoss} Validation loss {Number(run.valLoss).toFixed(3)}.{/if}</p>{/if}
    {#if run.message}<p class="note">{run.message}</p>{/if}
    <LossChart metrics={run.metrics ?? []} />
    {#if run.probes && run.probes.length}
      <div class="probes">
        <h3>Probes from this run</h3>
        {#each run.probes as probe}
          <div class="probe"><em>{probe.prompt}</em>{probe.completion.slice(probe.prompt.length)}</div>
        {/each}
      </div>
    {/if}
  {/if}
</div>
