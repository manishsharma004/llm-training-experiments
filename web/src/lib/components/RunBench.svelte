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
  let run = $state<RunStatus | null>(null);
  let error = $state('');
  let polling = $state(false);

  const running = $derived(run?.state === 'running');
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
        seed
      })
    });
    const body = await response.json();
    if (!response.ok) {
      error = body.error ?? 'Could not start.';
      if (body.run) run = body.run;
      return;
    }
    run = { state: 'running', phase, step: 0, totalSteps: phase === 'finetune' ? ftCount : stepCount, metrics: [], message: 'Starting.' };
    refresh(body.id);
  }

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
  <div class="row">
    {#if phase !== 'finetune'}
      <label class="field">Pretrain steps
        <input type="number" min="1" max="400" bind:value={stepCount} />
      </label>
    {/if}
    {#if phase !== 'pretrain'}
      <label class="field">Finetune steps
        <input type="number" min="1" max="400" bind:value={ftCount} />
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
      <div class="stat"><span class="note">loss</span><strong>{run.loss ? run.loss.toFixed(3) : '—'}</strong></div>
      <div class="stat"><span class="note">perplexity</span><strong>{perplexity ? perplexity.toFixed(1) : '—'}</strong></div>
    </div>
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
