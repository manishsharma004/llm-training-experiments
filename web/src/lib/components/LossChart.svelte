<script lang="ts">
  import type { Metric } from '#lib/types.ts';
  import { CHANCE_NATS } from '#lib/params.ts';

  let { metrics = [] as Metric[] }: { metrics?: Metric[] } = $props();

  const width = 680;
  const height = 240;
  const pad = 32;

  const maxLoss = $derived(Math.max(CHANCE_NATS, ...metrics.map((row) => row.loss), 0.5));
  const xAt = (index: number) => {
    if (metrics.length <= 1) return pad;
    return pad + (index / (metrics.length - 1)) * (width - pad * 2);
  };
  const yAt = (value: number) => pad + (1 - value / maxLoss) * (height - pad * 2);
  const path = $derived(
    metrics.map((row, index) => `${index === 0 ? 'M' : 'L'} ${xAt(index).toFixed(1)} ${yAt(row.loss).toFixed(1)}`).join(' ')
  );
  const baseline = $derived(yAt(CHANCE_NATS));
</script>

<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Training loss">
  <line x1={pad} y1={baseline} x2={width - pad} y2={baseline} stroke="#8d3f16" stroke-dasharray="4 4" />
  <text x={pad} y={baseline - 6} fill="#8d3f16" font-size="12">
    chance ln(8192) = {CHANCE_NATS.toFixed(2)}
  </text>
  {#if path}
    <path d={path} fill="none" stroke="#1e4642" stroke-width="2.5" />
  {/if}
  {#each metrics as row, index}
    <circle cx={xAt(index)} cy={yAt(row.loss)} r="3.5" fill={row.phase === 'finetune' ? '#c4622d' : '#1e4642'}>
      <title>{row.phase} step {row.step}: {row.loss.toFixed(3)}</title>
    </circle>
  {/each}
  {#if metrics.length === 0}
    <text x={width / 2} y={height / 2} text-anchor="middle" fill="#5c5146" font-size="14">
      Loss will plot here once a step finishes.
    </text>
  {/if}
</svg>
<p class="note">Pine points are pretraining. Copper points are finetuning. The dashed line is a model that picks the next byte uniformly.</p>
