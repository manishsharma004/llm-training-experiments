<script lang="ts">
  import { encodeBytes } from '#lib/bytes.ts';

  let { prompt, completion }: { prompt: string; completion: string } = $props();

  const promptTokens = $derived(encodeBytes(prompt));
  const completionTokens = $derived(encodeBytes(completion));
  const ids = $derived([...promptTokens, ...completionTokens]);
  const promptLen = $derived(promptTokens.length);
  const cells = $derived.by(() => {
    if (ids.length < 2) return [];
    const start = Math.max(0, promptLen - 8);
    const end = Math.min(ids.length - 1, Math.max(promptLen + 5, start + 1));
    const rows = [];
    for (let index = start; index < end; index += 1) {
      const predicts = index + 1;
      const supervised = predicts >= promptLen;
      rows.push({
        glyph: ids[index].glyph,
        label: supervised ? String(ids[predicts].id) : '−100',
        supervised,
        inputKind: index < promptLen ? 'prompt' : 'answer'
      });
    }
    return rows;
  });
  const omitted = $derived(Math.max(0, promptLen - 8));
</script>

<figure class="figure">
  <svg viewBox="0 0 980 250" role="img" aria-labelledby="mask-title mask-desc">
    <title id="mask-title">Which positions produce a gradient</title>
    <desc id="mask-desc">
      Input tokens include the question and the answer. The label at each position is the next token. Labels that still point at the question are −100 and are ignored by the loss. Labels that point at answer bytes are supervised.
    </desc>
    <rect width="980" height="250" fill="#fffaf3" rx="16" />
    <text class="label" x="20" y="36">input</text>
    <text class="label" x="20" y="118">label</text>
    <text class="label" x="20" y="176">gradient</text>

    {#each cells as cell, index}
      <g transform="translate({90 + index * 44} 48)">
        <rect width="40" height="32" rx="5" fill={cell.inputKind === 'prompt' ? '#d7e6e1' : '#f6d3bc'} stroke={cell.inputKind === 'prompt' ? '#1e4642' : '#c4622d'} />
        <text class="glyph" x="20" y="21" text-anchor="middle">{cell.glyph}</text>
      </g>
      <g transform="translate({90 + index * 44} 128)">
        <rect width="40" height="28" rx="5" fill={cell.supervised ? '#c4622d' : '#f6f1e8'} stroke={cell.supervised ? '#c4622d' : '#e3d6c4'} />
        <text class={cell.supervised ? 'on' : 'tiny'} x="20" y="18" text-anchor="middle">{cell.label}</text>
      </g>
      <text class={cell.supervised ? 'flow' : 'dash'} x={110 + index * 44} y="196" text-anchor="middle">{cell.supervised ? '↓' : '·'}</text>
    {/each}
  </svg>
  <figcaption>
    The first row is what the block can read. The second row is the target at that position: token i+1. Pine inputs are the question. Their targets stay −100, except the last question byte, which is asked to predict the first answer byte. That label is copper, and it is the first arrow back into Adam. Answer bytes after it are supervised the same way.
    {#if omitted}
      {omitted} earlier question bytes are off the left edge. Their labels are −100 as well.
    {/if}
  </figcaption>
</figure>

<style>
  .figure { margin: 8px 0 18px; }
  svg { width: 100%; height: auto; border: 1px solid var(--line); border-radius: 16px; display: block; }
  figcaption { color: var(--ink-soft); font-size: 0.95rem; margin-top: 8px; max-width: 78ch; }
  .label { font-family: var(--mono); font-size: 12px; fill: #8d3f16; }
  .glyph { font-family: var(--mono); font-size: 13px; fill: #241c16; }
  .tiny { font-family: var(--mono); font-size: 9px; fill: #8a8175; }
  .on { font-family: var(--mono); font-size: 10px; fill: #f6f3ec; }
  .dash { font-size: 18px; fill: #c8bbaa; }
  .flow { font-size: 16px; fill: #c4622d; }
</style>
