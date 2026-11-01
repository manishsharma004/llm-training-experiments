<script lang="ts">
  let { heads = 4, width = 128 }: { heads?: number; width?: number } = $props();

  const tokens = ['T', 'h', 'e', '␣', 'l', 'a', 'm', 'p'];
  const cell = 28;
  const originX = 132;
  const originY = 48;
  const headCount = $derived(Math.max(1, Math.round(Number(heads) || 1)));
  const embd = $derived(Math.max(1, Math.round(Number(width) || 1)));
  const headDim = $derived(embd % headCount === 0 ? embd / headCount : 0);

  function tone(row: number, column: number): string {
    if (column > row) return '#f6f1e8';
    const distance = row - column;
    const strength = 1 - distance / tokens.length;
    const alpha = 0.22 + strength * 0.78;
    return `rgba(196, 98, 45, ${alpha.toFixed(3)})`;
  }
</script>

<figure class="figure">
  <svg viewBox="0 0 760 390" role="img" aria-labelledby="attn-title attn-desc">
    <title id="attn-title">Causal attention on eight bytes of “The lamp”</title>
    <desc id="attn-desc">
      Each row is a query position. It may attend to its own column and to columns on the left. Cells above the diagonal are masked to negative infinity before the softmax. The same pattern is repeated for each head.
    </desc>
    <rect width="760" height="390" fill="#fffaf3" rx="16" />
    <text class="label" x="24" y="32">query ↓</text>
    <text class="label" x="132" y="32">key →</text>

    {#each tokens as token, column}
      <text class="glyph" x={originX + column * cell + 8} y="44">{token}</text>
    {/each}
    {#each tokens as token, row}
      <text class="glyph" x="96" y={originY + row * cell + 19}>{token}</text>
      {#each tokens as _, column}
        <rect
          x={originX + column * cell}
          y={originY + row * cell}
          width={cell - 3}
          height={cell - 3}
          rx="3"
          fill={tone(row, column)}
          stroke="#e3d6c4"
        />
      {/each}
    {/each}

    <text class="label" x="400" y="32">{headCount} heads · dim {headDim || '—'}</text>
    {#each Array(Math.min(headCount, 8)) as _, head}
      <g transform="translate({400 + (head % 4) * 86}, {48 + Math.floor(head / 4) * 150})">
        <text class="label" x="0" y="0">head {head}</text>
        {#each tokens as __, row}
          {#each tokens as ___, column}
            <rect
              x={column * 9}
              y={12 + row * 9}
              width="8"
              height="8"
              fill={column > row ? '#f6f1e8' : '#1e4642'}
              opacity={column > row ? 1 : 0.35 + 0.65 * (1 - (row - column) / tokens.length)}
            />
          {/each}
        {/each}
      </g>
    {/each}

    <text class="note" x="24" y="310">Q, K, V come from one matrix, {embd} → {embd * 3}, then split.</text>
    <text class="note" x="24" y="332">score = QKᵀ / √{headDim || '?'}. Masked cells are −∞, so softmax gives them weight 0.</text>
    <text class="note" x="24" y="354">The “p” row can see “The lam” and itself. It cannot see a byte that has not been read yet.</text>
    <text class="note" x="24" y="376">Output map {embd} → {embd}. With the packed map, attention is 4 × {embd}² per block.</text>
  </svg>
  <figcaption>
    This is the mask inside every block, drawn on the eight bytes of “The lamp”. The copper cells are legal attention. The blank cells above the diagonal are the future. The small grids are the same mask, one per head. Heads do not look at different positions; they keep separate {headDim || '—'}-wide slices of the same causal pattern.
  </figcaption>
</figure>

<style>
  .figure { margin: 8px 0 18px; }
  svg { width: 100%; height: auto; border: 1px solid var(--line); border-radius: 16px; }
  figcaption { color: var(--ink-soft); font-size: 0.95rem; margin-top: 8px; max-width: 78ch; }
  .label { font-family: var(--mono); font-size: 12px; fill: #8d3f16; }
  .glyph { font-family: var(--mono); font-size: 13px; fill: #241c16; }
  .note { font-family: var(--sans); font-size: 13px; fill: #5c5146; }
</style>
