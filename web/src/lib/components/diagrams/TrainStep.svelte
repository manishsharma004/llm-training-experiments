<script lang="ts">
  const parameters = 1_035_008;
  const bytes = {
    weights: parameters * 4,
    adam: parameters * 8,
    gradients: parameters * 4
  };
  const batch = 16;
  const block = 128;
  const embd = 128;
  const layers = 5;
  const heads = 4;
  const activations = batch * block * embd * 4 * layers * 8 + batch * heads * block * block * 4 * layers;
  const resident = bytes.weights + bytes.adam + bytes.gradients + activations;
  const vram = 4 * 1024 ** 3;

  function mb(value: number) {
    return `${(value / 1024 ** 2).toFixed(1)} MB`;
  }

  const slices = [
    ['Weights', bytes.weights, '#1e4642'],
    ['Gradients', bytes.gradients, '#c4622d'],
    ['Adam moments', bytes.adam, '#d89a55'],
    ['Activation bound', activations, '#8a8175']
  ] as const;
</script>

<figure class="figure">
  <svg viewBox="0 0 980 300" role="img" aria-labelledby="step-title step-desc">
    <title id="step-title">One AdamW step on a 128-byte window</title>
    <desc id="step-desc">
      A window of 128 bytes is paired with the bytes shifted one position left. The model writes 256 logits at each position. Cross-entropy compares them with the next byte. Gradients update the weights and the two Adam moments. On the 4GB card this working set is a small slice.
    </desc>
    <rect width="980" height="300" fill="#fffaf3" rx="16" />

    <g transform="translate(24 28)">
      <text class="kicker" x="0" y="0">1 · Window</text>
      {#each ['T', 'h', 'e', '␣', 'l'] as glyph, index}
        <g transform="translate({index * 34} 16)">
          <rect width="30" height="36" rx="5" fill="#d7e6e1" stroke="#1e4642" />
          <text class="glyph" x="8" y="24">{glyph}</text>
          <text class="tiny" x="0" y="56">x{index}</text>
        </g>
      {/each}
      <text class="tiny" x="0" y="96">input [16, 128]</text>
      <text class="tiny" x="0" y="114">label is x shifted +1</text>
    </g>

    <path d="M210 78 H250" stroke="#c4622d" stroke-width="2" marker-end="url(#step-arrow)" />

    <g transform="translate(262 28)">
      <text class="kicker" x="0" y="0">2 · Forward</text>
      {#each [0, 1, 2, 3, 4] as layer}
        <rect x="0" y={18 + layer * 16} width={150 - layer * 6} height="12" rx="3" fill="#c4622d" opacity={0.45 + layer * 0.1} />
      {/each}
      <text class="tiny" x="0" y="114">hidden [16, 128, 128]</text>
      <text class="tiny" x="0" y="132">five identical blocks</text>
    </g>

    <path d="M450 78 H490" stroke="#c4622d" stroke-width="2" marker-end="url(#step-arrow)" />

    <g transform="translate(502 28)">
      <text class="kicker" x="0" y="0">3 · Logits</text>
      {#each [0, 1, 2, 3, 4, 5, 6, 7] as index}
        <rect
          x={index * 16}
          y={70 - (index === 2 ? 48 : 12 + (index % 4) * 8)}
          width="12"
          height={index === 2 ? 48 : 12 + (index % 4) * 8}
          rx="2"
          fill={index === 2 ? '#1e4642' : '#e3d6c4'}
        />
      {/each}
      <text class="tiny" x="0" y="96">256 scores</text>
      <text class="tiny" x="0" y="114">one is the true next byte</text>
      <text class="tiny" x="0" y="132">loss = −log p(next)</text>
    </g>

    <path d="M680 90 H720" stroke="#c4622d" stroke-width="2" marker-end="url(#step-arrow)" />

    <g transform="translate(732 28)">
      <text class="kicker" x="0" y="0">4 · AdamW</text>
      <rect x="0" y="18" width="200" height="14" rx="3" fill="#1e4642" />
      <rect x="0" y="38" width="200" height="14" rx="3" fill="#c4622d" />
      <rect x="0" y="58" width="100" height="14" rx="3" fill="#d89a55" />
      <rect x="0" y="78" width="100" height="14" rx="3" fill="#8a8175" />
      <text class="tiny" x="0" y="114">weights, gradient,</text>
      <text class="tiny" x="0" y="132">two moments · clip 1.0</text>
    </g>

    <defs>
      <marker id="step-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
        <path d="M0 0 L6 3 L0 6 Z" fill="#c4622d" />
      </marker>
    </defs>
  </svg>

  <div class="card memory">
    <p class="kicker">Where the 4GB goes, for this preset</p>
    <div class="track" aria-hidden="true">
      <span style="width: {Math.max(0.8, (100 * resident) / vram)}%"></span>
    </div>
    <p class="note">
      The mark is the resident set, drawn at least a sliver wide so it stays visible. It is {mb(resident)} of {mb(vram)} ({((100 * resident) / vram).toFixed(1)}%). The card is not full. The 4GB limit is not what constrains batch 16 and context 128.
    </p>
    <div class="legend">
      {#each slices as [name, value, color]}
        <span><i class="swatch" style="background: {color}"></i>{name} {mb(value)}</span>
      {/each}
    </div>
  </div>
  <figcaption>
    Chance for a byte is ln(256) ≈ 5.545. The loss is the mean of −log p(next byte) over the window. Finetuning uses the same picture, except prompt positions contribute nothing to that mean. Adam keeps two moments per parameter, which is why the optimizer is twice the size of the weights.
  </figcaption>
</figure>

<style>
  .figure { margin: 8px 0 18px; }
  svg { width: 100%; height: auto; border: 1px solid var(--line); border-radius: 16px; display: block; }
  figcaption { color: var(--ink-soft); font-size: 0.95rem; margin-top: 8px; max-width: 78ch; }
  .kicker { font-family: var(--mono); font-size: 12px; fill: #8d3f16; }
  .glyph { font-family: var(--mono); font-size: 14px; fill: #241c16; }
  .tiny { font-family: var(--sans); font-size: 12px; fill: #5c5146; }
  .memory { margin-top: 12px; }
  .track { height: 18px; border-radius: 999px; background: #efe4d4; overflow: hidden; }
  .track span { display: block; height: 100%; background: #c4622d; }
</style>
