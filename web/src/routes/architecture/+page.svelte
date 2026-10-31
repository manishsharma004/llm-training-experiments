<script lang="ts">
  import { breakdown, defaultShape, formatCount, type ModelShape } from '#lib/params.ts';

  let shape = $state<ModelShape>({ ...defaultShape });
  const report = $derived(breakdown(shape));
  const parts = $derived([
    ['Token embedding', report.tokenEmbedding, '#1e4642'],
    ['Position embedding', report.positionEmbedding, '#3f7a74'],
    ['Attention', report.attention, '#c4622d'],
    ['MLP', report.mlp, '#d89a55'],
    ['Layer norm', report.layerNorm, '#8a8175']
  ] as const);
  const headDim = $derived(shape.nHead > 0 ? shape.nEmbd / shape.nHead : 0);

  function reset() {
    shape = { ...defaultShape };
  }
</script>

<p class="kicker">02 · Shape</p>
<h1>Where a million parameters go.</h1>
<p class="lede">
  Every number below is a weight matrix the trainer actually allocates. Linear layers have no bias. Each LayerNorm
  has a weight and a bias. The language-model head reuses the token embedding, so its extra cost is zero.
</p>

<div class="grid">
  <section class="card">
    <div class="row">
      <label class="field">Layers {shape.nLayer}
        <input type="range" min="1" max="12" bind:value={shape.nLayer} />
      </label>
      <label class="field">Width {shape.nEmbd}
        <input type="range" min="32" max="256" step="32" bind:value={shape.nEmbd} />
      </label>
      <label class="field">Heads {shape.nHead}
        <input type="range" min="1" max="8" bind:value={shape.nHead} />
      </label>
      <label class="field">Context {shape.blockSize}
        <input type="range" min="32" max="256" step="32" bind:value={shape.blockSize} />
      </label>
      <label class="field">MLP ratio {shape.mlpRatio}
        <input type="range" min="2" max="4" bind:value={shape.mlpRatio} />
      </label>
      <button class="btn ghost" onclick={reset}>Reset to the preset</button>
    </div>
    <p class="note">Vocabulary is fixed at 256 because the tokenizer is raw bytes. Head dimension is {headDim}.</p>
    {#if !report.valid}<p class="warn">{report.reason}</p>{/if}
  </section>
  <section class="card">
    <p class="kicker">Total</p>
    <h2>{formatCount(report.total)}</h2>
    <p class="note">
      {#if report.total >= 800_000 && report.total <= 1_200_000}
        Inside the one-million band (0.8M–1.2M).
      {:else}
        Outside the one-million band. The training preset is {formatCount(breakdown().total)}.
      {/if}
    </p>
    <div class="meter" aria-hidden="true">
      {#each parts as [, value, color]}
        <i style="width: {(100 * value) / report.total}%; background: {color}"></i>
      {/each}
    </div>
    <div class="legend">
      {#each parts as [name, value, color]}
        <span><i class="swatch" style="background: {color}"></i>{name} {formatCount(value)}</span>
      {/each}
    </div>
  </section>
</div>

<section class="card stack">
  <h2>One residual block, repeated {shape.nLayer} times</h2>
  <div class="block">
    <strong>bytes → token embedding + position embedding</strong>
    <ol>
      <li>LayerNorm, then causal self-attention (query, key, value, and the output map: 4 × width²).</li>
      <li>Add the residual, so the block can leave its input unchanged.</li>
      <li>LayerNorm, then an MLP that widens to {shape.mlpRatio}× and comes back with GELU.</li>
      <li>Add the residual again.</li>
    </ol>
  </div>
  <p>After the stack, a final LayerNorm and the tied head produce one logit per byte. Per block this preset spends {formatCount(report.perBlock)} parameters.</p>
  <table>
    <thead><tr><th>Group</th><th>Expression</th><th>Count</th></tr></thead>
    <tbody>
      <tr><td>Token embedding</td><td>vocab × width</td><td>{formatCount(report.tokenEmbedding)}</td></tr>
      <tr><td>Positions</td><td>context × width</td><td>{formatCount(report.positionEmbedding)}</td></tr>
      <tr><td>Attention</td><td>layers × 4 × width²</td><td>{formatCount(report.attention)}</td></tr>
      <tr><td>MLP</td><td>layers × 2 × width × (ratio × width)</td><td>{formatCount(report.mlp)}</td></tr>
      <tr><td>Layer norm</td><td>layers × 4 × width + 2 × width</td><td>{formatCount(report.layerNorm)}</td></tr>
      <tr><td>Tied head</td><td>shared with the token embedding</td><td>0</td></tr>
    </tbody>
  </table>
</section>
