<script lang="ts">
  let {
    layers = 5,
    width = 128,
    heads = 4,
    context = 128,
    mlpRatio = 4
  }: {
    layers?: number;
    width?: number;
    heads?: number;
    context?: number;
    mlpRatio?: number;
  } = $props();

  let yaw = $state(-28);
  let pitch = $state(58);
  let selected = $state(2);
  let drag: { x: number; y: number; yaw: number; pitch: number } | null = null;

  const layerCount = $derived(Math.max(1, Math.round(Number(layers) || 1)));
  const embd = $derived(Math.max(1, Math.round(Number(width) || 1)));
  const headCount = $derived(Math.max(1, Math.round(Number(heads) || 1)));
  const block = $derived(Math.max(1, Math.round(Number(context) || 1)));
  const ratio = $derived(Math.max(1, Math.round(Number(mlpRatio) || 1)));
  const headDim = $derived(embd % headCount === 0 ? embd / headCount : 0);
  const plates = $derived.by(() => {
    const rows: { key: string; label: string; tone: 'sand' | 'pine' | 'copper'; body: string }[] = [
      {
        key: 'bytes',
        label: 'Bytes',
        tone: 'sand',
        body: `The batch is integer ids, shape [batch, ${block}]. Each id is one UTF-8 byte, so the vocabulary is 256. Position 0 of a window might be the byte for “T”.`
      },
      {
        key: 'embed',
        label: 'Embed',
        tone: 'pine',
        body: `token_embedding[id] is a vector of length ${embd} (${256 * embd} parameters). position_embedding[t] is another (${block * embd}). They are added. The token matrix is the same storage the output head will multiply with, so the head adds no parameters.`
      }
    ];
    for (let index = 0; index < layerCount; index += 1) {
      rows.push({
        key: `block-${index}`,
        label: `Block ${index + 1}`,
        tone: 'copper',
        body: headDim
          ? `Pre-norm residual block. Attention packs query, key, and value into one bias-free map ${embd} → ${3 * embd}, then an output map ${embd} → ${embd}. Those two maps are 4 × ${embd}² = ${4 * embd * embd} parameters. Scores are QKᵀ / √${headDim}, across ${headCount} heads of dimension ${headDim}. A lower triangle keeps position t from seeing the future. The MLP maps ${embd} → ${embd * ratio}, GELU, then back. Each sublayer is added onto the residual stream it read.`
          : `Width ${embd} does not divide into ${headCount} heads, so this shape is not a model the trainer will build.`
      });
    }
    rows.push({
      key: 'head',
      label: 'Tied head',
      tone: 'pine',
      body: `A final LayerNorm, then logits = hidden × token_embeddingᵀ. That is 256 scores at every position. Cross-entropy compares the score vector with the next byte. The matrix is not a second copy of the embedding.`
    });
    return rows;
  });

  $effect(() => {
    if (selected > plates.length - 1) selected = plates.length - 1;
  });

  const active = $derived(plates[Math.min(selected, plates.length - 1)]);

  function pointerDown(event: PointerEvent) {
    const node = event.currentTarget as HTMLElement;
    node.setPointerCapture(event.pointerId);
    drag = { x: event.clientX, y: event.clientY, yaw, pitch };
  }

  function pointerMove(event: PointerEvent) {
    if (!drag) return;
    yaw = drag.yaw + (event.clientX - drag.x) * 0.45;
    pitch = Math.min(78, Math.max(28, drag.pitch - (event.clientY - drag.y) * 0.25));
  }

  function pointerUp() {
    drag = null;
  }
</script>

<figure class="figure">
  <div class="layout">
    <div
      class="stage"
      role="application"
      aria-label="Rotatable stack of the model. Drag to turn it."
      style="--yaw: {yaw}deg; --pitch: {pitch}deg"
      onpointerdown={pointerDown}
      onpointermove={pointerMove}
      onpointerup={pointerUp}
      onpointercancel={pointerUp}
    >
      <div class="rig" style="--count: {plates.length}">
        {#each plates as plate, index (plate.key)}
          <button
            type="button"
            class="plate {plate.tone}"
            class:selected={index === selected}
            style="--i: {index}; --n: {plates.length}"
            aria-pressed={index === selected}
            onclick={() => (selected = index)}
          >
            <span class="skin"></span>
            <span class="top">{plate.label}</span>
          </button>
        {/each}
      </div>
      <p class="hint">Drag to turn. Select a plate.</p>
    </div>
    <div class="readout">
      <p class="kicker">{active?.label}</p>
      <p>{active?.body}</p>
      <div class="row">
        <button type="button" class="btn ghost" onclick={() => (yaw -= 20)}>Turn left</button>
        <button type="button" class="btn ghost" onclick={() => (yaw += 20)}>Turn right</button>
      </div>
      {#if active?.key.startsWith('block')}
        <ol class="cutaway">
          <li>LayerNorm</li>
          <li>Causal attention, then add the residual</li>
          <li>LayerNorm</li>
          <li>MLP, then add the residual</li>
        </ol>
      {/if}
    </div>
  </div>
  <figcaption>
    Data enters at the bottom plate and leaves at the top. The copper plates are copies of one block. Turning the stack shows depth: each plate reads a tensor of shape [batch, {block}, {embd}] and writes a tensor of that same shape.
  </figcaption>
</figure>

<style>
  .figure { margin: 8px 0 18px; }
  figcaption { color: var(--ink-soft); font-size: 0.95rem; margin-top: 8px; max-width: 78ch; }
  .layout { display: grid; grid-template-columns: minmax(280px, 1.2fr) minmax(240px, 0.8fr); gap: 14px; align-items: stretch; }
  .stage {
    position: relative;
    min-height: 460px;
    border: 1px solid var(--line);
    border-radius: 16px;
    background:
      radial-gradient(420px 220px at 50% 42%, rgba(196, 98, 45, 0.12), transparent 70%),
      #fffaf3;
    perspective: 1100px;
    overflow: hidden;
    touch-action: none;
    cursor: grab;
  }
  .stage:active { cursor: grabbing; }
  .rig {
    position: absolute;
    left: 50%;
    top: 54%;
    width: 250px;
    height: 132px;
    transform-style: preserve-3d;
    transform: translate(-50%, -50%) rotateX(var(--pitch)) rotateZ(var(--yaw));
  }
  .plate {
    position: absolute;
    inset: 0;
    border: 0;
    padding: 0;
    background: transparent;
    transform-style: preserve-3d;
    transform: translateZ(calc((var(--i) - (var(--n) - 1) / 2) * 36px));
    cursor: pointer;
  }
  .skin, .top {
    position: absolute;
    inset: 0;
    border-radius: 12px;
    border: 1px solid rgba(36, 28, 22, 0.35);
  }
  .skin { transform: translateZ(-12px); filter: brightness(0.72); }
  .top {
    display: grid;
    place-items: center;
    font-family: var(--mono);
    font-size: 0.82rem;
    letter-spacing: 0.04em;
    color: #f6f3ec;
    transform: translateZ(12px);
  }
  .sand .skin, .sand .top { background: #d89a55; color: #241c16; }
  .pine .skin, .pine .top { background: #1e4642; }
  .copper .skin, .copper .top { background: #c4622d; }
  .plate.selected { transform: translateZ(calc((var(--i) - (var(--n) - 1) / 2) * 36px + 22px)); }
  .plate.selected .top { outline: 2px solid #241c16; }
  .hint {
    position: absolute;
    left: 12px;
    bottom: 10px;
    margin: 0;
    font-size: 0.78rem;
    color: var(--ink-soft);
  }
  .readout {
    border: 1px solid var(--line);
    border-radius: 16px;
    background: var(--card);
    padding: 16px 18px;
  }
  .cutaway { margin: 8px 0 0; padding-left: 18px; }
  @media (max-width: 860px) {
    .layout { grid-template-columns: 1fr; }
    .stage { min-height: 380px; }
  }
</style>
