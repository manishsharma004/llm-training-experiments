<script lang="ts">
  import { encodeBytes } from '#lib/bytes.ts';

  let text = $state('The lighthouse lamp is copper.');
  let block = $state(16);
  let cursor = $state(0);
  const tokens = $derived(encodeBytes(text));
  const maxStart = $derived(Math.max(0, tokens.length - 2));
  const start = $derived(Math.min(cursor, maxStart));
  const windowTokens = $derived(tokens.slice(start, start + block));

  function step(direction: number) {
    cursor = Math.min(maxStart, Math.max(0, start + direction));
  }
</script>

<p class="kicker">03 · Bytes</p>
<h1>A window, and the token that follows it.</h1>
<p class="lede">
  Pretraining never sees words as units. It sees UTF-8 bytes. The model reads the highlighted window and is scored
  on the next byte after each position: the target row is the input row, shifted left by one.
</p>

<label class="field">Text
  <textarea bind:value={text}></textarea>
</label>

<div class="card">
  <div class="row">
    <label class="field">Context length {block}
      <input type="range" min="4" max="48" bind:value={block} />
    </label>
    <button class="btn ghost" onclick={() => step(-1)} disabled={start === 0}>Previous window</button>
    <button class="btn ghost" onclick={() => step(1)} disabled={start >= maxStart}>Next window</button>
  </div>
  <p class="note">{tokens.length} bytes. Window starts at index {start}.</p>
  <div class="chips" aria-label="All tokens">
    {#each tokens as token}
      <span class="chip" class:dim={token.index < start || token.index >= start + block} class:target={token.index >= start && token.index < start + block}>
        <b>{token.glyph}</b>{token.id}
      </span>
    {/each}
  </div>
</div>

<div class="grid">
  <section class="card">
    <h3>Input</h3>
    <div class="chips">
      {#each windowTokens as token}
        <span class="chip"><b>{token.glyph}</b>{token.id}</span>
      {/each}
    </div>
  </section>
  <section class="card">
    <h3>Target</h3>
    <div class="chips">
      {#each tokens.slice(start + 1, start + 1 + block) as token}
        <span class="chip supervised"><b>{token.glyph}</b>{token.id}</span>
      {/each}
    </div>
    <p class="note">Position i of the input is trained to predict position i+1. The last target can sit just outside the input window.</p>
  </section>
</div>
