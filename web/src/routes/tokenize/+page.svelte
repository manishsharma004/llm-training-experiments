<script lang="ts">
  import { onMount } from 'svelte';
  import { encodeText } from '#lib/bytes.ts';
  import { ensureTokenizer, tokenizerState } from '#lib/tokenizer.svelte.ts';

  let text = $state('The Old Testament of the King James Version of the Bible');
  let block = $state(16);
  let cursor = $state(0);
  let corpus = $state<{ train_tokens?: number; val_tokens?: number; train_bytes?: number; bytes_per_token?: number; preview?: string; tokenizer?: { vocab_size?: number; merges?: number } } | null>(null);

  const tokens = $derived(tokenizerState.ready ? encodeText(text) : []);
  const maxStart = $derived(Math.max(0, tokens.length - 2));
  const start = $derived(Math.min(cursor, maxStart));
  const windowTokens = $derived(tokens.slice(start, start + block));

  onMount(async () => {
    await ensureTokenizer();
    const response = await fetch('/api/data');
    const body = await response.json();
    corpus = body.prepare ?? null;
    if (typeof corpus?.preview === 'string' && corpus.preview.trim()) text = corpus.preview.slice(0, 280);
  });

  function step(direction: number) {
    cursor = Math.min(maxStart, Math.max(0, start + direction));
  }
</script>

<p class="kicker">05 · Tokens</p>
<h1>A vocabulary trained on the books.</h1>
<p class="lede">
  Byte-level BPE starts from all 256 UTF-8 bytes and merges the pairs that the cleaned books use most often, up to
  {tokenizerState.vocabSize.toLocaleString()} pieces ({tokenizerState.merges.toLocaleString()} merges). Any text still round-trips.
  The training file is these ids, not the raw bytes.
</p>

{#if corpus}
  <div class="stats">
    <div class="stat"><span class="note">vocabulary</span><strong>{(corpus.tokenizer?.vocab_size ?? tokenizerState.vocabSize).toLocaleString()}</strong></div>
    <div class="stat"><span class="note">train tokens</span><strong>{Number(corpus.train_tokens ?? 0).toLocaleString()}</strong></div>
    <div class="stat"><span class="note">validation tokens</span><strong>{Number(corpus.val_tokens ?? 0).toLocaleString()}</strong></div>
    <div class="stat"><span class="note">bytes / token</span><strong>{corpus.bytes_per_token ?? '—'}</strong></div>
  </div>
{/if}

<label class="field">Text from the book stream, or anything you type
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
  <p class="note">{tokenizerState.ready ? `${tokens.length} tokens` : 'Loading the merge table…'} · window starts at index {start}.</p>
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
