<script lang="ts">
  import { onMount } from 'svelte';
  import { softmax } from '#lib/bytes.ts';
  import { lab } from '#lib/lab.svelte.ts';
  import Source from '#lib/components/Source.svelte';
  import { checkpointStory, type CheckpointFact } from '#lib/checkpointStory.ts';
  import { readStore, sanitizeSample, writeStoreSoon } from '#lib/persist.ts';

  const demo = [
    { glyph: 'c', logit: 2.4 },
    { glyph: '␣', logit: 1.1 },
    { glyph: 'b', logit: 0.4 },
    { glyph: 'd', logit: 0.2 },
    { glyph: 't', logit: -0.4 },
    { glyph: 'x', logit: -1.6 }
  ];

  let prompt = $state('The lighthouse lamp is');
  let maxNewTokens = $state(24);
  let topK = $state(0);
  let checkpoints = $state<CheckpointFact[]>([]);
  let trainTokens = $state<number | null>(null);
  let pending = $state(false);
  let error = $state('');
  let samples = $state<{ checkpoint: string; text: string; prompt: string; vocabSize?: number; error?: string }[]>([]);
  let remember = $state(false);

  const probs = $derived(distribution(lab.temperature));
  const greedy = $derived(distribution(0));

  function distribution(scale: number): number[] {
    const logits = demo.map((row) => row.logit);
    if (scale <= 0) {
      const best = logits.indexOf(Math.max(...logits));
      return logits.map((_, index) => (index === best ? 1 : 0));
    }
    return softmax(logits, scale);
  }

  onMount(async () => {
    const saved = sanitizeSample(await readStore('sample'));
    if (saved) {
      prompt = saved.prompt;
      maxNewTokens = saved.maxNewTokens;
      topK = saved.topK;
    }
    remember = true;
    const [files, data] = await Promise.all([
      fetch('/api/checkpoints').then((response) => response.json()),
      fetch('/api/data').then((response) => response.json())
    ]);
    checkpoints = files.checkpoints ?? [];
    trainTokens = data.prepare?.train_tokens ?? null;
  });

  $effect(() => {
    const snap = { prompt, maxNewTokens, topK };
    if (!remember) return;
    writeStoreSoon('sample', snap);
  });

  async function sampleAll() {
    error = '';
    if (!checkpoints.length) {
      error = 'No checkpoint yet. Pretrain or run the pipeline first.';
      return;
    }
    pending = true;
    samples = [];
    try {
      samples = await Promise.all(
        checkpoints.map(async (row) => {
          const response = await fetch('/api/sample', {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({
              checkpoint: row.path,
              prompt,
              temperature: lab.temperature,
              maxNewTokens,
              topK
            })
          });
          const body = await response.json();
          if (!response.ok) {
            return {
              checkpoint: row.path,
              prompt,
              text: '',
              error: String(body.error ?? 'Sample failed')
            };
          }
          return body as { checkpoint: string; text: string; prompt: string; vocabSize?: number };
        })
      );
    } catch (caught) {
      error = caught instanceof Error ? caught.message : 'Sample failed';
    } finally {
      pending = false;
    }
  }
</script>

<p class="kicker">06 · Sample</p>
<h1>Turn the last logits into a byte.</h1>
<p class="lede">
  The bars are a toy distribution, not the live model: six candidate next bytes and their logits. Temperature
  divides the logits before softmax. Zero is greedy, which always takes the largest logit. The sampler below uses
  the real checkpoint with the same controls.
</p>

<section class="card">
  <label class="field">Temperature {lab.temperature.toFixed(2)}
    <input type="range" min="0" max="1.6" step="0.05" bind:value={lab.temperature} />
  </label>
  <div class="grid">
    <div>
      <h3>{lab.temperature === 0 ? 'Greedy' : 'Softmax'}</h3>
      <div class="bars">
        {#each demo as row, index}
          <div class="bar-row">
            <span>{row.glyph}</span>
            <div class="bar-track"><span style="width: {Math.max(2, probs[index] * 100)}%"></span></div>
            <span>{(probs[index] * 100).toFixed(1)}%</span>
          </div>
        {/each}
      </div>
    </div>
    <div>
      <h3>Same logits, temperature 0</h3>
      <div class="bars">
        {#each demo as row, index}
          <div class="bar-row">
            <span>{row.glyph}</span>
            <div class="bar-track"><span style="width: {Math.max(2, greedy[index] * 100)}%; background:#1e4642"></span></div>
            <span>{greedy[index] === 1 ? '100%' : '0%'}</span>
          </div>
        {/each}
      </div>
    </div>
  </div>
</section>

<section class="card stack">
  <h2>Why the three files differ</h2>
  <p>
    Pretraining on the <a href="/train">Train</a> page draws windows from the whole book stream.
    A file is fluent only after those steps cover the stream and the loss falls near 1 nat.
    Finetuning uses a different loss: the harbor question is context, and only the answer tokens are trained.
    The sampler then encodes the prompt with the vocabulary stored in the file.
  </p>
  <Source
    file="tinyllm/tokenizer.py"
    code={`def encode_for_vocab(text: str, vocab_size: int) -> list[int]:
    ids = encode_bytes(text) if vocab_size <= 256 else encode(text)
    overflow = next((token_id for token_id in ids if token_id >= vocab_size), None)
    if overflow is not None:
        raise ValueError(
            f"This checkpoint has a vocabulary of {vocab_size}. "
            f"The prompt needs id {overflow}, which that table does not contain."
        )
    return ids`}
  />
  {#each checkpoints as row}
    <article class="probe">
      <strong>{row.path}</strong>
      <div>{checkpointStory(row, trainTokens)}</div>
    </article>
  {/each}
</section>

<section class="card stack">
  <h2>Checkpoint</h2>
  <label class="field">Prompt
    <textarea bind:value={prompt}></textarea>
  </label>
  <div class="row">
    <button class="btn ghost" onclick={() => (prompt = 'The lighthouse lamp is')}>Prose probe</button>
    <button class="btn ghost" onclick={() => (prompt = 'Question: What is the lighthouse lamp?\nAnswer:')}>Question probe</button>
    <label class="field">New tokens
      <input type="number" min="1" max="80" bind:value={maxNewTokens} />
    </label>
    <label class="field">Top-k (0 off)
      <input type="number" min="0" max="64" bind:value={topK} />
    </label>
    <button class="btn" onclick={sampleAll} disabled={pending}>{pending ? 'Sampling' : 'Sample every checkpoint'}</button>
  </div>
  {#if !checkpoints.length}
    <p class="note">No files in checkpoints/ yet.</p>
  {:else}
    <p class="note">{checkpoints.map((row) => row.path).join(', ')}</p>
  {/if}
  {#if error}<p class="warn">{error}</p>{/if}
  {#each samples as sample}
    <article class="probe">
      <strong>{sample.checkpoint}</strong>
      {#if sample.vocabSize === 256}
        <p class="note">This file uses the earlier 256-byte vocabulary, so the prompt is encoded as bytes.</p>
      {/if}
      {#if sample.error}
        <p class="warn">{sample.error}</p>
      {:else}
        <div><em>{sample.prompt}</em>{sample.text.slice(sample.prompt.length)}</div>
      {/if}
    </article>
  {/each}
</section>
