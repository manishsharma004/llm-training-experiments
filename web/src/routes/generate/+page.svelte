<script lang="ts">
  import { onMount } from 'svelte';
  import { lab } from '#lib/lab.svelte.ts';
  import { live } from '#lib/live.svelte.ts';
  import { rankNext } from '#lib/distribution.ts';
  import Source from '#lib/components/Source.svelte';
  import { checkpointStory, type CheckpointFact } from '#lib/checkpointStory.ts';
  import { readStore, sanitizeSample, writeStoreSoon } from '#lib/persist.ts';

  let prompt = $state('The lighthouse lamp is');
  let maxNewTokens = $state(24);
  let topK = $state(0);
  let checkpoint = $state('');
  let checkpoints = $state<CheckpointFact[]>([]);
  let trainTokens = $state<number | null>(null);
  let pending = $state(false);
  let error = $state('');
  let samples = $state<{ checkpoint: string; text: string; prompt: string; vocabSize?: number; error?: string }[]>([]);
  let remember = $state(false);
  let logitTicket = 0;

  const ranked = $derived(rankNext(live.logits, lab.temperature, topK, 8));
  const greedy = $derived(rankNext(live.logits, 0, 0, 8));

  onMount(async () => {
    const saved = sanitizeSample(await readStore('sample'));
    if (saved) {
      prompt = saved.prompt;
      maxNewTokens = saved.maxNewTokens;
      topK = saved.topK;
      checkpoint = saved.checkpoint;
    }
    remember = true;
    const [files, data] = await Promise.all([
      fetch('/api/checkpoints').then((response) => response.json()),
      fetch('/api/data').then((response) => response.json())
    ]);
    checkpoints = files.checkpoints ?? [];
    trainTokens = data.prepare?.train_tokens ?? null;
    if (!checkpoint || !checkpoints.some((row) => row.path === checkpoint)) {
      checkpoint = checkpoints.find((row) => row.path === 'checkpoints/pretrain.pt')?.path ?? checkpoints[0]?.path ?? '';
    }
  });

  $effect(() => {
    const snap = { prompt, maxNewTokens, topK, checkpoint };
    if (!remember) return;
    writeStoreSoon('sample', snap);
  });

  $effect(() => {
    live.topK = topK;
  });

  $effect(() => {
    const chosen = checkpoint;
    const text = prompt;
    const previous = live.checkpoint;
    live.checkpoint = chosen;
    live.prompt = text;
    if (previous && previous !== chosen) live.logits = [];
    if (!chosen) return;
    const timer = setTimeout(() => {
      void pullLogits(chosen, text);
    }, 300);
    return () => clearTimeout(timer);
  });

  async function pullLogits(chosen: string, text: string) {
    const mine = ++logitTicket;
    live.loading = true;
    live.error = '';
    try {
      const response = await fetch('/api/logits', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ checkpoint: chosen, prompt: text })
      });
      const body = await response.json();
      if (mine !== logitTicket) return;
      if (!response.ok) {
        live.error = String(body.error ?? 'Could not read the logits.');
        live.logits = [];
        return;
      }
      live.logits = Array.isArray(body.logits) ? body.logits.map(Number) : [];
      live.vocabSize = Number(body.vocabSize ?? live.logits.length);
    } catch (caught) {
      if (mine !== logitTicket) return;
      live.error = caught instanceof Error ? caught.message : 'Could not read the logits.';
      live.logits = [];
    } finally {
      if (mine === logitTicket) live.loading = false;
    }
  }

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
<h1>Turn the last logits into the next piece.</h1>
<p class="lede">
  One forward pass of the selected checkpoint produces a logit for every vocabulary piece. Temperature divides
  those logits before softmax. Zero is greedy and takes the largest logit. The bars here and the 3D stage use that
  same vector. Heights on the stage are scaled to the tallest bar. The percentages are the model’s probabilities.
</p>

<section class="card">
  <div class="row">
    <label class="field">Checkpoint
      <select bind:value={checkpoint}>
        {#each checkpoints as row}
          <option value={row.path}>{row.path}</option>
        {/each}
      </select>
    </label>
    <label class="field">Temperature {lab.temperature.toFixed(2)}
      <input type="range" min="0" max="1.6" step="0.05" bind:value={lab.temperature} />
    </label>
  </div>
  {#if live.loading && !live.logits.length}
    <p class="note">Running {checkpoint || 'the checkpoint'} forward for the next-token logits.</p>
  {:else if live.error}
    <p class="warn">{live.error}</p>
  {:else if live.logits.length}
    <p class="note">
      {live.vocabSize.toLocaleString()} logits from {checkpoint}. The eight largest pieces are listed. The rest of the vocabulary is the remaining mass.
    </p>
  {/if}
  <div class="grid">
    <div>
      <h3>{lab.temperature === 0 ? 'Greedy' : 'Softmax'}</h3>
      <div class="bars">
        {#each ranked.pieces as row}
          <div class="bar-row">
            <span>{row.glyph}</span>
            <div class="bar-track"><span style="width: {Math.max(2, row.prob * 100)}%"></span></div>
            <span>{(row.prob * 100).toFixed(1)}%</span>
          </div>
        {/each}
        {#if ranked.rest > 0.005}
          <div class="bar-row">
            <span>rest</span>
            <div class="bar-track"><span style="width: {Math.max(2, ranked.rest * 100)}%"></span></div>
            <span>{(ranked.rest * 100).toFixed(1)}%</span>
          </div>
        {/if}
      </div>
    </div>
    <div>
      <h3>Same logits, temperature 0</h3>
      <div class="bars">
        {#each greedy.pieces as row}
          <div class="bar-row">
            <span>{row.glyph}</span>
            <div class="bar-track"><span style="width: 100%; background:#1e4642"></span></div>
            <span>100%</span>
          </div>
        {/each}
      </div>
    </div>
  </div>
</section>

<Source
  file="tinyllm/sample.py"
  code={`logits, _loss = model(tokens)
last = logits[0, -1]`}
/>

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
