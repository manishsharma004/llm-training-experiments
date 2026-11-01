<script lang="ts">
  import { onMount } from 'svelte';
  import { breakdown, defaultShape, formatCount } from '#lib/params.ts';

  const report = breakdown();
  let data = $state<Record<string, unknown> | null>(null);

  onMount(async () => {
    const response = await fetch('/api/data');
    data = await response.json();
  });

  const prepare = $derived((data?.prepare ?? null) as { train_tokens?: number; val_tokens?: number; train_documents?: number; val_documents?: number; bytes_per_token?: number; train_bytes?: number } | null);
  const dataset = $derived((data?.dataset ?? null) as { kind?: string; work?: string; raw?: string } | null);
  const gather = $derived((data?.gather ?? null) as { documents?: number; bytes?: number } | null);

  const stations = [
    ['/gather', '02 Gather', 'The book files under data/corpus/raw. Nothing is filtered yet.'],
    ['/clean', '03 Clean', 'Strip HTML, drop junk and exact duplicates, and keep the rejection reasons.'],
    ['/prepare', '04 Prepare', 'Hold out validation books and encode the rest with the 8192-piece tokenizer.'],
    ['/tokenize', '05 Tokens', 'Inspect the byte-level BPE vocabulary trained on those books.'],
    ['/train', '07 Train', 'The finished book pass stays visible. A new run starts at 20 steps on the token stream.'],
    ['/finetune', '08 Finetune', 'The question stays in context. Only the answer tokens enter the loss.']
  ];
</script>

<p class="kicker">A lab notebook, not a chatbot</p>
<h1>Train on the books, with a vocabulary you can read.</h1>
<p class="lede">
  The pages follow the book corpus in <span class="formula" style="display:inline;padding:2px 6px">data/corpus</span>:
  gather, clean, then encode with a byte-level BPE of {formatCount(defaultShape.vocabSize)} pieces.
  The same five-block, width-128 network is now {formatCount(report.total)} parameters because the embedding grew with the vocabulary.
  The small harbor notes are only the finetune questions. The stage on the right reads the book stream.
</p>

<div class="grid">
  <article class="card">
    <p class="kicker">{dataset?.kind === 'corpus' ? 'Book corpus' : 'Waiting for the book stream'}</p>
    <h2>{prepare ? Number(prepare.train_tokens).toLocaleString() : '—'}</h2>
    <p class="note">
      training tokens
      {#if prepare?.val_tokens != null}
        and {Number(prepare.val_tokens).toLocaleString()} validation tokens
      {/if}
      {#if prepare?.bytes_per_token}
        · {prepare.bytes_per_token} bytes per token
      {/if}
      {#if gather?.documents}
        · {gather.documents.toLocaleString()} documents, {Number(gather.bytes).toLocaleString()} raw bytes
      {/if}
      {#if dataset?.work}
        · {dataset.work}
      {/if}
    </p>
  </article>
  <article class="card">
    <p class="kicker">Default model</p>
    <h2>{formatCount(report.total)}</h2>
    <p class="note">parameters. Five blocks, width 128, four heads, context 128, vocabulary 8192. The output head is tied to the token embedding, so it is not counted twice.</p>
  </article>
</div>

<div class="grid">
  {#each stations as [href, title, copy]}
    <a class="card stretch" {href}>
      <h3>{title}</h3>
      <p class="note">{copy}</p>
    </a>
  {/each}
</div>
