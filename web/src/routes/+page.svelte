<script lang="ts">
  import { breakdown, formatCount } from '#lib/params.ts';

  const report = breakdown();
  const stations = [
    ['/gather', '02 Gather', 'Read data/raw, or paste a document. Nothing is filtered yet.'],
    ['/clean', '03 Clean', 'Strip HTML, drop junk and exact duplicates, and keep the rejection reasons.'],
    ['/prepare', '04 Prepare', 'Hold out validation documents and pack the rest into byte windows.'],
    ['/train', '06 Train', 'Run the million-parameter preset on CPU, or on the RX 5500 when PyTorch can see it.'],
    ['/finetune', '07 Finetune', 'The question stays in context. Only the answer tokens enter the loss.'],
    ['/generate', '08 Sample', 'Read a checkpoint. Compare the prose probe with the question probe.']
  ];
</script>

<p class="kicker">A lab notebook, not a chatbot</p>
<h1>Train a million-parameter model you can read.</h1>
<p class="lede">
  The path is the one a training run actually follows: gather documents, clean them with reasons you can read,
  pack the bytes into a train and validation split, then train a 1,035,008-parameter model. On an RX 5500 the same
  loop runs on the GPU when ROCm exposes the card. Every stage leaves a report, so the loss curve is not the only
  thing you can inspect.
</p>

<div class="grid">
  <article class="card">
    <p class="kicker">Default model</p>
    <h2>{formatCount(report.total)}</h2>
    <p class="note">parameters. Five blocks, width 128, four heads, context 128, byte vocabulary. The output head is tied to the token embedding, so it is not counted twice.</p>
  </article>
  <article class="card">
    <p class="kicker">What “learning” means here</p>
    <p>Pretraining copies the prose style of the harbor. Finetuning is the same network on a new loss: questions that never appear in that prose. If the samples start answering <em>Question:</em>, the mask did its job.</p>
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

<section class="card">
  <h2>The order inside one update</h2>
  <ol>
    <li>Take a window of bytes. The label vector is that window shifted one step left.</li>
    <li>Embed tokens and positions, run the residual blocks, project to 256 logits.</li>
    <li>Cross-entropy against the labels. Finetuning sets prompt labels to <span class="formula" style="display:inline;padding:2px 6px">-100</span>, which PyTorch ignores.</li>
    <li>AdamW step with linear warmup and cosine decay, gradients clipped at 1.</li>
    <li>Every few steps, greedy-decode four fixed probes so you can read the weights, not just the loss.</li>
  </ol>
</section>
