<script lang="ts">
  import { breakdown, formatCount } from '#lib/params.ts';

  const report = breakdown();
  const stations = [
    ['/architecture', '02 Shape', 'Count every matrix. The default preset is the model the trainer builds.'],
    ['/tokenize', '03 Bytes', 'See a sentence become ids, then a sliding window whose target is shifted by one.'],
    ['/pretrain', '04 Pretrain', 'Next-byte prediction on the harbor prose. Chance is ln(256).'],
    ['/finetune', '05 Finetune', 'The question stays in context. Only the answer tokens enter the loss.'],
    ['/generate', '06 Sample', 'Temperature changes a distribution. A checkpoint turns a prompt into bytes.']
  ];
</script>

<p class="kicker">A lab notebook, not a chatbot</p>
<h1>Train a million-parameter model you can read.</h1>
<p class="lede">
  This repository is a full loop: a decoder-only Transformer, a pretraining corpus, a supervised finetune with a
  prompt mask, and this lab. The harbor it learns is invented and small, so a CPU run can memorize it and you can
  see the objective change the samples.
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
