<script lang="ts">
  import RunBench from '#lib/components/RunBench.svelte';
  import { CHANCE_NATS } from '#lib/params.ts';
</script>

<p class="kicker">04 · Pretrain</p>
<h1>Next token, every token.</h1>
<p class="lede">
  The loss is the average negative log probability of the true next token. A model that has learned nothing useful sits near
  {CHANCE_NATS.toFixed(3)} nats, because a uniform guess over 8192 pieces has that cross-entropy.
</p>

<section class="card">
  <p class="formula">loss = − mean log p(token t+1 | tokens up to t)</p>
  <p>Perplexity is exp(loss). At chance it equals 8192. If perplexity falls to 8, the model is about as sure as a uniform guess among 8 pieces.</p>
  <p class="note">The lab run uses the default million-parameter preset, AdamW, ten warmup steps, and cosine decay. Probes include “The lighthouse lamp is” and a question the prose has never contained.</p>
</section>

<h2>Run it</h2>
<RunBench phase="pretrain" steps={20} learningRate={0.003} />
