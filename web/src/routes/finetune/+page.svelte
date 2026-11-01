<script lang="ts">
  import { onMount } from 'svelte';
  import RunBench from '#lib/components/RunBench.svelte';
  import { lab } from '#lib/lab.svelte.ts';
  import { encodeText } from '#lib/bytes.ts';
  import Source from '#lib/components/Source.svelte';
  import { ensureTokenizer, tokenizerState } from '#lib/tokenizer.svelte.ts';
  import type { Harbor } from '#lib/types.ts';

  let harbor = $state<Harbor | null>(null);
  let index = $state(0);

  onMount(async () => {
    await ensureTokenizer();
    const response = await fetch('/api/harbor');
    harbor = await response.json();
    if (harbor && harbor.pairs[0]) {
      lab.prompt = harbor.pairs[0].prompt;
      lab.completion = harbor.pairs[0].completion;
    }
  });

  const promptTokens = $derived(tokenizerState.ready ? encodeText(lab.prompt) : []);
  const completionTokens = $derived(tokenizerState.ready ? encodeText(lab.completion) : []);
  const promptLen = $derived(promptTokens.length);

  function choose(next: number) {
    if (!harbor) return;
    index = next;
    lab.prompt = harbor.pairs[next].prompt;
    lab.completion = harbor.pairs[next].completion;
  }
</script>

<p class="kicker">08 · Finetune</p>
<h1>Same model. Loss only on the answer.</h1>
<p class="lede">
  Supervised finetuning does not swap the architecture. It changes which tokens are allowed to produce a gradient.
  The prompt is context. Labels that would ask the model to copy the question are set to −100 and ignored.
  Copper volumes on the stage are the positions the loss sees. Drag along them.
</p>

<div class="grid">
  <section class="card">
    <label class="field">Prompt
      <textarea bind:value={lab.prompt}></textarea>
    </label>
    <label class="field">Completion
      <textarea bind:value={lab.completion}></textarea>
    </label>
    {#if harbor}
      <div class="row">
        {#each harbor.pairs as pair, pairIndex}
          <button class="btn ghost" onclick={() => choose(pairIndex)} disabled={pairIndex === index}>
            {pairIndex + 1}
          </button>
        {/each}
      </div>
    {/if}
  </section>
  <section class="card">
    <h3>Who is supervised?</h3>
    <div class="chips">
      {#each promptTokens as token}
        <span class="chip context" title="context, no loss"><b>{token.glyph}</b>{token.id}</span>
      {/each}
      {#each completionTokens as token}
        <span class="chip supervised" title="loss when this token is predicted"><b>{token.glyph}</b>{token.id}</span>
      {/each}
    </div>
    <p class="note">
      The first supervised label is at input index {Math.max(0, promptLen - 1)}. That position still sees the prompt and is asked to predict the first answer token.
      Prompt length {promptLen}, answer length {completionTokens.length}.
    </p>
  </section>
</div>

<h2>The same forward, a different label</h2>
<p>
  The book pretrain on the <a href="/train">Train</a> page predicts every next token.
  Finetune keeps that network and that checkpoint, then sets the question’s labels to −100.
  The last prompt token stays supervised, because it is the position that predicts the first answer token.
  The harbor notes are the only pairs in this lab. A loss near zero means those answers were memorized, which is why a prose prompt on the <a href="/generate">Sample</a> page comes back as fragments of them.
</p>
<Source
  file="tinyllm/data.py"
  code={`for index in range(length):
    predicts = index + 1
    if predicts >= prompt_len:
        labels[row, index] = target[index]`}
/>
<Source
  file="tinyllm/model.py"
  code={`loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    labels.view(-1),
    ignore_index=-100,
)`}
/>

<h2>Continue from the pretrained checkpoint</h2>
<p class="note">Finetune reads <span class="formula" style="display:inline;padding:2px 6px">checkpoints/pretrain.pt</span> and writes <span class="formula" style="display:inline;padding:2px 6px">checkpoints/sft.pt</span>. The learning rate is lower than pretraining so the new answers can land. The step count and loss of the file on disk are on the <a href="/train">Train</a> page and again beside the sample.</p>
<RunBench phase="finetune" finetuneSteps={8} learningRate={0.001} />

<h2>Or run the whole pipeline</h2>
<p class="note">Pretrain, then finetune, into the same checkpoint paths. The chart keeps both phases: pine, then copper. If the learning rate is above 1e-3, the finetune half lowers it to 1e-3. The harbor prose is a few repeated sentences, so a couple of hundred steps can memorize “copper” there. The book stream on the Train page is millions of tokens, and the same step count does not cover it.</p>
<RunBench phase="pipeline" steps={20} finetuneSteps={8} learningRate={0.003} />
