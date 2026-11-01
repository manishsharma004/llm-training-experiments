<script lang="ts">
  import { onMount } from 'svelte';
  import RunBench from '#lib/components/RunBench.svelte';
  import { lab } from '#lib/lab.svelte.ts';
  import { encodeBytes } from '#lib/bytes.ts';
  import type { Harbor } from '#lib/types.ts';

  let harbor = $state<Harbor | null>(null);
  let index = $state(0);

  onMount(async () => {
    const response = await fetch('/api/harbor');
    harbor = await response.json();
    if (harbor && harbor.pairs[0]) {
      lab.prompt = harbor.pairs[0].prompt;
      lab.completion = harbor.pairs[0].completion;
    }
  });

  const promptTokens = $derived(encodeBytes(lab.prompt));
  const completionTokens = $derived(encodeBytes(lab.completion));
  const promptLen = $derived(promptTokens.length);

  function choose(next: number) {
    if (!harbor) return;
    index = next;
    lab.prompt = harbor.pairs[next].prompt;
    lab.completion = harbor.pairs[next].completion;
  }
</script>

<p class="kicker">07 · Finetune</p>
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
        <span class="chip supervised" title="loss when this byte is predicted"><b>{token.glyph}</b>{token.id}</span>
      {/each}
    </div>
    <p class="note">
      The first supervised label is at input index {Math.max(0, promptLen - 1)}. That position still sees the prompt and is asked to predict the first answer byte.
      Prompt length {promptLen}, answer length {completionTokens.length}.
    </p>
  </section>
</div>

<h2>Continue from the pretrained checkpoint</h2>
<p class="note">Finetune reads <span class="formula" style="display:inline;padding:2px 6px">checkpoints/pretrain.pt</span> and writes <span class="formula" style="display:inline;padding:2px 6px">checkpoints/sft.pt</span>. The learning rate is lower than pretraining so the new format can land without washing out the prose.</p>
<RunBench phase="finetune" finetuneSteps={80} learningRate={0.001} />

<h2>Or run the whole pipeline</h2>
<p class="note">Pretrain, then finetune, into the same checkpoint paths. The chart keeps both phases: pine, then copper. If the learning rate is above 1e-3, the finetune half of the pipeline lowers it to 1e-3. About 200 pretrain steps is where the prose probe starts finishing “copper”; the question probe should stay messy until the finetune half. Finetuning can blur the prose. That tradeoff is the point of comparing the two checkpoints on the sample page.</p>
<RunBench phase="pipeline" steps={200} finetuneSteps={80} learningRate={0.003} />
