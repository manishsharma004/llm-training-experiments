<script lang="ts">
  import { onMount } from 'svelte';
  import RunBench from '#lib/components/RunBench.svelte';
  import Source from '#lib/components/Source.svelte';
  import { checkpointStory, type CheckpointFact } from '#lib/checkpointStory.ts';

  let hardware = $state<Record<string, unknown> | null>(null);
  let probeError = $state('');
  let trainTokens = $state<number | null>(null);
  let checkpoints = $state<CheckpointFact[]>([]);

  const perStep = 16 * 128;
  const passSteps = $derived(trainTokens == null ? null : Math.ceil(trainTokens / perStep));

  onMount(async () => {
    const [device, data, files] = await Promise.all([
      fetch('/api/device').then((response) => response.json()),
      fetch('/api/data').then((response) => response.json()),
      fetch('/api/checkpoints').then((response) => response.json())
    ]);
    if (device.error) probeError = device.error;
    else hardware = device;
    trainTokens = data.prepare?.train_tokens ?? null;
    checkpoints = files.checkpoints ?? [];
  });

  const budget = $derived((hardware?.budget as Record<string, number> | undefined) ?? null);
  const sensors = $derived(
    (hardware?.sensors ?? null) as {
      power_w?: number | null;
      vram_used_bytes?: number | null;
      temperature_c?: Record<string, number>;
    } | null
  );
  const guidance = $derived((hardware?.guidance as string[] | undefined) ?? []);

  function mb(bytes: number) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
</script>

<p class="kicker">07 · Train</p>
<h1>Train on the book token stream.</h1>
<p class="lede">
  Auto uses a GPU when this PyTorch process can see one, and the CPU otherwise. Choosing GPU on a machine
  where the RX 5500 is not visible stops the run and explains why, instead of silently training on the CPU.
  The preset stays in fp32. Pretraining draws windows from the whole book stream in <span class="formula" style="display:inline;padding:2px 6px">data/corpus/work/train.bin</span>.
  The validation books are scored and then discarded. They never enter <span class="formula" style="display:inline;padding:2px 6px">loss.backward()</span>.
</p>

<section class="card stack">
  <p class="formula">loss = − mean log p(token t+1 | tokens up to t)</p>
  <p>
    A step reads {perStep.toLocaleString()} tokens: batch 16, context 128.
    {#if trainTokens != null && passSteps != null}
      The stream holds {trainTokens.toLocaleString()} training tokens, so one pass is {passSteps.toLocaleString()} steps.
    {/if}
    The windows are random and can repeat. The step box below starts at 20, which is { (20 * perStep).toLocaleString() } tokens, so a click can show the loop without a pass over the books.
    A new run uses the 8,192-piece vocabulary and writes <span class="formula" style="display:inline;padding:2px 6px">checkpoints/pretrain.pt</span>.
  </p>
  <Source
    file="tinyllm/data.py"
    code={`starts = torch.randint(0, high + 1, (batch_size,))
inputs = torch.stack([stream[start : start + block_size] for start in starts])
labels = torch.stack([stream[start + 1 : start + 1 + block_size] for start in starts])`}
  />
  <Source
    file="tinyllm/model.py"
    code={`loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    labels.view(-1),
    ignore_index=-100,
)`}
  />
  <Source
    file="tinyllm/train.py"
    code={`inputs, labels = pretrain_batch(stream, model_config.block_size, train_config.batch_size, device)
_logits, loss = model(inputs, labels=labels)
loss.backward()
optimizer.step()`}
  />
  <Source
    file="tinyllm/train.py"
    code={`def _validation_loss(model, stream, block_size, batch_size, device):
    model.eval()
    with torch.no_grad():
        inputs, labels = pretrain_batch(stream, block_size, batch_size, device)
        _logits, loss = model(inputs, labels=labels)
    model.train()
    return float(loss.item())`}
  />
  {#if checkpoints.length}
    <h3>Files already on disk</h3>
    {#each checkpoints as row}
      <article class="probe">
        <strong>{row.path}</strong>
        <div>{checkpointStory(row, trainTokens)}</div>
      </article>
    {/each}
    <p class="note">Sampling these files is on the <a href="/generate">Sample</a> page. The same numbers are shown there.</p>
  {/if}
</section>

{#if probeError}<p class="warn">{probeError}</p>{/if}
{#if hardware}
  <section class="card">
    <p class="kicker">{hardware.torch_gpu ? hardware.backend : 'no GPU in this process'}</p>
    <h2>{hardware.name}</h2>
    <p class="note">
      {#if hardware.arch}Architecture {hardware.arch}. {/if}
      {#if hardware.hip}HIP {hardware.hip}. {/if}
      {#if hardware.total_vram_bytes}Visible memory {(Number(hardware.total_vram_bytes) / 1024 ** 3).toFixed(2)} GB.{/if}
    </p>
    {#if sensors}
      <p class="note">
        {#if sensors.power_w != null}Power {Number(sensors.power_w).toFixed(1)} W.{/if}
        {#if sensors.vram_used_bytes != null}
          Card memory {(Number(sensors.vram_used_bytes) / 1024 ** 3).toFixed(2)} GB in use.
        {/if}
        {#each Object.entries(sensors.temperature_c ?? {}) as [label, celsius]}
          {label} {celsius} °C.{' '}
        {/each}
      </p>
    {/if}
    {#if hardware.gpu_blocker}<p>{hardware.gpu_blocker}</p>{/if}
    {#if !hardware.torch_gpu && hardware.install}
      <p class="note">Install the gfx1012 wheel into this repo's virtualenv, then probe again:</p>
      <pre class="probe">{hardware.install}</pre>
    {/if}
    {#if budget}
      <table>
        <tbody>
          <tr><td>Weights</td><td>{mb(budget.weights_bytes)}</td></tr>
          <tr><td>AdamW moments</td><td>{mb(budget.optimizer_bytes)}</td></tr>
          <tr><td>Gradients</td><td>{mb(budget.gradient_bytes)}</td></tr>
          <tr><td>Activation bound</td><td>{mb(budget.activation_bytes)}</td></tr>
          <tr><td>Estimate</td><td>{mb(budget.total_bytes)} of {mb(budget.vram_bytes)}</td></tr>
        </tbody>
      </table>
      <p class="note">{budget.fits ? 'This preset fits in the 4GB budget with the 25% reserve left unused.' : 'This preset is over the 4GB reserve. Lower the batch or the context.'}</p>
    {/if}
    <ul>
      {#each guidance as line}<li>{line}</li>{/each}
    </ul>
  </section>
{/if}

<h2>Run</h2>
<p class="note">Compatibility mode, on the bench below, sets HSA_OVERRIDE_GFX_VERSION=10.3.0 only for that process. Use it when the RX 5500 is in the machine and a ROCm wheel that contains gfx1030 kernels still cannot see gfx1012. A native gfx1012 wheel does not need it.</p>
<RunBench phase="pretrain" steps={20} learningRate={0.003} />
