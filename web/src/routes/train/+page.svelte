<script lang="ts">
  import { onMount } from 'svelte';
  import RunBench from '#lib/components/RunBench.svelte';

  let hardware = $state<Record<string, unknown> | null>(null);
  let probeError = $state('');

  onMount(async () => {
    const response = await fetch('/api/device');
    const body = await response.json();
    if (!response.ok) probeError = body.error ?? 'Device probe failed';
    else hardware = body;
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

<p class="kicker">06 · Train</p>
<h1>Train the million-parameter model on the prepared bytes.</h1>
<p class="lede">
  Auto uses a GPU when this PyTorch process can see one, and the CPU otherwise. Choosing GPU on a machine
  where the RX 5500 is not visible stops the run and explains why, instead of silently training on the CPU.
  The preset stays in fp32: weights, gradients, and AdamW states are about 16MB, which fits in 4GB with room to spare.
  The stage shows the book stream, the 23 million training bytes, and the finished pass over that stream. The step box starts at 20 so a new run stays short.
</p>

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
