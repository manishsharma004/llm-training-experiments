<script lang="ts">
  import { onMount } from 'svelte';

  let report = $state<Record<string, unknown> | null>(null);
  let error = $state('');
  let busy = $state(false);

  async function prepare() {
    error = '';
    busy = true;
    try {
      const response = await fetch('/api/data/prepare', { method: 'POST' });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error ?? 'Prepare failed');
      report = body;
    } catch (caught) {
      error = caught instanceof Error ? caught.message : 'Prepare failed';
    } finally {
      busy = false;
    }
  }

  onMount(async () => {
    const response = await fetch('/api/data');
    const body = await response.json();
    if (body.prepare) report = body.prepare;
  });

  const rows = $derived((report?.assignment as { source: string; split: string }[] | undefined) ?? []);
</script>

<p class="kicker">04 · Prepare</p>
<h1>Pack cleaned documents into the bytes the model sees.</h1>
<p class="lede">
  Documents are split into train and validation before they are concatenated, so a validation loss is not just a replay of the training windows.
  A newline byte separates documents. The byte tokenizer from the lab is the same encoding written into <span class="formula" style="display:inline;padding:2px 6px">train.bin</span>.
</p>

<button class="btn" onclick={prepare} disabled={busy}>{busy ? 'Preparing' : 'Prepare train and validation streams'}</button>
{#if error}<p class="warn">{error}</p>{/if}

{#if report}
  <div class="stats" style="margin-top:16px">
    <div class="stat"><span class="note">train bytes</span><strong>{report.train_tokens}</strong></div>
    <div class="stat"><span class="note">validation bytes</span><strong>{report.val_tokens}</strong></div>
    <div class="stat"><span class="note">windows</span><strong>{report.train_windows}</strong></div>
    <div class="stat"><span class="note">question pairs</span><strong>{report.sft_pairs}</strong></div>
  </div>
  <section class="card">
    <h3>First bytes of the training stream</h3>
    <pre class="probe">{report.preview}</pre>
    <p class="note">Context length used for the window count is {report.block_size}. {report.separator}.</p>
  </section>
  <section class="card">
    <table>
      <thead><tr><th>Document</th><th>Split</th></tr></thead>
      <tbody>
        {#each rows as row}
          <tr><td>{row.source}</td><td>{row.split}</td></tr>
        {/each}
      </tbody>
    </table>
  </section>
{/if}
