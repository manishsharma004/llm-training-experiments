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
<h1>Encode the books into the training stream.</h1>
<p class="lede">
  Documents are split into train and validation before they are concatenated, so a validation loss is not just a replay of the training windows.
  A newline separates documents. The byte-level BPE from the Tokens page is the encoding written into <span class="formula" style="display:inline;padding:2px 6px">data/corpus/work/train.bin</span>.
</p>

<button class="btn" onclick={prepare} disabled={busy}>{busy ? 'Preparing' : 'Prepare train and validation streams'}</button>
{#if error}<p class="warn">{error}</p>{/if}

{#if report}
  <div class="stats" style="margin-top:16px">
    <div class="stat"><span class="note">train tokens</span><strong>{Number(report.train_tokens).toLocaleString()}</strong></div>
    <div class="stat"><span class="note">validation tokens</span><strong>{Number(report.val_tokens).toLocaleString()}</strong></div>
    <div class="stat"><span class="note">windows</span><strong>{Number(report.train_windows).toLocaleString()}</strong></div>
    <div class="stat"><span class="note">bytes / token</span><strong>{report.bytes_per_token ?? '—'}</strong></div>
  </div>
  <section class="card">
    <h3>First tokens, decoded</h3>
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
