<script lang="ts">
  import { onMount } from 'svelte';

  let report = $state<Record<string, unknown> | null>(null);
  let error = $state('');
  let busy = $state(false);
  let name = $state('field-note');
  let text = $state('');

  async function load() {
    const response = await fetch('/api/data');
    report = await response.json();
  }

  onMount(load);

  async function gather() {
    error = '';
    busy = true;
    try {
      const response = await fetch('/api/data/gather', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(text.trim() ? { name, text } : {})
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error ?? 'Gather failed');
      report = { ...(report ?? {}), gather: body, raw: body.sources };
      text = '';
      await load();
    } catch (caught) {
      error = caught instanceof Error ? caught.message : 'Gather failed';
    } finally {
      busy = false;
    }
  }

  const sources = $derived((report?.gather as { sources?: { source: string; bytes: number; chars: number }[] } | null)?.sources ?? []);
  const raw = $derived((report?.raw as { source: string; bytes: number }[] | undefined) ?? []);
</script>

<p class="kicker">02 · Gather</p>
<h1>Bring documents in before any training.</h1>
<p class="lede">
  Raw files live in <span class="formula" style="display:inline;padding:2px 6px">{(report?.dataset as { raw?: string } | undefined)?.raw ?? 'data/corpus/raw'}</span>.
  Gathering only records them. It does not delete HTML, duplicates, or junk. That is the next stage, so you can see the difference.
</p>

<section class="card stack">
  <h2>Add a document</h2>
  <div class="row">
    <label class="field">File name
      <input bind:value={name} />
    </label>
  </div>
  <label class="field">Text
    <textarea bind:value={text} placeholder="Paste notes. Leave this empty to gather the files already on disk."></textarea>
  </label>
  <button class="btn" onclick={gather} disabled={busy}>{busy ? 'Gathering' : 'Gather'}</button>
  {#if error}<p class="warn">{error}</p>{/if}
</section>

<section class="card">
  <h2>On disk</h2>
  <table>
    <thead><tr><th>Source</th><th>Bytes</th></tr></thead>
    <tbody>
      {#each raw as row}
        <tr><td>{row.source}</td><td>{row.bytes}</td></tr>
      {/each}
    </tbody>
  </table>
  {#if sources.length}
    <p class="note">Last gather recorded {sources.length} documents, {(report?.gather as { bytes?: number })?.bytes ?? 0} bytes.</p>
  {/if}
</section>
