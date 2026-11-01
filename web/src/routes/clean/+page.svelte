<script lang="ts">
  import { onMount } from 'svelte';

  let report = $state<Record<string, unknown> | null>(null);
  let error = $state('');
  let busy = $state(false);

  async function clean() {
    error = '';
    busy = true;
    try {
      const response = await fetch('/api/data/clean', { method: 'POST' });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error ?? 'Clean failed');
      report = body;
    } catch (caught) {
      error = caught instanceof Error ? caught.message : 'Clean failed';
    } finally {
      busy = false;
    }
  }

  onMount(async () => {
    const response = await fetch('/api/data');
    const body = await response.json();
    if (body.clean) report = body.clean;
  });

  const reasons = $derived((report?.reasons as Record<string, number> | undefined) ?? {});
  const rules = $derived((report?.rules as string[] | undefined) ?? []);
  const dropped = $derived((report?.dropped_examples as { source: string; reasons: string[]; excerpt: string }[] | undefined) ?? []);
  const kept = $derived((report?.kept_preview as { source: string; chars: number; after: string }[] | undefined) ?? []);
</script>

<p class="kicker">03 · Clean</p>
<h1>Decide what is allowed to become training text.</h1>
<p class="lede">
  Every drop has a reason. Exact duplicates are removed after normalization, so two copies of the same notes do not get twice the weight.
  HTML is stripped rather than learned as tags.
</p>

<button class="btn" onclick={clean} disabled={busy}>{busy ? 'Cleaning' : 'Clean the gathered documents'}</button>
{#if error}<p class="warn">{error}</p>{/if}

{#if report}
  <div class="stats" style="margin-top:16px">
    <div class="stat"><span class="note">in</span><strong>{report.input_documents}</strong></div>
    <div class="stat"><span class="note">kept</span><strong>{report.kept}</strong></div>
    <div class="stat"><span class="note">dropped</span><strong>{report.dropped}</strong></div>
  </div>
  <div class="grid">
    <section class="card">
      <h3>Rules</h3>
      <ul>
        {#each rules as rule}<li>{rule}</li>{/each}
      </ul>
    </section>
    <section class="card">
      <h3>Why documents were dropped</h3>
      <table>
        <tbody>
          {#each Object.entries(reasons) as [reason, count]}
            <tr><td>{reason}</td><td>{count}</td></tr>
          {/each}
        </tbody>
      </table>
      {#each dropped as row}
        <p class="note"><strong>{row.source}</strong> · {row.reasons.join(', ')}</p>
        <pre class="probe">{row.excerpt}</pre>
      {/each}
    </section>
  </div>
  <section class="card">
    <h3>Kept, after cleaning</h3>
    {#each kept as row}
      <p class="note">{row.source} · {row.chars} characters</p>
      <pre class="probe">{row.after}</pre>
    {/each}
  </section>
{/if}
