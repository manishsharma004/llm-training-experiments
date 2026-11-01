<script lang="ts">
  import '../app.css';
  import '@fontsource/literata/600.css';
  import '@fontsource/ibm-plex-sans/400.css';
  import '@fontsource/ibm-plex-sans/500.css';
  import '@fontsource/ibm-plex-mono/400.css';
  import { onMount } from 'svelte';
  import { page } from '$app/state';
  import StageDock from '#lib/components/stage/StageDock.svelte';
  import { finishHydration, lab } from '#lib/lab.svelte.ts';
  import { readStore, sanitizeLab, writeStoreSoon } from '#lib/persist.ts';

  let { children } = $props();
  let remember = $state(false);

  onMount(async () => {
    let saved = false;
    try {
      const snap = sanitizeLab(await readStore('lab'));
      if (snap) {
        lab.shape = snap.shape;
        lab.prompt = snap.prompt;
        lab.completion = snap.completion;
        lab.temperature = snap.temperature;
        saved = true;
      }
    } finally {
      remember = true;
      finishHydration(saved);
    }
  });

  $effect(() => {
    const snap = {
      shape: { ...lab.shape },
      prompt: lab.prompt,
      completion: lab.completion,
      temperature: lab.temperature
    };
    if (!remember) return;
    writeStoreSoon('lab', snap);
  });

  const links = [
    ['/', 'Notes', '01'],
    ['/gather', 'Gather', '02'],
    ['/clean', 'Clean', '03'],
    ['/prepare', 'Prepare', '04'],
    ['/tokenize', 'Tokens', '05'],
    ['/architecture', 'Shape', '06'],
    ['/train', 'Train', '07'],
    ['/finetune', 'Finetune', '08'],
    ['/generate', 'Sample', '09']
  ] as const;
</script>

<svelte:head>
  <title>Book lab</title>
</svelte:head>

<div class="shell">
  <aside class="rail">
    <p class="brand">Book lab<small>8,192-piece vocabulary</small></p>
    <nav>
      {#each links as [href, label, index]}
        <a {href} aria-current={page.url.pathname === href ? 'page' : undefined}>
          <span>{index}</span>{label}
        </a>
      {/each}
    </nav>
  </aside>
  <main class="main">
    {@render children()}
  </main>
  <StageDock />
</div>
