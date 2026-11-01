<script lang="ts">
  import '../app.css';
  import '@fontsource/literata/600.css';
  import '@fontsource/ibm-plex-sans/400.css';
  import '@fontsource/ibm-plex-sans/500.css';
  import '@fontsource/ibm-plex-mono/400.css';
  import { page } from '$app/state';
  import StageDock from '#lib/components/stage/StageDock.svelte';

  let { children } = $props();

  const links = [
    ['/', 'Notes', '01'],
    ['/gather', 'Gather', '02'],
    ['/clean', 'Clean', '03'],
    ['/prepare', 'Prepare', '04'],
    ['/architecture', 'Shape', '05'],
    ['/train', 'Train', '06'],
    ['/finetune', 'Finetune', '07'],
    ['/generate', 'Sample', '08']
  ] as const;
</script>

<svelte:head>
  <title>Harbor lab</title>
</svelte:head>

<div class="shell">
  <aside class="rail">
    <p class="brand">Harbor lab<small>one million parameters</small></p>
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
