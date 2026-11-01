<script lang="ts">
  let { file, code }: { file: string; code: string } = $props();
</script>

<figure class="source">
  <figcaption>{file}</figcaption>
  <pre><code>{code}</code></pre>
</figure>
