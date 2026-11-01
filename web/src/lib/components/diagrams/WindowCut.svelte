<figure class="figure">
  <svg viewBox="0 0 980 280" role="img" aria-labelledby="window-title window-desc">
    <title id="window-title">How cleaned documents become training windows</title>
    <desc id="window-desc">
      One document is held out as validation before the others are joined. A newline byte separates documents in the training stream. Each step cuts a context-sized window at a random start and uses the following byte as the label.
    </desc>
    <rect width="980" height="280" fill="#fffaf3" rx="16" />
    <text class="label" x="24" y="36">documents after cleaning</text>
    <g transform="translate(24 52)">
      <rect width="200" height="46" rx="8" fill="#d7e6e1" stroke="#1e4642" />
      <text class="sheet" x="12" y="28">harbor-notes → train</text>
      <rect x="216" width="180" height="46" rx="8" fill="#f6e2c8" stroke="#8d3f16" />
      <text class="sheet" x="228" y="28">watch-log → val</text>
      <rect x="412" width="200" height="46" rx="8" fill="#d7e6e1" stroke="#1e4642" />
      <text class="sheet" x="424" y="28">scrape, tags gone → train</text>
    </g>

    <text class="label" x="24" y="138">train.bin, documents joined by a newline byte</text>
    <g transform="translate(24 152)">
      {#each Array(42) as _, index}
        <rect
          x={index * 18}
          y="0"
          width="16"
          height="28"
          rx="3"
          fill={index === 14 ? '#241c16' : index >= 6 && index < 20 ? '#c4622d' : '#1e4642'}
          opacity={index >= 6 && index < 20 ? 1 : 0.35}
        />
      {/each}
      <text class="tiny" x="0" y="52">Copper cells are one sampled window. The ribbon is drawn shorter than 128 bytes so the cut stays visible.</text>
      <text class="tiny" x="0" y="72">The dark cell is a newline between documents.</text>
    </g>
  </svg>
  <figcaption>
    The holdout is chosen from whole documents, then the train documents are concatenated. The model does not see document boundaries as a special token. It sees whatever byte was written there, including the newline. A step does not scan left to right. It samples a start index, reads 128 bytes, and predicts the byte one past each of those positions.
  </figcaption>
</figure>

<style>
  .figure { margin: 8px 0 18px; }
  svg { width: 100%; height: auto; border: 1px solid var(--line); border-radius: 16px; display: block; }
  figcaption { color: var(--ink-soft); font-size: 0.95rem; margin-top: 8px; max-width: 78ch; }
  .label { font-family: var(--mono); font-size: 12px; fill: #8d3f16; }
  .sheet { font-family: var(--sans); font-size: 14px; fill: #241c16; }
  .tiny { font-family: var(--sans); font-size: 12px; fill: #5c5146; }
</style>
