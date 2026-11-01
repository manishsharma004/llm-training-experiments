<figure class="figure">
  <svg viewBox="0 0 980 430" role="img" aria-labelledby="pipeline-title pipeline-desc">
    <title id="pipeline-title">From raw files to a gradient step</title>
    <desc id="pipeline-desc">
      Gather keeps every file. Clean strips HTML and drops duplicates and junk. Prepare holds out a validation document and packs the rest into a byte stream. Training draws a 128-byte window and updates the million-parameter model.
    </desc>
    <rect width="980" height="430" fill="#fffaf3" rx="16" />

    <!-- gather -->
    <g transform="translate(28 36)">
      <text class="stage" x="0" y="16">1 · Gather</text>
      <g transform="translate(8 40)">
        <rect x="18" y="18" width="92" height="118" rx="6" fill="#f6e2c8" stroke="#c4622d" />
        <text class="sheet" x="28" y="48">&lt;p&gt;lamp</text>
        <text class="sheet" x="28" y="70">is copper</text>
        <rect x="0" y="0" width="92" height="118" rx="6" fill="#fffaf3" stroke="#241c16" />
        <text class="sheet" x="10" y="28">harbor-notes</text>
        <text class="tiny" x="10" y="52">The lamp is</text>
        <text class="tiny" x="10" y="70">copper.</text>
        <rect x="36" y="78" width="92" height="118" rx="6" fill="#fffaf3" stroke="#8d2b22" />
        <text class="sheet" x="46" y="106">notes-copy</text>
        <text class="tiny" x="46" y="128">same bytes</text>
        <text class="stamp" x="48" y="160">duplicate</text>
      </g>
      <text class="caption" x="0" y="268">Nothing is rejected.</text>
      <text class="caption" x="0" y="286">HTML, copies, and junk</text>
      <text class="caption" x="0" y="304">all sit in the tray.</text>
    </g>

    <path d="M210 150 H248" stroke="#c4622d" stroke-width="2" marker-end="url(#arrow)" />

    <!-- clean -->
    <g transform="translate(258 36)">
      <text class="stage" x="0" y="16">2 · Clean</text>
      <g transform="translate(6 48)">
        <path d="M20 0 H150 L128 54 H42 Z" fill="#d7e6e1" stroke="#1e4642" />
        <text class="tiny" x="48" y="28">strip tags · NFKC</text>
        <rect x="0" y="78" width="86" height="78" rx="6" fill="#fffaf3" stroke="#1e4642" />
        <text class="sheet" x="10" y="106">kept</text>
        <text class="tiny" x="10" y="128">plain prose</text>
        <rect x="100" y="78" width="86" height="78" rx="6" fill="#fde8e4" stroke="#8d2b22" />
        <text class="sheet" x="110" y="106">dropped</text>
        <text class="tiny" x="110" y="128">duplicate</text>
        <text class="tiny" x="110" y="144">too short</text>
      </g>
      <text class="caption" x="0" y="268">Each rejection keeps</text>
      <text class="caption" x="0" y="286">a reason. Tags never</text>
      <text class="caption" x="0" y="304">become tokens.</text>
    </g>

    <path d="M460 150 H498" stroke="#c4622d" stroke-width="2" marker-end="url(#arrow)" />

    <!-- prepare -->
    <g transform="translate(508 36)">
      <text class="stage" x="0" y="16">3 · Prepare</text>
      <g transform="translate(0 52)">
        <rect x="0" y="0" width="210" height="22" rx="4" fill="#1e4642" />
        <text class="on-dark" x="8" y="15">train.bin</text>
        <rect x="78" y="0" width="46" height="22" fill="none" stroke="#f6e2c8" stroke-width="2" />
        <text class="tiny" x="0" y="44">window of 128 bytes, start chosen at random</text>
        <rect x="0" y="62" width="92" height="22" rx="4" fill="#d89a55" />
        <text class="on-dark" x="8" y="77">val.bin</text>
        <text class="tiny" x="0" y="106">held out before packing</text>
        <text class="tiny" x="0" y="138">↵ between documents</text>
      </g>
      <text class="caption" x="0" y="268">A newline byte separates</text>
      <text class="caption" x="0" y="286">documents. Validation is</text>
      <text class="caption" x="0" y="304">not a copy of the train windows.</text>
    </g>

    <path d="M740 150 H778" stroke="#c4622d" stroke-width="2" marker-end="url(#arrow)" />

    <!-- train -->
    <g transform="translate(788 36)">
      <text class="stage" x="0" y="16">4 · Step</text>
      <g transform="translate(4 46)">
        <rect x="8" y="28" width="150" height="86" rx="8" fill="#241c16" />
        <rect x="18" y="40" width="130" height="48" rx="4" fill="#1e4642" />
        <path d="M28 74 C 50 50, 70 58, 92 46 S 130 40, 140 52" fill="none" stroke="#f6e2c8" stroke-width="2" />
        <text class="on-dark" x="22" y="104">RX 5500 · fp32</text>
        <text class="tiny" x="0" y="140">1,035,008 weights</text>
        <text class="tiny" x="0" y="158">predict the next byte</text>
      </g>
      <text class="caption" x="0" y="268">AdamW updates those</text>
      <text class="caption" x="0" y="286">weights. The card holds</text>
      <text class="caption" x="0" y="304">weights, moments, gradients.</text>
    </g>

    <defs>
      <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
        <path d="M0 0 L6 3 L0 6 Z" fill="#c4622d" />
      </marker>
    </defs>
  </svg>
  <figcaption>
    Gather records files as they are. Clean is the first place a document can disappear, and it has to say why. Prepare turns the survivors into two byte streams. A training step never sees a file, only a 128-byte window and the byte that follows each position.
  </figcaption>
</figure>

<style>
  .figure { margin: 8px 0 18px; }
  svg { width: 100%; height: auto; border: 1px solid var(--line); border-radius: 16px; }
  figcaption { color: var(--ink-soft); font-size: 0.95rem; margin-top: 8px; max-width: 78ch; }
  .stage { font-family: var(--mono); font-size: 13px; fill: #8d3f16; letter-spacing: 0.04em; }
  .sheet, .on-dark { font-family: var(--sans); font-size: 13px; fill: #241c16; }
  .on-dark { fill: #f4efe6; font-size: 12px; }
  .tiny, .caption { font-family: var(--sans); font-size: 12px; fill: #5c5146; }
  .stamp { font-family: var(--mono); font-size: 11px; fill: #8d2b22; }
</style>
