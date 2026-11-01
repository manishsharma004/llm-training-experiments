<script lang="ts">
  export type StagePoint = {
    id: string;
    x: number;
    y: number;
    z: number;
    color: string;
    label: string;
    detail: string;
    size?: number;
  };

  export type StageBox = {
    id: string;
    x: number;
    y: number;
    z: number;
    dx: number;
    dy: number;
    dz: number;
    color: string;
    label: string;
    detail: string;
  };

  export type StageSegment = {
    x1: number;
    y1: number;
    z1: number;
    x2: number;
    y2: number;
    z2: number;
    color: string;
  };

  let {
    points = [] as StagePoint[],
    boxes = [] as StageBox[],
    segments = [] as StageSegment[],
    ariaLabel = 'Interactive 3D view',
    onpick
  }: {
    points?: StagePoint[];
    boxes?: StageBox[];
    segments?: StageSegment[];
    ariaLabel?: string;
    onpick?: (label: string, detail: string) => void;
  } = $props();

  let canvas = $state<HTMLCanvasElement | null>(null);
  let yaw = $state(0.45);
  let pitch = $state(0.5);
  let distance = $state(5.4);
  let panX = $state(0);
  let panY = $state(0);
  let hover = $state('Drag to orbit. Scroll to zoom. Shift-drag to pan. Click a mark to pin it.');
  let pinned = $state('');

  type Drag = {
    x: number;
    y: number;
    yaw: number;
    pitch: number;
    panX: number;
    panY: number;
    button: number;
    moved: boolean;
  };
  let drag: Drag | null = null;

  function project(x: number, y: number, z: number, width: number, height: number) {
    const cosYaw = Math.cos(yaw);
    const sinYaw = Math.sin(yaw);
    const xYaw = x * cosYaw + z * sinYaw;
    const zYaw = -x * sinYaw + z * cosYaw;
    const cosPitch = Math.cos(pitch);
    const sinPitch = Math.sin(pitch);
    const yPitch = y * cosPitch - zYaw * sinPitch;
    const zPitch = y * sinPitch + zYaw * cosPitch;
    const depth = distance + zPitch;
    const scale = (Math.min(width, height) * 0.64) / Math.max(0.4, distance);
    const foreshorten = 1 / (1 + zPitch * 0.045);
    const drawn = scale * foreshorten;
    return {
      sx: width / 2 + (xYaw + panX) * drawn,
      sy: height / 2 - (yPitch + panY) * drawn,
      depth,
      scale: drawn
    };
  }

  function shade(hex: string, factor: number) {
    const raw = hex.replace('#', '');
    if (raw.length !== 6) return hex;
    const channels = [0, 2, 4].map((start) =>
      Math.max(0, Math.min(255, Math.round(parseInt(raw.slice(start, start + 2), 16) * factor)))
    );
    return `rgb(${channels.join(',')})`;
  }

  function draw(marks: { points: StagePoint[]; boxes: StageBox[]; segments: StageSegment[] }) {
    const { points, boxes, segments } = marks;
    const node = canvas;
    if (!node) return;
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    const rect = node.getBoundingClientRect();
    const width = Math.max(1, rect.width);
    const height = Math.max(1, rect.height);
    const pixelWidth = Math.floor(width * ratio);
    const pixelHeight = Math.floor(height * ratio);
    if (node.width !== pixelWidth || node.height !== pixelHeight) {
      node.width = pixelWidth;
      node.height = pixelHeight;
    }
    const ctx = node.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.fillStyle = '#fffaf3';
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = 'rgba(227, 214, 196, 0.95)';
    ctx.lineWidth = 1;
    for (let line = -2; line <= 2; line += 1) {
      const across = [project(line, 0, -2.4, width, height), project(line, 0, 2.4, width, height)];
      const depth = [project(-2.4, 0, line, width, height), project(2.4, 0, line, width, height)];
      ctx.beginPath();
      ctx.moveTo(across[0].sx, across[0].sy);
      ctx.lineTo(across[1].sx, across[1].sy);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(depth[0].sx, depth[0].sy);
      ctx.lineTo(depth[1].sx, depth[1].sy);
      ctx.stroke();
    }

    const faces: { depth: number; path: { sx: number; sy: number }[]; color: string }[] = [];
    const cornerSigns: [number, number, number][] = [
      [-1, -1, -1],
      [-1, -1, 1],
      [-1, 1, -1],
      [-1, 1, 1],
      [1, -1, -1],
      [1, -1, 1],
      [1, 1, -1],
      [1, 1, 1]
    ];
    const quads = [
      [0, 1, 3, 2],
      [4, 6, 7, 5],
      [0, 4, 5, 1],
      [2, 3, 7, 6],
      [0, 2, 6, 4],
      [1, 5, 7, 3]
    ];
    const shades = [0.76, 0.95, 0.68, 1, 0.6, 0.88];
    for (const box of boxes) {
      const corners = cornerSigns.map(([sx, sy, sz]) =>
        project(box.x + sx * box.dx, box.y + sy * box.dy, box.z + sz * box.dz, width, height)
      );
      quads.forEach((quad, index) => {
        const path = quad.map((corner) => corners[corner]);
        faces.push({
          depth: path.reduce((sum, point) => sum + point.depth, 0) / path.length,
          path,
          color: shade(box.color, shades[index])
        });
      });
    }
    faces.sort((left, right) => right.depth - left.depth);
    for (const face of faces) {
      ctx.beginPath();
      face.path.forEach((point, index) => (index === 0 ? ctx.moveTo(point.sx, point.sy) : ctx.lineTo(point.sx, point.sy)));
      ctx.closePath();
      ctx.fillStyle = face.color;
      ctx.fill();
      ctx.strokeStyle = 'rgba(36, 28, 22, 0.28)';
      ctx.stroke();
    }

    const lines = segments
      .map((segment) => {
        const start = project(segment.x1, segment.y1, segment.z1, width, height);
        const end = project(segment.x2, segment.y2, segment.z2, width, height);
        return { start, end, color: segment.color, depth: (start.depth + end.depth) / 2 };
      })
      .sort((left, right) => right.depth - left.depth);
    for (const line of lines) {
      ctx.strokeStyle = line.color;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(line.start.sx, line.start.sy);
      ctx.lineTo(line.end.sx, line.end.sy);
      ctx.stroke();
    }

    const dots = points
      .map((point) => ({ point, at: project(point.x, point.y, point.z, width, height) }))
      .sort((left, right) => right.at.depth - left.at.depth);
    for (const dot of dots) {
      ctx.beginPath();
      ctx.fillStyle = dot.point.color;
      ctx.arc(dot.at.sx, dot.at.sy, Math.max(2.2, (dot.point.size ?? 5) * (dot.at.scale / 48)), 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.font = '600 13px "Iowan Old Style", Palatino, Georgia, serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const paintLabel = (text: string, sx: number, sy: number) => {
      const shown = [...text].slice(0, 12).join('');
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'rgba(36, 28, 22, 0.78)';
      ctx.strokeText(shown, sx, sy);
      ctx.fillStyle = '#fffaf3';
      ctx.fillText(shown, sx, sy);
    };
    for (const box of boxes) {
      const at = project(box.x, box.y + box.dy, box.z, width, height);
      if (at.sx < -20 || at.sy < -10 || at.sx > width + 20 || at.sy > height + 10) continue;
      paintLabel(box.label, at.sx, at.sy - 11);
    }
    const parked: { sx: number; sy: number }[] = [];
    for (const dot of dots) {
      if ((dot.point.size ?? 5) < 8 || !dot.point.label) continue;
      if (dot.at.sx < -20 || dot.at.sy < -10 || dot.at.sx > width + 20 || dot.at.sy > height + 10) continue;
      let sy = dot.at.sy - 14;
      while (parked.some((item) => Math.abs(item.sx - dot.at.sx) < 54 && Math.abs(item.sy - sy) < 15)) sy -= 15;
      parked.push({ sx: dot.at.sx, sy });
      paintLabel(dot.point.label, dot.at.sx, sy);
    }
  }

  function hit(clientX: number, clientY: number) {
    const node = canvas;
    if (!node) return null;
    const rect = node.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    let best: { distance: number; label: string; detail: string } | null = null;
    for (const point of points) {
      const at = project(point.x, point.y, point.z, rect.width, rect.height);
      const distance = Math.hypot(at.sx - x, at.sy - y);
      if (distance < 16 && (!best || distance < best.distance)) best = { distance, label: point.label, detail: point.detail };
    }
    for (const box of boxes) {
      const at = project(box.x, box.y, box.z, rect.width, rect.height);
      const distance = Math.hypot(at.sx - x, at.sy - y);
      if (distance < 32 && (!best || distance < best.distance)) best = { distance, label: box.label, detail: box.detail };
    }
    return best;
  }

  $effect(() => {
    const marks = { points, boxes, segments };
    yaw;
    pitch;
    distance;
    panX;
    panY;
    canvas;
    draw(marks);
  });

  $effect(() => {
    const node = canvas;
    if (!node) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      distance = Math.max(2.4, Math.min(22, distance * Math.exp(event.deltaY * 0.0011)));
    };
    node.addEventListener('wheel', onWheel, { passive: false });
    return () => node.removeEventListener('wheel', onWheel);
  });

  function pointerDown(event: PointerEvent) {
    drag = { x: event.clientX, y: event.clientY, yaw, pitch, panX, panY, button: event.button, moved: false };
    try {
      (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
    } catch {
      // Synthetic events and a few browsers reject capture. Orbit still tracks the pointer.
    }
  }

  function pointerMove(event: PointerEvent) {
    if (!drag) {
      const found = hit(event.clientX, event.clientY);
      hover = found ? `${found.label}. ${found.detail}` : pinned || 'Drag to orbit. Scroll to zoom. Shift-drag to pan.';
      return;
    }
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    if (Math.hypot(dx, dy) > 3) drag.moved = true;
    if (event.shiftKey || drag.button === 2) {
      panX = drag.panX + dx * 0.012;
      panY = drag.panY - dy * 0.012;
    } else {
      yaw = drag.yaw + dx * 0.008;
      pitch = Math.max(-1.15, Math.min(1.15, drag.pitch + dy * 0.006));
    }
  }

  function pointerUp(event: PointerEvent) {
    if (drag && !drag.moved) {
      const found = hit(event.clientX, event.clientY);
      if (found) {
        pinned = `${found.label}. ${found.detail}`;
        hover = pinned;
        onpick?.(found.label, found.detail);
      }
    }
    drag = null;
  }

  function resetView() {
    yaw = 0.45;
    pitch = 0.5;
    distance = 5.4;
    panX = 0;
    panY = 0;
  }

  function nudge(key: string) {
    if (key === 'ArrowLeft') yaw -= 0.12;
    if (key === 'ArrowRight') yaw += 0.12;
    if (key === 'ArrowUp') pitch = Math.max(-1.15, pitch - 0.08);
    if (key === 'ArrowDown') pitch = Math.min(1.15, pitch + 0.08);
    if (key === '+' || key === '=') distance = Math.max(2.4, distance - 0.4);
    if (key === '-' || key === '_') distance = Math.min(22, distance + 0.4);
  }
</script>

<div class="view">
  <canvas
    bind:this={canvas}
    aria-label={ariaLabel}
    tabindex="0"
    onpointerdown={pointerDown}
    onpointermove={pointerMove}
    onpointerup={pointerUp}
    onpointercancel={pointerUp}
    oncontextmenu={(event) => event.preventDefault()}
    onkeydown={(event) => nudge(event.key)}
  ></canvas>
  <div class="toolbar">
    <button type="button" class="btn ghost" onclick={() => (distance = Math.max(2.4, distance - 0.7))}>Closer</button>
    <button type="button" class="btn ghost" onclick={() => (distance = Math.min(22, distance + 0.7))}>Farther</button>
    <button type="button" class="btn ghost" onclick={resetView}>Reset view</button>
  </div>
  <p class="hover">{hover}</p>
</div>

<style>
  .view { display: grid; gap: 8px; }
  canvas {
    width: 100%;
    height: min(58vh, 640px);
    min-height: 380px;
    border: 1px solid var(--line);
    border-radius: 16px;
    background: #fffaf3;
    touch-action: none;
    cursor: grab;
  }
  canvas:active { cursor: grabbing; }
  canvas:focus { outline: 2px solid var(--copper); outline-offset: 2px; }
  .toolbar { display: flex; flex-wrap: wrap; gap: 8px; }
  .hover {
    margin: 0;
    min-height: 4.6em;
    color: var(--ink-soft);
    font-size: 0.92rem;
  }
</style>
