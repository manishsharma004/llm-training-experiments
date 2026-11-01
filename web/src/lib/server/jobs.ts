import { spawn, spawnSync, type ChildProcess } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';

export type Phase = 'pretrain' | 'finetune' | 'pipeline';

export type RunRequest = {
  phase: Phase;
  steps: number;
  finetuneSteps: number;
  seed: number;
  batchSize: number;
  learningRate: number;
  device: 'auto' | 'cpu' | 'gpu';
  gfxOverride: boolean;
};

const MAX_STEPS = 20_000;

export function repoRoot(): string {
  const cwd = process.cwd();
  if (existsSync(path.join(cwd, 'tinyllm', 'cli.py'))) return cwd;
  const parent = path.resolve(cwd, '..');
  if (existsSync(path.join(parent, 'tinyllm', 'cli.py'))) return parent;
  throw new Error('Start the lab from the web/ directory so it can find the tinyllm package.');
}

function pythonBin(root: string): string {
  const bin = path.join(root, '.venv', 'bin', 'python');
  if (!existsSync(bin)) {
    throw new Error('Python environment missing. From the repo root run: uv sync --group dev');
  }
  return bin;
}

function runsDir(root: string): string {
  return path.join(root, 'runs');
}

/** The book stream when it is on disk, otherwise the small harbor sample. */
export function datasetPaths(root = repoRoot()) {
  const corpus = path.join(root, 'data', 'corpus', 'work', 'train.bin');
  if (existsSync(corpus)) {
    return { raw: 'data/corpus/raw', work: 'data/corpus/work', kind: 'corpus' as const };
  }
  return { raw: 'data/raw', work: 'data/work', kind: 'harbor' as const };
}

function presentRun(id: string, status: Record<string, unknown>, message = '') {
  const metrics = Array.isArray(status.metrics) ? status.metrics : [];
  const last = (metrics[metrics.length - 1] ?? {}) as { tokens_per_sec?: number; tokensPerSec?: number };
  const tokensPerSec = status.tokensPerSec ?? last.tokensPerSec ?? last.tokens_per_sec;
  return {
    id,
    ...status,
    tokensPerSec,
    state: String(status.state ?? 'unknown'),
    message: typeof status.message === 'string' ? status.message : message
  };
}

export function readJson(file: string): Record<string, unknown> | null {
  if (!existsSync(file)) return null;
  try {
    return JSON.parse(readFileSync(file, 'utf8')) as Record<string, unknown>;
  } catch {
    return null;
  }
}

function alive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function clampInt(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, Math.round(value)));
}

export function loadRun(root: string, id: string) {
  if (!/^[A-Za-z0-9._-]+$/.test(id)) return null;
  if (id === 'corpus') {
    const status = readJson(path.join(runsDir(root), 'corpus-status.json'));
    if (!status) return null;
    return { ...presentRun('corpus', status), pid: 0 };
  }
  const dir = path.join(runsDir(root), id);
  const status = readJson(path.join(dir, 'status.json'));
  const meta = readJson(path.join(dir, 'meta.json'));
  if (!status && !meta) return null;
  const pid = Number(meta?.pid ?? 0);
  let state = String(status?.state ?? 'unknown');
  let message = typeof status?.message === 'string' ? status.message : '';
  const resumeReady = existsSync(path.join(dir, 'resume.pt'));
  const processAlive = Boolean(pid) && alive(pid);
  if ((state === 'running' || state === 'starting') && pid && !processAlive) {
    if (resumeReady) {
      state = 'paused';
      message = `Paused after step ${status?.step ?? '?'}. The process stopped. Resume reloads that step.`;
    } else {
      state = 'error';
      message = message || 'The training process stopped before writing a final status.';
    }
  }
  if (state === 'paused' && !processAlive && !resumeReady) {
    state = 'error';
    message = 'The training process stopped and the paused step was not saved.';
  }
  return { ...presentRun(id, status ?? {}, message), state, message, pid };
}

export function listRuns(root: string) {
  const dir = runsDir(root);
  if (!existsSync(dir)) return [] as ReturnType<typeof loadRun>[];
  const ids = readdirSync(dir)
    .filter((name) => existsSync(path.join(dir, name, 'status.json')))
    .sort()
    .reverse();
  const runs = ids.map((id) => loadRun(root, id)).filter((run) => run !== null);
  const corpus = loadRun(root, 'corpus');
  if (corpus && !runs.some((run) => run.id === 'corpus')) runs.push(corpus);
  return runs;
}

export function latestRun(root: string) {
  return listRuns(root)[0] ?? null;
}

export type CheckpointFact = {
  path: string;
  phase?: string;
  step?: number;
  loss?: number;
  vocabSize?: number;
  blockSize?: number;
  batchSize?: number;
};

let checkpointCache: { key: string; rows: CheckpointFact[] } | null = null;

export function listCheckpoints(root: string): CheckpointFact[] {
  const dir = path.join(root, 'checkpoints');
  if (!existsSync(dir)) return [];
  const files: string[] = [];
  const walk = (current: string) => {
    for (const name of readdirSync(current, { withFileTypes: true })) {
      const abs = path.join(current, name.name);
      if (name.isDirectory()) walk(abs);
      else if (name.name.endsWith('.pt')) files.push(abs);
    }
  };
  walk(dir);
  const key = files.map((abs) => `${abs}:${statSync(abs).mtimeMs}`).sort().join('|');
  if (checkpointCache?.key === key) return checkpointCache.rows;
  const fallback = files
    .map((abs) => ({ path: path.relative(root, abs).split(path.sep).join('/') }))
    .sort((left, right) => left.path.localeCompare(right.path));
  const result = spawnSync(pythonBin(root), ['-m', 'tinyllm.cli', 'checkpoints', '--root', root], {
    cwd: root,
    encoding: 'utf8',
    timeout: 60_000
  });
  if (result.status !== 0 || !result.stdout.trim()) {
    checkpointCache = { key, rows: fallback };
    return fallback;
  }
  const parsed = JSON.parse(result.stdout) as CheckpointFact[];
  const rows = parsed.filter((row) => typeof row.path === 'string').sort((left, right) => left.path.localeCompare(right.path));
  checkpointCache = { key, rows };
  return rows;
}

function checkpointAllowed(root: string, rel: string): string | null {
  if (!rel.endsWith('.pt') || rel.includes('..')) return null;
  const abs = path.resolve(root, rel);
  const base = path.join(root, 'checkpoints') + path.sep;
  if (!abs.startsWith(base)) return null;
  if (!existsSync(abs)) return null;
  return abs;
}

function writeControl(dir: string, command: 'run' | 'pause') {
  const file = path.join(dir, 'control.json');
  const temporary = `${file}.tmp`;
  writeFileSync(temporary, JSON.stringify({ command }));
  renameSync(temporary, file);
}

function watchChild(dir: string, statusPath: string, child: ChildProcess) {
  const stderrChunks: Buffer[] = [];
  child.stderr?.on('data', (chunk: Buffer) => {
    stderrChunks.push(chunk);
  });
  child.on('exit', (code) => {
    const metaNow = readJson(path.join(dir, 'meta.json'));
    if (Number(metaNow?.pid ?? 0) !== child.pid) return;
    const current = readJson(statusPath) ?? {};
    const text = Buffer.concat(stderrChunks).toString('utf8').trim();
    if (text) writeFileSync(path.join(dir, 'stderr.txt'), text);
    if (code === 0 && current.state === 'done') return;
    if (existsSync(path.join(dir, 'resume.pt')) && current.state !== 'done') {
      const step = current.step ?? '?';
      writeFileSync(
        statusPath,
        JSON.stringify(
          {
            ...current,
            state: 'paused',
            message:
              typeof current.message === 'string' && current.message
                ? current.message
                : `Paused after step ${step}. Resume reloads that step and continues.`
          },
          null,
          2
        )
      );
      return;
    }
    writeFileSync(
      statusPath,
      JSON.stringify(
        {
          ...current,
          state: code === 0 ? 'done' : 'error',
          message: code === 0 ? '' : text.slice(-2000) || `Training exited with code ${code}`
        },
        null,
        2
      )
    );
  });
}

export function controlRun(id: string, command: string) {
  const root = repoRoot();
  if (!/^[A-Za-z0-9._-]+$/.test(id) || id === 'corpus') {
    return { ok: false as const, status: 404, error: 'Unknown run.' };
  }
  const run = loadRun(root, id);
  if (!run) return { ok: false as const, status: 404, error: 'Unknown run.' };
  const dir = path.join(runsDir(root), id);
  if (command === 'pause') {
    if (run.state !== 'running') {
      return { ok: false as const, status: 409, error: 'Pause applies to a run that is on a step.', run };
    }
    const pid = Number(run.pid ?? 0);
    if (!pid || !alive(pid)) {
      return { ok: false as const, status: 409, error: 'The training process is not running.', run };
    }
    writeControl(dir, 'pause');
    return { ok: true as const, run };
  }
  if (command !== 'resume') {
    return { ok: false as const, status: 400, error: 'Command must be pause or resume.' };
  }
  if (run.state !== 'paused') {
    return { ok: false as const, status: 409, error: 'Resume continues a paused run.', run };
  }
  const pid = Number(run.pid ?? 0);
  if (pid && alive(pid)) {
    writeControl(dir, 'run');
    return { ok: true as const, run };
  }
  const blob = path.join(dir, 'resume.pt');
  if (!existsSync(blob)) {
    return { ok: false as const, status: 409, error: 'The paused step was not saved, so it cannot resume.', run };
  }
  const meta = readJson(path.join(dir, 'meta.json')) ?? {};
  const device = meta.device === 'cpu' || meta.device === 'gpu' || meta.device === 'auto' ? String(meta.device) : 'auto';
  const statusPath = path.join(dir, 'status.json');
  writeControl(dir, 'run');
  const current = readJson(statusPath) ?? {};
  writeFileSync(
    statusPath,
    JSON.stringify({ ...current, state: 'running', message: 'Resuming from the saved step.' }, null, 2)
  );
  const env = { ...process.env, PYTHONUNBUFFERED: '1' } as NodeJS.ProcessEnv;
  if (meta.gfxOverride) env.HSA_OVERRIDE_GFX_VERSION = '10.3.0';
  const child = spawn(pythonBin(root), ['-m', 'tinyllm.cli', 'resume', '--blob', blob, '--status', statusPath, '--device', device], {
    cwd: root,
    env
  });
  writeFileSync(
    path.join(dir, 'meta.json'),
    JSON.stringify({ ...meta, pid: child.pid, resumedAt: new Date().toISOString() }, null, 2)
  );
  watchChild(dir, statusPath, child);
  return {
    ok: true as const,
    run: { ...run, state: 'running', message: 'Resuming from the saved step.' }
  };
}

export function startRun(request: RunRequest) {
  const root = repoRoot();
  const busy = listRuns(root).find(
    (run) => run && run.id !== 'corpus' && (run.state === 'running' || run.state === 'paused' || run.state === 'starting')
  );
  if (busy) {
    const error =
      busy.state === 'paused'
        ? 'A training run is paused. Resume it before starting another.'
        : 'A training run is already in progress.';
    return { ok: false as const, status: 409, error, run: busy };
  }

  const phase = request.phase;
  if (phase !== 'pretrain' && phase !== 'finetune' && phase !== 'pipeline') {
    return { ok: false as const, status: 400, error: 'Phase must be pretrain, finetune, or pipeline.' };
  }
  const data = datasetPaths(root);
  const trainBin = path.join(root, data.work, 'train.bin');
  if ((phase === 'pretrain' || phase === 'pipeline') && !existsSync(trainBin)) {
    return {
      ok: false as const,
      status: 400,
      error: `Prepare the dataset first. ${data.work}/train.bin is not there yet.`
    };
  }
  const init = 'checkpoints/pretrain.pt';
  if (phase === 'finetune' && !existsSync(path.join(root, init))) {
    return {
      ok: false as const,
      status: 400,
      error:
        'Pretrain with the current 8192-piece tokenizer first. checkpoints/pretrain.pt is not there yet. The earlier book pass in checkpoints/corpus.pt used a 256-byte vocabulary and is not loaded into this model.'
    };
  }

  const steps = clampInt(request.steps, 1, MAX_STEPS);
  const finetuneSteps = clampInt(request.finetuneSteps, 1, MAX_STEPS);
  const batchSize = clampInt(request.batchSize, 1, 64);
  const seed = clampInt(request.seed, 0, 1_000_000);
  const learningRate = Number.isFinite(request.learningRate)
    ? Math.min(0.1, Math.max(1e-5, request.learningRate))
    : 3e-3;
  const device = request.device === 'cpu' || request.device === 'gpu' ? request.device : 'auto';

  const id = `${new Date().toISOString().replace(/[:.]/g, '-')}-${Math.random().toString(16).slice(2, 6)}`;
  const dir = path.join(runsDir(root), id);
  mkdirSync(dir, { recursive: true });
  mkdirSync(path.join(root, 'checkpoints'), { recursive: true });
  const statusPath = path.join(dir, 'status.json');
  const total = phase === 'finetune' ? finetuneSteps : steps;

  const shared = [
    '--batch-size',
    String(batchSize),
    '--learning-rate',
    String(learningRate),
    '--seed',
    String(seed),
    '--device',
    device,
    '--status',
    statusPath
  ];
  const sft = path.join(root, data.work, 'sft.jsonl');
  const dataArgs = existsSync(trainBin) ? ['--data', data.work] : [];
  const sftArgs = existsSync(sft) ? ['--sft', path.join(data.work, 'sft.jsonl')] : [];
  const args = ['-m', 'tinyllm.cli'];
  if (phase === 'pretrain') {
    args.push(
      'pretrain',
      '--steps',
      String(steps),
      '--sample-every',
      '40',
      '--out',
      'checkpoints/pretrain.pt',
      ...dataArgs,
      ...shared
    );
  } else if (phase === 'finetune') {
    args.push(
      'finetune',
      '--steps',
      String(finetuneSteps),
      '--sample-every',
      '10',
      '--init',
      init,
      '--out',
      'checkpoints/sft.pt',
      ...sftArgs,
      ...shared
    );
  } else {
    args.push(
      'pipeline',
      '--steps',
      String(steps),
      '--finetune-steps',
      String(finetuneSteps),
      '--sample-every',
      '40',
      '--out',
      'checkpoints',
      ...dataArgs,
      ...sftArgs,
      ...shared
    );
  }

  writeFileSync(
    statusPath,
    JSON.stringify(
      {
        state: 'running',
        phase,
        step: 0,
        totalSteps: total,
        metrics: [],
        probes: [],
        tokensTrained: 0,
        tokensSeen: 0,
        elapsedS: 0,
        vocabSize: 8192,
        dataSource: phase === 'finetune' ? path.join(data.work, 'sft.jsonl') : path.join(data.work, 'train.bin'),
        compute: null,
        message: 'Loading PyTorch and building the model. Token, power, and temperature totals update on each step.'
      },
      null,
      2
    )
  );

  const env = { ...process.env, PYTHONUNBUFFERED: '1' } as NodeJS.ProcessEnv;
  if (request.gfxOverride) {
    env.HSA_OVERRIDE_GFX_VERSION = '10.3.0';
  }
  const child: ChildProcess = spawn(pythonBin(root), args, {
    cwd: root,
    env
  });
  writeFileSync(
    path.join(dir, 'meta.json'),
    JSON.stringify(
      { pid: child.pid, phase, device, gfxOverride: Boolean(request.gfxOverride), startedAt: new Date().toISOString() },
      null,
      2
    )
  );
  watchChild(dir, statusPath, child);

  return { ok: true as const, id };
}

export function sampleCheckpoint(input: {
  checkpoint: string;
  prompt: string;
  temperature: number;
  maxNewTokens: number;
  topK: number;
}) {
  const root = repoRoot();
  const abs = checkpointAllowed(root, input.checkpoint);
  if (!abs) {
    return { ok: false as const, status: 404, error: 'That checkpoint is not in checkpoints/.' };
  }
  const prompt = input.prompt.slice(0, 500);
  const maxNewTokens = clampInt(input.maxNewTokens, 1, 80);
  const topK = clampInt(input.topK, 0, 64);
  const temperature = Number.isFinite(input.temperature) ? Math.min(2, Math.max(0, input.temperature)) : 0;
  const result = spawnSync(
    pythonBin(root),
    [
      '-m',
      'tinyllm.cli',
      'sample',
      '--ckpt',
      path.relative(root, abs),
      '--prompt',
      prompt,
      '--max-new-tokens',
      String(maxNewTokens),
      '--temperature',
      String(temperature),
      '--top-k',
      String(topK),
      '--json'
    ],
    { cwd: root, encoding: 'utf8', timeout: 120_000 }
  );
  if (result.status !== 0) {
    const raw = result.stderr || result.stdout || 'Sampling failed.';
    const error = raw
      .split('\n')
      .filter((line) => !line.includes('Failed to initialize NumPy') && !line.includes('tensor_numpy.cpp') && !line.includes('_conversion_method_template'))
      .join('\n')
      .trim();
    return {
      ok: false as const,
      status: 500,
      error: (error || 'Sampling failed.').slice(-1000)
    };
  }
  const parsed = JSON.parse(result.stdout) as { prompt: string; text: string; vocabSize?: number };
  return { ok: true as const, ...parsed, checkpoint: input.checkpoint };
}

export function readHarbor(root = repoRoot()) {
  const file = path.join(root, 'tinyllm', 'harbor.json');
  const harbor = JSON.parse(readFileSync(file, 'utf8')) as {
    facts: { subject: string; value: string }[];
    paraphrases: { prompt: string; completion: string }[];
  };
  const sentences = harbor.facts.map((row) => `The ${row.subject} is ${row.value}.`);
  const pairs = harbor.facts.map((row) => ({
    prompt: `Question: What is the ${row.subject}?\nAnswer:`,
    completion: ` ${row.value}\n`
  }));
  return {
    facts: harbor.facts,
    paraphrases: harbor.paraphrases,
    prose: sentences.join(' '),
    pairs: pairs.concat(harbor.paraphrases)
  };
}

function stageResult(command: string[], input?: string, timeout = 60_000) {
  const root = repoRoot();
  const result = spawnSync(pythonBin(root), ['-m', 'tinyllm.stages', ...command], {
    cwd: root,
    encoding: 'utf8',
    timeout,
    input
  });
  if (result.status !== 0) {
    return { ok: false as const, status: 400, error: (result.stderr || result.stdout || 'Stage failed.').slice(-1000) };
  }
  return { ok: true as const, report: JSON.parse(result.stdout) as Record<string, unknown> };
}

export function dataReport() {
  const data = datasetPaths();
  const result = stageResult(['report', '--raw', data.raw, '--work', data.work]);
  if (!result.ok) return result;
  return { ok: true as const, report: { ...result.report, dataset: data } };
}

export function runGather() {
  const data = datasetPaths();
  return stageResult(['gather', '--raw', data.raw, '--work', data.work]);
}

export function runClean() {
  const data = datasetPaths();
  return stageResult(['clean', '--work', data.work]);
}

export function runPrepare() {
  const data = datasetPaths();
  return stageResult(['prepare', '--work', data.work, '--block-size', '128'], undefined, 180_000);
}

export function ingestDocument(name: string, text: string) {
  const data = datasetPaths();
  return stageResult(['ingest', '--raw', data.raw, '--name', name.slice(0, 80)], text.slice(0, 200_000));
}

let deviceCache: { at: number; gfx: boolean; body: string } | null = null;

export function probeDevice(gfxOverride: boolean) {
  const now = Date.now();
  if (deviceCache && deviceCache.gfx === gfxOverride && now - deviceCache.at < 20_000) {
    return { ok: true as const, device: JSON.parse(deviceCache.body) as Record<string, unknown> };
  }
  const root = repoRoot();
  const env = { ...process.env, PYTHONUNBUFFERED: '1' } as NodeJS.ProcessEnv;
  if (gfxOverride) env.HSA_OVERRIDE_GFX_VERSION = '10.3.0';
  const result = spawnSync(pythonBin(root), ['-m', 'tinyllm.device'], {
    cwd: root,
    encoding: 'utf8',
    timeout: 120_000,
    env
  });
  if (result.status !== 0) {
    return { ok: false as const, status: 500, error: (result.stderr || result.stdout || 'Device probe failed.').slice(-1000) };
  }
  deviceCache = { at: now, gfx: gfxOverride, body: result.stdout };
  return { ok: true as const, device: JSON.parse(result.stdout) as Record<string, unknown> };
}

function streamIds(file: Buffer, limit: number): { total: number; ids: number[] } {
  if (file.subarray(0, 4).toString('utf8') === 'TKv1') {
    const total = Math.floor((file.length - 4) / 2);
    const shown = Math.min(total, limit);
    const ids: number[] = [];
    for (let index = 0; index < shown; index += 1) ids.push(file.readUInt16LE(4 + index * 2));
    return { total, ids };
  }
  return { total: file.length, ids: [...file.subarray(0, limit)] };
}

export function trainExcerpt(root = repoRoot()) {
  const data = datasetPaths(root);
  const candidates = [`${data.work}/train.bin`, 'data/corpus/work/train.bin', 'data/work/train.bin'];
  for (const rel of candidates) {
    const abs = path.join(root, rel);
    if (!existsSync(abs)) continue;
    const stream = streamIds(readFileSync(abs), 240);
    return {
      path: rel,
      total: stream.total,
      bytes: stream.ids
    };
  }
  return null;
}

let geometryCache: { key: string; body: string } | null = null;

export function modelGeometry() {
  const root = repoRoot();
  // corpus.pt and older pretrain files are the 256-byte model. The stage projects
  // the current 8192-piece table instead of that checkpoint.
  const rel = '';
  const key = rel ? `${rel}:${statSync(path.join(root, rel)).mtimeMs}:axes` : 'init';
  if (geometryCache?.key === key) {
    return { ok: true as const, geometry: JSON.parse(geometryCache.body) as Record<string, unknown> };
  }
  const args = ['-m', 'tinyllm.inspect'];
  if (rel) args.push('--checkpoint', rel);
  const result = spawnSync(pythonBin(root), args, {
    cwd: root,
    encoding: 'utf8',
    timeout: 180_000,
    maxBuffer: 16_000_000,
    env: { ...process.env, PYTHONUNBUFFERED: '1' } as NodeJS.ProcessEnv
  });
  if (result.status !== 0) {
    return { ok: false as const, status: 500, error: (result.stderr || result.stdout || 'Could not project the model.').slice(-1200) };
  }
  geometryCache = { key, body: result.stdout };
  return { ok: true as const, geometry: JSON.parse(result.stdout) as Record<string, unknown> };
}
