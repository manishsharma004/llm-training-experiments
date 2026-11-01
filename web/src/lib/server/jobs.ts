import { spawn, spawnSync, type ChildProcess } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
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

const MAX_STEPS = 2000;

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
  const dir = path.join(runsDir(root), id);
  const status = readJson(path.join(dir, 'status.json'));
  const meta = readJson(path.join(dir, 'meta.json'));
  if (!status && !meta) return null;
  const pid = Number(meta?.pid ?? 0);
  let state = String(status?.state ?? 'unknown');
  let message = typeof status?.message === 'string' ? status.message : '';
  if ((state === 'running' || state === 'starting') && pid && !alive(pid)) {
    state = 'error';
    message = message || 'The training process stopped before writing a final status.';
  }
  return { id, ...(status ?? {}), state, message, pid };
}

export function listRuns(root: string) {
  const dir = runsDir(root);
  if (!existsSync(dir)) return [] as ReturnType<typeof loadRun>[];
  const ids = readdirSync(dir)
    .filter((name) => existsSync(path.join(dir, name, 'status.json')))
    .sort()
    .reverse();
  return ids.map((id) => loadRun(root, id)).filter((run) => run !== null);
}

export function latestRun(root: string) {
  return listRuns(root)[0] ?? null;
}

export function listCheckpoints(root: string) {
  const dir = path.join(root, 'checkpoints');
  if (!existsSync(dir)) return [] as { path: string }[];
  const files: string[] = [];
  const walk = (current: string) => {
    for (const name of readdirSync(current, { withFileTypes: true })) {
      const abs = path.join(current, name.name);
      if (name.isDirectory()) walk(abs);
      else if (name.name.endsWith('.pt')) files.push(abs);
    }
  };
  walk(dir);
  return files
    .map((abs) => ({ path: path.relative(root, abs).split(path.sep).join('/') }))
    .sort((a, b) => a.path.localeCompare(b.path));
}

function checkpointAllowed(root: string, rel: string): string | null {
  if (!rel.endsWith('.pt') || rel.includes('..')) return null;
  const abs = path.resolve(root, rel);
  const base = path.join(root, 'checkpoints') + path.sep;
  if (!abs.startsWith(base)) return null;
  if (!existsSync(abs)) return null;
  return abs;
}

export function startRun(request: RunRequest) {
  const root = repoRoot();
  const active = latestRun(root);
  if (active && active.state === 'running') {
    return { ok: false as const, status: 409, error: 'A training run is already in progress.', run: active };
  }

  const phase = request.phase;
  if (phase !== 'pretrain' && phase !== 'finetune' && phase !== 'pipeline') {
    return { ok: false as const, status: 400, error: 'Phase must be pretrain, finetune, or pipeline.' };
  }
  if ((phase === 'pretrain' || phase === 'pipeline') && !existsSync(path.join(root, 'data', 'work', 'train.bin'))) {
    return {
      ok: false as const,
      status: 400,
      error: 'Prepare the dataset first. data/work/train.bin is not there yet.'
    };
  }
  if (phase === 'finetune' && !existsSync(path.join(root, 'checkpoints', 'pretrain.pt'))) {
    return {
      ok: false as const,
      status: 400,
      error: 'Pretrain first. checkpoints/pretrain.pt is not there yet.'
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
  const sft = path.join(root, 'data', 'work', 'sft.jsonl');
  const dataArgs = existsSync(path.join(root, 'data', 'work', 'train.bin')) ? ['--data', 'data/work'] : [];
  const sftArgs = existsSync(sft) ? ['--sft', 'data/work/sft.jsonl'] : [];
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
      'checkpoints/pretrain.pt',
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
        message: 'Loading PyTorch and building the model.'
      },
      null,
      2
    )
  );

  const stderrChunks: Buffer[] = [];
  const env = { ...process.env, PYTHONUNBUFFERED: '1' } as NodeJS.ProcessEnv;
  if (request.gfxOverride) {
    env.HSA_OVERRIDE_GFX_VERSION = '10.3.0';
  }
  const child: ChildProcess = spawn(pythonBin(root), args, {
    cwd: root,
    env
  });
  child.stderr?.on('data', (chunk: Buffer) => {
    stderrChunks.push(chunk);
  });
  writeFileSync(
    path.join(dir, 'meta.json'),
    JSON.stringify({ pid: child.pid, phase, startedAt: new Date().toISOString() }, null, 2)
  );

  child.on('exit', (code) => {
    const current = readJson(statusPath) ?? {};
    const text = Buffer.concat(stderrChunks).toString('utf8').trim();
    if (text) writeFileSync(path.join(dir, 'stderr.txt'), text);
    if (code === 0 && current.state === 'done') return;
    writeFileSync(
      path.join(dir, 'status.json'),
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
    return {
      ok: false as const,
      status: 500,
      error: (result.stderr || result.stdout || 'Sampling failed.').slice(-1000)
    };
  }
  const parsed = JSON.parse(result.stdout) as { prompt: string; text: string };
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

function stageResult(command: string[], input?: string) {
  const root = repoRoot();
  const result = spawnSync(pythonBin(root), ['-m', 'tinyllm.stages', ...command], {
    cwd: root,
    encoding: 'utf8',
    timeout: 60_000,
    input
  });
  if (result.status !== 0) {
    return { ok: false as const, status: 400, error: (result.stderr || result.stdout || 'Stage failed.').slice(-1000) };
  }
  return { ok: true as const, report: JSON.parse(result.stdout) as Record<string, unknown> };
}

export function dataReport() {
  return stageResult(['report', '--raw', 'data/raw', '--work', 'data/work']);
}

export function runGather() {
  return stageResult(['gather', '--raw', 'data/raw', '--work', 'data/work']);
}

export function runClean() {
  return stageResult(['clean', '--work', 'data/work']);
}

export function runPrepare() {
  return stageResult(['prepare', '--work', 'data/work', '--block-size', '128']);
}

export function ingestDocument(name: string, text: string) {
  return stageResult(['ingest', '--raw', 'data/raw', '--name', name.slice(0, 80)], text.slice(0, 200_000));
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
