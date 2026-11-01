import { json } from '@sveltejs/kit';
import { latestRun, listRuns, repoRoot, startRun, type Phase } from '#lib/server/jobs.ts';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = () => {
  const root = repoRoot();
  const all = listRuns(root).filter((run) => run !== null);
  const corpus = all.find((run) => run.id === 'corpus');
  const recent = all.filter((run) => run.id !== 'corpus').slice(0, 8);
  return json({ run: latestRun(root), runs: corpus ? [...recent, corpus] : recent });
};

export const POST: RequestHandler = async ({ request }) => {
  const body = (await request.json()) as Partial<{
    phase: Phase;
    steps: number;
    finetuneSteps: number;
    seed: number;
    batchSize: number;
    learningRate: number;
    device: 'auto' | 'cpu' | 'gpu';
    gfxOverride: boolean;
  }>;
  try {
    const result = startRun({
      phase: body.phase ?? 'pretrain',
      steps: Number(body.steps ?? 20),
      finetuneSteps: Number(body.finetuneSteps ?? 8),
      seed: Number(body.seed ?? 7),
      batchSize: Number(body.batchSize ?? 16),
      learningRate: Number(body.learningRate ?? 3e-3),
      device: body.device ?? 'auto',
      gfxOverride: Boolean(body.gfxOverride)
    });
    if (!result.ok) {
      return json({ error: result.error, run: result.run ?? null }, { status: result.status });
    }
    return json({ id: result.id });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Could not start training.';
    return json({ error: message }, { status: 500 });
  }
};
