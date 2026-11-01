import { json } from '@sveltejs/kit';
import { controlRun, loadRun, repoRoot } from '#lib/server/jobs.ts';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = ({ params }) => {
  const run = loadRun(repoRoot(), params.id);
  if (!run) return json({ error: 'Unknown run.' }, { status: 404 });
  return json({ run });
};

export const POST: RequestHandler = async ({ params, request }) => {
  const body = (await request.json().catch(() => ({}))) as { command?: string };
  const result = controlRun(params.id, String(body.command ?? ''));
  if (!result.ok) return json({ error: result.error, run: 'run' in result ? result.run : null }, { status: result.status });
  return json({ ok: true, run: result.run });
};
