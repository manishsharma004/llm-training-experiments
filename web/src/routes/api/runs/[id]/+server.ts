import { json } from '@sveltejs/kit';
import { loadRun, repoRoot } from '#lib/server/jobs.ts';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = ({ params }) => {
  const run = loadRun(repoRoot(), params.id);
  if (!run) return json({ error: 'Unknown run.' }, { status: 404 });
  return json({ run });
};
