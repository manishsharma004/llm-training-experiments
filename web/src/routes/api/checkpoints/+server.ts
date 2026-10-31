import { json } from '@sveltejs/kit';
import { listCheckpoints, repoRoot } from '#lib/server/jobs.ts';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = () => {
  return json({ checkpoints: listCheckpoints(repoRoot()) });
};
