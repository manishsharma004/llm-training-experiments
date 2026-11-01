import { json } from '@sveltejs/kit';
import { runClean } from '#lib/server/jobs.ts';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = () => {
  const result = runClean();
  if (!result.ok) return json({ error: result.error }, { status: result.status });
  return json(result.report);
};
