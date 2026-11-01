import { json } from '@sveltejs/kit';
import { dataReport } from '#lib/server/jobs.ts';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = () => {
  const result = dataReport();
  if (!result.ok) return json({ error: result.error }, { status: result.status });
  return json(result.report);
};
