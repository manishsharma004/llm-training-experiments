import { json } from '@sveltejs/kit';
import { probeDevice } from '#lib/server/jobs.ts';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = ({ url }) => {
  const result = probeDevice(url.searchParams.get('gfx') === '1');
  if (!result.ok) return json({ error: result.error }, { status: result.status });
  return json(result.device);
};
