import { json } from '@sveltejs/kit';
import { ingestDocument, runGather } from '#lib/server/jobs.ts';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ request }) => {
  const body = (await request.json().catch(() => ({}))) as { name?: string; text?: string };
  if (body.text) {
    const saved = ingestDocument(String(body.name ?? 'note'), String(body.text));
    if (!saved.ok) return json({ error: saved.error }, { status: saved.status });
  }
  const result = runGather();
  if (!result.ok) return json({ error: result.error }, { status: result.status });
  return json(result.report);
};
