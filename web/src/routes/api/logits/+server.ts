import { json } from '@sveltejs/kit';
import { nextLogits } from '#lib/server/jobs.ts';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ request }) => {
  const body = (await request.json()) as Partial<{ checkpoint: string; prompt: string }>;
  try {
    const result = nextLogits(String(body.checkpoint ?? ''), String(body.prompt ?? ''));
    if (!result.ok) return json({ error: result.error }, { status: result.status });
    return json(result.distribution);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Could not read the logits.';
    return json({ error: message }, { status: 500 });
  }
};
