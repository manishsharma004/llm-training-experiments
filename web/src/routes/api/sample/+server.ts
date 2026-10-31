import { json } from '@sveltejs/kit';
import { sampleCheckpoint } from '#lib/server/jobs.ts';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ request }) => {
  const body = (await request.json()) as Partial<{
    checkpoint: string;
    prompt: string;
    temperature: number;
    maxNewTokens: number;
    topK: number;
  }>;
  try {
    const result = sampleCheckpoint({
      checkpoint: String(body.checkpoint ?? ''),
      prompt: String(body.prompt ?? ''),
      temperature: Number(body.temperature ?? 0),
      maxNewTokens: Number(body.maxNewTokens ?? 24),
      topK: Number(body.topK ?? 0)
    });
    if (!result.ok) return json({ error: result.error }, { status: result.status });
    return json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Sampling failed.';
    return json({ error: message }, { status: 500 });
  }
};
