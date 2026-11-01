import { json } from '@sveltejs/kit';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { repoRoot } from '#lib/server/jobs.ts';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = () => {
  const file = path.join(repoRoot(), 'tinyllm', 'bpe.json');
  const payload = JSON.parse(readFileSync(file, 'utf8')) as Record<string, unknown>;
  return json(payload);
};
