import { json } from '@sveltejs/kit';
import { readHarbor } from '#lib/server/jobs.ts';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = () => {
  return json(readHarbor());
};
