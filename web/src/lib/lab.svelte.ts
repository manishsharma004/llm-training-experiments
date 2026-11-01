import { defaultShape, type ModelShape } from './params.ts';

export const lab = $state({
  shape: { ...defaultShape } as ModelShape,
  prompt: 'Question: What is the lighthouse lamp?\nAnswer:',
  completion: ' copper\n',
  temperature: 0.8
});

const hydratedFlag = '__bookLabHydrated';

let markHydrated: (saved: boolean) => void = () => undefined;

/** Resolves once IndexedDB has been applied. The boolean is true when a saved lab was found. */
export const labHydrated: Promise<boolean> = new Promise((resolve) => {
  markHydrated = resolve;
  if (hydratedFlag in globalThis) resolve(Boolean((globalThis as Record<string, unknown>)[hydratedFlag]));
});

export function finishHydration(saved: boolean): void {
  (globalThis as Record<string, unknown>)[hydratedFlag] = saved;
  markHydrated(saved);
}
