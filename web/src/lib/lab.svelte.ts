import { defaultShape, type ModelShape } from './params.ts';

export const lab = $state({
  shape: { ...defaultShape } as ModelShape,
  prompt: 'Question: What is the lighthouse lamp?\nAnswer:',
  completion: ' copper\n',
  temperature: 0.8
});
