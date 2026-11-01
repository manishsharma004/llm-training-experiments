/** The checkpoint forward the sample page and its 3D stage share. */

export const live = $state({
  checkpoint: '',
  prompt: '',
  topK: 0,
  logits: [] as number[],
  vocabSize: 0,
  loading: false,
  error: ''
});
