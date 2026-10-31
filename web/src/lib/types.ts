export type Metric = {
  step: number;
  loss: number;
  lr: number;
  phase: string;
};

export type Probe = {
  prompt: string;
  completion: string;
};

export type RunStatus = {
  id?: string;
  state: string;
  phase?: string;
  step?: number;
  totalSteps?: number;
  loss?: number;
  lr?: number;
  metrics?: Metric[];
  probes?: Probe[];
  checkpoint?: string;
  message?: string;
};

export type Harbor = {
  facts: { subject: string; value: string }[];
  paraphrases: { prompt: string; completion: string }[];
  prose: string;
  pairs: { prompt: string; completion: string }[];
};
