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
  device?: string;
  arch?: string;
  parameters?: number;
  dataSource?: string;
  tokensPerSec?: number;
  tokensTrained?: number;
  tokensSeen?: number;
  elapsedS?: number;
  valLoss?: number | null;
  compute?: ComputeReport | null;
};

export type RangeStat = {
  min: number;
  avg: number;
  max: number;
  last: number;
};

export type ComputeReport = {
  samples?: number;
  elapsed_s?: number;
  power_w?: RangeStat | null;
  power_cap_w?: number | null;
  energy_wh?: number;
  vram_bytes?: RangeStat | null;
  vram_total_bytes?: number | null;
  process_vram_bytes?: RangeStat | null;
  temperature_c?: Record<string, RangeStat>;
};

export type Harbor = {
  facts: { subject: string; value: string }[];
  paraphrases: { prompt: string; completion: string }[];
  prose: string;
  pairs: { prompt: string; completion: string }[];
};
