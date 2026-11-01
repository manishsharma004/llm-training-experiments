# TinyLLM lab

A small, readable pipeline for training a decoder-only language model of about **one million parameters**, plus a SvelteKit lab that walks through the same steps.

The model is not a general chatbot. It memorizes a short invented harbor (the lighthouse lamp is copper, the captain is Ivo) and then learns to answer questions about it. That is small enough to read the whole training loop and to finish on a CPU.

## Data, then weights

The lab uses `data/corpus/work/train.bin` when that book stream is present, and `data/work/train.bin` otherwise. A stream does not exist until the three data stages have run.

```bash
uv run python -m tinyllm.stages gather --raw data/raw --work data/work
uv run python -m tinyllm.stages clean --work data/work
uv run python -m tinyllm.stages prepare --work data/work
uv run tinyllm device
uv run tinyllm pretrain --data data/work --device auto --steps 200 --out checkpoints/pretrain.pt
uv run tinyllm finetune --init checkpoints/pretrain.pt --sft data/work/sft.jsonl --device auto --steps 80
```

`data/raw` ships with messy harbor notes: a duplicate file, an HTML scrape, and a junk file, so the clean report has something to reject. Question pairs live in `data/raw/sft.jsonl` and are copied into `data/work/sft.jsonl` by prepare.

The lab pages Gather, Clean, Prepare, and Train call these same commands.

## RX 5500 (4GB, gfx1012)

The preset is 1,035,008 parameters in fp32. Weights, gradients, and AdamW states are about 16MB, so the 4GB card is not short of memory for this model. PyTorch on AMD uses the CUDA API (`torch.cuda`) via ROCm.

The RX 5500 is Navi 14. Generic ROCm wheels often omit `gfx1012`, so the card can be in the machine while `torch.cuda.is_available()` is false.

- Preferred: install AMD's PyTorch build for `gfx1012`. From the repo root, after `uv sync --group dev`:

```bash
bash scripts/install-rx5500.sh
```

That puts `torch==2.14.0+rocm10.1.0` and `amd-torch-device-gfx1012` into `.venv`. It is a local replacement of the CPU wheel in `uv.lock`. A later `uv sync` will put the CPU wheel back. The lab then lists the RX 5500 and trains on it when Device is Auto or GPU.
- Fallback: a ROCm wheel that contains gfx1030 kernels can be pointed at the card by checking **RX 5500 compatibility** in the lab. That sets `HSA_OVERRIDE_GFX_VERSION=10.3.0` for the training process only. Those kernels were compiled for a different chip and can be unstable.
- If you choose GPU and nothing is visible, the run stops with the reason. Auto falls back to the CPU.

`tinyllm device` prints the PCI scan, whether PyTorch sees a GPU, and the memory estimate.

## What the pipeline does

1. **Tokens.** Byte-level BPE, vocabulary 8192, trained on the cleaned book corpus. The merge table is `tinyllm/bpe.json`. Streams are `TKv1` plus little-endian uint16 ids. Rebuild the table with `scripts/train_bpe.py` after `pip install tokenizers` in `.venv` (do not `uv sync` if the AMD torch wheel is installed).
2. **Shape.** Five pre-norm Transformer blocks, width 128, four heads, context 128. The token embedding and the output head share one matrix. The default configuration is **2,050,816** parameters.
3. **Pretrain.** Predict the next token of the book stream when `data/corpus/work/train.bin` is present. Loss is cross-entropy. A uniform model sits at `ln(8192) ≈ 9.011`; anything below that means the model is less surprised than chance.
4. **Finetune.** Same weights, new examples. The prompt is still in the context, but those positions are labeled `-100` and dropped from the loss. Only the answer tokens are supervised. The question format never appears in the pretraining text, so a change in the samples is the finetune working.
5. **Sample.** Greedy decoding (`temperature = 0`) or temperature sampling with optional top-k.

## Layout

| Path | Role |
| --- | --- |
| `tinyllm/stages.py` | Gather, clean, and prepare |
| `tinyllm/device.py` | GPU probe, RX 5500 notes, memory estimate |
| `tinyllm/model.py` | The Transformer |
| `tinyllm/params.py` | Parameter breakdown (mirrored in `web/src/lib/params.ts`) |
| `tinyllm/data.py` | Pretrain windows and the finetune loss mask |
| `tinyllm/train.py` | AdamW, warmup, cosine decay, gradient clipping |
| `tinyllm/cli.py` | `params`, `pretrain`, `finetune`, `pipeline`, `sample` |
| `tinyllm/harbor.json` | The facts both the trainer and the lab read |
| `web/` | SvelteKit lab |

## Train from the terminal

```bash
uv sync --group dev
uv run tinyllm params
uv run pytest
uv run tinyllm pipeline --steps 200 --finetune-steps 80 --out checkpoints
uv run tinyllm sample --ckpt checkpoints/sft.pt --prompt "Question: What is the lighthouse lamp?
Answer:"
```

Checkpoints land in `checkpoints/pretrain.pt` and `checkpoints/sft.pt` when `--out` is the `checkpoints` directory. They are gitignored. During `pipeline`, a learning rate above `1e-3` is lowered to `1e-3` for the finetune half.

Useful knobs: `--steps`, `--finetune-steps`, `--batch-size`, `--learning-rate`, `--seed`, `--n-layer`, `--n-embd`. `tinyllm params` prints the resulting parameter count before you spend the training time.

## Lab

```bash
cd web
bun install
bun run dev
```

Open the dev server (port 5173, or another free port). The pages follow the pipeline: notes, gather, clean, prepare, shape, train, finetune, and sample. Training buttons call this repo's virtualenv (`.venv/bin/python`) and poll `runs/<id>/status.json`. The server refuses a second run while one is alive and caps a lab run at 2000 steps. The step box starts at 20. Pretraining uses the book stream when `data/corpus/work/train.bin` is present.

## Reading a run

- The chart's dashed line is chance, `ln(vocab)`.
- Perplexity is `exp(loss)`. Chance for this vocabulary is 8192. A perplexity of 10 means the model is effectively choosing among about ten next pieces.
- Probes run during training. "The lighthouse lamp is" should start to finish the memorized fact. The `Question:` probes should stay aimless after pretraining and tighten after finetuning, because that format is new.
- Finetune usually wants a lower learning rate than pretraining. The lab defaults to `3e-3` then `1e-3`.

## Tests

`uv run pytest` checks the parameter formula against `model.parameters()`, the prompt mask, the learning-rate schedule, and a few optimizer steps on a smaller model.
