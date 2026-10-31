# TinyLLM lab

A small, readable pipeline for training a decoder-only language model of about **one million parameters**, plus a SvelteKit lab that walks through the same steps.

The model is not a general chatbot. It memorizes a short invented harbor (the lighthouse lamp is copper, the captain is Ivo) and then learns to answer questions about it. That is small enough to read the whole training loop and to finish on a CPU.

## What the pipeline does

1. **Bytes.** Text is UTF-8. Each byte is a token, so the vocabulary is 256 and there is no separate merge table.
2. **Shape.** Five pre-norm Transformer blocks, width 128, four heads, context 128. The token embedding and the output head share one matrix. The default configuration is **1,035,008** parameters.
3. **Pretrain.** Predict the next byte of the harbor prose. Loss is cross-entropy. A uniform model sits at `ln(256) ≈ 5.545`; anything below that means the model is less surprised than chance.
4. **Finetune.** Same weights, new examples. The prompt is still in the context, but those positions are labeled `-100` and dropped from the loss. Only the answer tokens are supervised. The question format never appears in the pretraining text, so a change in the samples is the finetune working.
5. **Sample.** Greedy decoding (`temperature = 0`) or temperature sampling with optional top-k.

## Layout

| Path | Role |
| --- | --- |
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

Open the dev server (port 5173). The pages are the notes, the shape explorer, the byte window, pretraining, finetuning, and sampling. Training buttons call this repo's virtualenv (`.venv/bin/python`) and poll `runs/<id>/status.json`. The server refuses a second run while one is alive and caps a lab run at 400 steps.

## Reading a run

- The chart's dashed line is chance, `ln(vocab)`.
- Perplexity is `exp(loss)`. Chance for bytes is 256. A perplexity of 10 means the model is effectively choosing among about ten next bytes.
- Probes run during training. "The lighthouse lamp is" should start to finish the memorized fact. The `Question:` probes should stay aimless after pretraining and tighten after finetuning, because that format is new.
- Finetune usually wants a lower learning rate than pretraining. The lab defaults to `3e-3` then `1e-3`.

## Tests

`uv run pytest` checks the parameter formula against `model.parameters()`, the prompt mask, the learning-rate schedule, and a few optimizer steps on a smaller model.
