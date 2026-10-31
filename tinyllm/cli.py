"""Command line entry for the training pipeline.

Examples
--------
tinyllm params
tinyllm pretrain --steps 200 --out checkpoints/pretrain.pt
tinyllm finetune --init checkpoints/pretrain.pt --steps 80 --out checkpoints/sft.pt
tinyllm sample --ckpt checkpoints/sft.pt --prompt "Question: What is the captain?\\nAnswer:"
tinyllm pipeline --steps 200 --finetune-steps 80 --out checkpoints/harbor
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tinyllm.config import ModelConfig, TrainConfig
from tinyllm.corpus import finetune_examples, prose_corpus
from tinyllm.params import about_one_million, breakdown
from tinyllm.sample import complete
from tinyllm.status import write_status
from tinyllm.tokenizer import describe
from tinyllm.train import finetune, pretrain


def _model_from_args(args: argparse.Namespace) -> ModelConfig:
    cfg = ModelConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    )
    cfg.validate()
    return cfg


def _train_from_args(args: argparse.Namespace, steps: int) -> TrainConfig:
    cfg = TrainConfig(
        steps=steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        seed=args.seed,
        log_every=args.log_every,
        sample_every=args.sample_every,
        corpus_repeats=args.corpus_repeats,
    )
    cfg.validate()
    return cfg


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=5)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--min-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--corpus-repeats", type=int, default=80)


def _print_summary(summary: dict) -> None:
    print(json.dumps({k: v for k, v in summary.items() if k != "metrics"}, indent=2))
    if summary.get("metrics"):
        last = summary["metrics"][-1]
        print(f"final step {last['step']} loss {last['loss']:.4f}")


def cmd_params(args: argparse.Namespace) -> int:
    cfg = _model_from_args(args)
    report = breakdown(cfg)
    report["about_one_million"] = about_one_million(cfg)
    report["config"] = cfg.to_dict()
    print(json.dumps(report, indent=2))
    return 0


def cmd_tokenize(args: argparse.Namespace) -> int:
    print(json.dumps({"tokens": describe(args.text)}, indent=2))
    return 0


def cmd_pretrain(args: argparse.Namespace) -> int:
    model_cfg = _model_from_args(args)
    train_cfg = _train_from_args(args, args.steps)
    status = Path(args.status) if args.status else None
    try:
        summary = pretrain(model_cfg, train_cfg, Path(args.out), status)
    except Exception as exc:
        if status:
            write_status(status, {"state": "error", "phase": "pretrain", "message": str(exc)})
        raise
    _print_summary(summary)
    return 0


def cmd_finetune(args: argparse.Namespace) -> int:
    train_cfg = _train_from_args(args, args.steps)
    status = Path(args.status) if args.status else None
    try:
        summary = finetune(Path(args.init), train_cfg, Path(args.out), status)
    except Exception as exc:
        if status:
            write_status(status, {"state": "error", "phase": "finetune", "message": str(exc)})
        raise
    _print_summary(summary)
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Pretrain, then finetune. Checkpoints land in the output directory."""

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pre_path = out_dir / "pretrain.pt"
    sft_path = out_dir / "sft.pt"
    status = Path(args.status) if args.status else None
    model_cfg = _model_from_args(args)
    try:
        pre_cfg = _train_from_args(args, args.steps)
        pre_summary = pretrain(model_cfg, pre_cfg, pre_path, status)
        ft_cfg = _train_from_args(args, args.finetune_steps)
        ft_cfg.warmup_steps = min(ft_cfg.warmup_steps, max(0, args.finetune_steps // 10))
        # Finetuning starts from useful weights, so keep the step size at or below 1e-3.
        if ft_cfg.learning_rate > 1e-3:
            ft_cfg.learning_rate = 1e-3
        ft_summary = finetune(pre_path, ft_cfg, sft_path, status)
    except Exception as exc:
        if status:
            write_status(status, {"state": "error", "phase": "pipeline", "message": str(exc)})
        raise
    print(
        json.dumps(
            {
                "pretrain_checkpoint": str(pre_path),
                "sft_checkpoint": str(sft_path),
                "pretrain_loss": pre_summary["loss"],
                "finetune_loss": ft_summary["loss"],
                "probes": ft_summary.get("probes", []),
            },
            indent=2,
        )
    )
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    text = complete(
        Path(args.ckpt),
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    if args.json:
        print(json.dumps({"prompt": args.prompt, "text": text}))
    else:
        print(text)
    return 0


def cmd_show_data(args: argparse.Namespace) -> int:
    if args.kind == "prose":
        print(prose_corpus(args.repeats))
    else:
        print(json.dumps(finetune_examples(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tinyllm", description="Train and inspect a ~1M parameter language model.")
    sub = parser.add_subparsers(dest="command", required=True)

    params = sub.add_parser("params", help="Print a parameter breakdown for a configuration.")
    _add_model_args(params)
    params.set_defaults(func=cmd_params)

    tokenize = sub.add_parser("tokenize", help="Show byte tokens for a string.")
    tokenize.add_argument("--text", required=True)
    tokenize.set_defaults(func=cmd_tokenize)

    pre = sub.add_parser("pretrain", help="Next-token pretraining on the harbor corpus.")
    _add_model_args(pre)
    _add_train_args(pre)
    pre.add_argument("--steps", type=int, default=200)
    pre.add_argument("--out", default="checkpoints/pretrain.pt")
    pre.add_argument("--status", default="")
    pre.set_defaults(func=cmd_pretrain)

    ft = sub.add_parser("finetune", help="Supervised finetune with prompt tokens masked out of the loss.")
    _add_train_args(ft)
    ft.add_argument("--steps", type=int, default=80)
    ft.add_argument("--init", required=True)
    ft.add_argument("--out", default="checkpoints/sft.pt")
    ft.add_argument("--status", default="")
    ft.set_defaults(func=cmd_finetune)

    pipe = sub.add_parser("pipeline", help="Pretrain and then finetune.")
    _add_model_args(pipe)
    _add_train_args(pipe)
    pipe.add_argument("--steps", type=int, default=200)
    pipe.add_argument("--finetune-steps", type=int, default=80)
    pipe.add_argument("--out", default="checkpoints/harbor")
    pipe.add_argument("--status", default="")
    pipe.set_defaults(func=cmd_pipeline)

    sample = sub.add_parser("sample", help="Complete a prompt from a checkpoint.")
    sample.add_argument("--ckpt", required=True)
    sample.add_argument("--prompt", required=True)
    sample.add_argument("--max-new-tokens", type=int, default=40)
    sample.add_argument("--temperature", type=float, default=0.0)
    sample.add_argument("--top-k", type=int, default=0)
    sample.add_argument("--json", action="store_true")
    sample.set_defaults(func=cmd_sample)

    data = sub.add_parser("data", help="Print the prose corpus or the finetune pairs.")
    data.add_argument("--kind", choices=("prose", "sft"), default="prose")
    data.add_argument("--repeats", type=int, default=1)
    data.set_defaults(func=cmd_show_data)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "top_k", None) == 0:
        args.top_k = None
    try:
        return args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
