"""Offline command line workflow: prepare, inspect, train, evaluate, generate."""

import argparse
from dataclasses import asdict
import glob
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import load_model

from .checkpoint import read_checkpoint_metadata
from .config import ExperimentConfig, load_config
from .data import PreparedData, prepare_data, validation_batches
from .evaluation import evaluate
from .generation import generate
from .metrics import estimate_transformer_flops
from .model import TransformerLM, count_parameters, transformer_parameter_formula
from .objectives import CausalLanguageModeling
from .runtime import setup_runtime
from .trainer import Trainer


def _positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parser():
    root = argparse.ArgumentParser(prog="llm2", description="LLM2-Lab controlled experiments")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="split documents and train tokenizer offline")
    prepare.add_argument("--input", nargs="+", required=True, help="UTF-8 paths or glob patterns")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--tokenizer", choices=("byte", "bpe"), default="bpe")
    prepare.add_argument("--vocab-size", type=_positive, default=8192)
    prepare.add_argument("--val-fraction", type=float, default=0.1)
    prepare.add_argument("--seed", type=int, default=42)
    inspect = commands.add_parser("inspect", help="calculate parameter count and estimate FLOPs")
    inspect.add_argument("--config", required=True)
    train = commands.add_parser("train", help="train from random weights or resume trusted state")
    train.add_argument("--config", required=True)
    train.add_argument("--resume", help="trusted complete checkpoint directory")
    train.add_argument("--max-steps", type=_positive, help="override stop step, keeping the fixed LR schedule")
    train.add_argument("--until-step", type=_positive, help="checkpoint and stop early for a staged run")
    train.add_argument("--output-dir", help="override run directory (relative to current working directory)")
    for name in ("evaluate", "generate"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--checkpoint", required=True)
        cmd.add_argument("--data", help="relocated prepared data; manifest must match checkpoint")
        cmd.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        cmd.add_argument("--precision", choices=("fp32", "bf16"))
        if name == "evaluate":
            cmd.add_argument("--max-batches", type=_positive, help="default: all full validation windows")
        else:
            cmd.add_argument("--prompt", default="")
            cmd.add_argument("--max-new-tokens", type=_positive, default=64)
            cmd.add_argument("--temperature", type=float, default=1.0, help="0 selects greedy decoding")
            cmd.add_argument("--top-k", type=_positive, default=40)
            cmd.add_argument("--seed", type=int, default=42)
    return root


def _inference(args):
    # Metadata and safetensors only; does not unpickle the optimizer/RNG state.
    metadata = read_checkpoint_metadata(args.checkpoint)
    cfg = ExperimentConfig.from_dict(metadata["config"])
    cfg.train.device = args.device
    cfg.train.precision = args.precision or ("bf16" if args.device == "cuda" else "fp32")
    cfg.train.__post_init__()
    device = setup_runtime(cfg.train)
    data = PreparedData(args.data or cfg.data.path)
    if data.fingerprint != metadata["dataset_sha256"]:
        raise ValueError("dataset fingerprint differs from checkpoint")
    if metadata["model_class"] != "llm2.model.TransformerLM" or metadata["objective_class"] != "llm2.objectives.CausalLanguageModeling":
        raise ValueError("CLI evaluation/generation supports the baseline; use the Python interfaces for custom models")
    model = TransformerLM(cfg.model).to(device)
    load_model(model, str(Path(args.checkpoint) / "model.safetensors"), device=str(device))
    return cfg, model, data, device


def main(argv=None):
    # Windows pipes may default to a legacy code page, while generated text is arbitrary Unicode.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    try:
        if args.command == "prepare":
            paths = sorted({path for pattern in args.input for path in glob.glob(str(Path(pattern).expanduser()))})
            if not paths:
                raise ValueError("no input files matched")
            result = prepare_data(paths, args.output, kind=args.tokenizer, vocab_size=args.vocab_size,
                                  val_fraction=args.val_fraction, seed=args.seed)
        elif args.command == "inspect":
            cfg = load_config(args.config)
            model = TransformerLM(cfg.model)
            result = {"parameters_calculated": count_parameters(model),
                      "parameters_formula": transformer_parameter_formula(cfg.model),
                      "flops_estimate": asdict(estimate_transformer_flops(cfg.model, cfg.data.seq_len))}
        elif args.command == "train":
            cfg = load_config(args.config)
            if args.max_steps is not None:
                cfg.train.max_steps = args.max_steps
            if args.output_dir:
                cfg.train.output_dir = str(Path(args.output_dir).resolve())
            cfg = ExperimentConfig.from_dict(cfg.to_dict())
            with Trainer(cfg, resume=args.resume) as trainer:
                metrics = trainer.fit(args.until_step)
                result = {**metrics, "checkpoint": str(trainer.last_checkpoint)}
        else:
            cfg, model, data, device = _inference(args)
            if args.command == "evaluate":
                result = evaluate(model, CausalLanguageModeling(),
                                  validation_batches(data.arrays["val"], cfg.data.seq_len,
                                                     cfg.data.eval_batch_size, args.max_batches, device),
                                  device=device, precision=cfg.train.precision,
                                  deterministic=cfg.train.deterministic)
            else:
                ids = data.tokenizer.encode(args.prompt) or [data.tokenizer.eos_id]
                output = generate(
                    model, torch.tensor([ids], device=device), max_seq_len=cfg.model.max_seq_len,
                    max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
                    eos_id=data.tokenizer.eos_id, vocab_size=data.tokenizer.vocab_size, seed=args.seed,
                    precision=cfg.train.precision, deterministic=cfg.train.deterministic,
                )
                print(data.tokenizer.decode(output[0].tolist()))
                return
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (ValueError, FileNotFoundError, FileExistsError, FloatingPointError, RuntimeError) as exc:
        arg_parser.exit(1, f"llm2: {exc}\n")

