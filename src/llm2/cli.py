"""Offline command line workflow: prepare, inspect, train, evaluate, generate."""

import argparse
from dataclasses import asdict
import glob
import json
import os
from pathlib import Path
import sys
import time
from contextlib import ExitStack, nullcontext

import torch
import yaml
from safetensors.torch import load_model

from .checkpoint import read_checkpoint_metadata
from .config import ExperimentConfig, load_config
from .data import PreparedData, prepare_data, validation_batches, write_json
from .evaluation import evaluate
from .generation import generate
from .metrics import estimate_transformer_flops
from .model import TransformerLM, count_parameters, transformer_parameter_formula
from .objectives import CausalLanguageModeling
from .runtime import setup_runtime
from .trainer import Trainer
from .research.factory import build_backbone, build_objective
from .research.accounting import estimate_research_flops
from .checkpoint import class_name


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
    prepare.add_argument("--format", choices=("paragraphs", "documents-jsonl"), default="paragraphs")
    inspect = commands.add_parser("inspect", help="calculate parameter count and estimate FLOPs")
    inspect.add_argument("--config", required=True)
    inspect.add_argument("--meta", action="store_true", help="count large planned profiles without allocating weights")
    train = commands.add_parser("train", help="train from random weights or resume trusted state")
    train.add_argument("--config", required=True)
    train.add_argument("--resume", help="trusted complete checkpoint directory")
    train.add_argument("--max-steps", type=_positive, help="override stop step, keeping the fixed LR schedule")
    train.add_argument("--until-step", type=_positive, help="checkpoint and stop early for a staged run")
    train.add_argument("--output-dir", help="override run directory (relative to current working directory)")
    train.add_argument("--data", help="override prepared dataset location")
    train.add_argument("--seed", type=int)
    report = commands.add_parser("report", help="export comparison JSON/CSV/Markdown and scientific plots")
    report.add_argument("--runs", nargs="+", required=True)
    report.add_argument("--output", required=True)
    report.add_argument("--target-loss", type=float)
    budget = commands.add_parser("plan-budget", help="estimate time using user-provided credit tariff")
    budget.add_argument("--credits", type=float, default=200)
    budget.add_argument("--config", help="planning YAML with plan_budget keyword arguments")
    budget.add_argument("--rate", type=float, default=5.3)
    budget.add_argument("--reserve", type=float, default=0.2)
    budget.add_argument("--session-minutes", type=float, default=30)
    profile = commands.add_parser("profile", help="opt-in single-step profiler trace")
    profile.add_argument("--config", required=True)
    profile.add_argument("--output", required=True)
    profile.add_argument("--execute", action="store_true")
    profile.add_argument("--warmup-steps", type=int, default=2)
    profile.add_argument("--measure-steps", type=int, default=5)
    experiment = commands.add_parser("experiment", help="inspect one experiment; train only with --execute")
    experiment.add_argument("--config", required=True)
    experiment.add_argument("--execute", action="store_true")
    experiment.add_argument("--output-dir")
    experiment.add_argument("--data")
    experiment.add_argument("--seed", type=int)
    for name in ("evaluate", "generate", "reasoning", "benchmark", "diagnose"):
        cmd = commands.add_parser(name)
        if name == "diagnose":
            src_group = cmd.add_mutually_exclusive_group(required=True)
            src_group.add_argument("--checkpoint", help="saved checkpoint directory")
            src_group.add_argument("--config", help="experiment config YAML to diagnose at random initialization")
        else:
            cmd.add_argument("--checkpoint", required=True)
        cmd.add_argument("--data", help="relocated prepared data; manifest must match checkpoint")
        cmd.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        cmd.add_argument("--precision", choices=("fp32", "bf16"))
        if name == "evaluate":
            cmd.add_argument("--split", choices=("val", "test"), default="val")
            cmd.add_argument("--max-batches", type=_positive, help="default: all full validation windows")
            cmd.add_argument("--output", help="new identified evaluation JSON")
            cmd.add_argument("--windows-output", help="new JSONL of per-window language losses")
        elif name == "generate":
            cmd.add_argument("--prompt", default="")
            cmd.add_argument("--max-new-tokens", type=_positive, default=64)
            cmd.add_argument("--temperature", type=float, default=1.0, help="0 selects greedy decoding")
            cmd.add_argument("--top-k", type=_positive, default=40)
            cmd.add_argument("--seed", type=int, default=42)
        elif name == "reasoning":
            cmd.add_argument("--examples", type=_positive, default=2)
            cmd.add_argument("--difficulty", type=_positive, default=1)
            cmd.add_argument("--max-new-tokens", type=_positive, default=32)
            cmd.add_argument("--iterations", type=_positive)
            cmd.add_argument("--seen-examples")
            cmd.add_argument("--seed", type=int, default=42)
            cmd.add_argument("--output", required=True)
        elif name == "diagnose":
            cmd.add_argument("--samples", type=_positive, default=128)
            cmd.add_argument("--batch-size", type=_positive, default=1,
                             help="number of contiguous validation windows in the fixed diagnostic batch")
            cmd.add_argument("--output", required=True)
        else:
            cmd.add_argument("--prompt", default="The laboratory")
            cmd.add_argument("--max-new-tokens", type=_positive, default=16)
            cmd.add_argument("--warmup-tokens", type=int, default=0)
            cmd.add_argument("--output", required=True)
    return root


def _inference(args):
    # Metadata and safetensors only; does not unpickle the optimizer/RNG state.
    if getattr(args, "checkpoint", None) is None:
        cfg = load_config(args.config)
        cfg.train.device = args.device
        cfg.train.precision = args.precision or (cfg.train.precision if args.device == "cuda" else "fp32")
        cfg.train.__post_init__()
        device = setup_runtime(cfg.train)
        data = PreparedData(args.data or cfg.data.path)
        model = build_backbone(cfg).to(device)
        metadata = {"files": {"model.safetensors": None}, "step": 0, "initialization_only": True}
        return cfg, model, data, device, metadata
    metadata = read_checkpoint_metadata(args.checkpoint)
    cfg = ExperimentConfig.from_dict(metadata["config"])
    cfg.train.device = args.device
    cfg.train.precision = args.precision or (cfg.train.precision if args.device == "cuda" else "fp32")
    cfg.train.__post_init__()
    device = setup_runtime(cfg.train)
    data = PreparedData(args.data or cfg.data.path)
    if data.fingerprint != metadata["dataset_sha256"]:
        raise ValueError("dataset fingerprint differs from checkpoint")
    model = build_backbone(cfg).to(device)
    if metadata["model_class"] != class_name(model):
        raise ValueError("checkpoint architecture differs from its configuration")
    load_model(model, str(Path(args.checkpoint) / "model.safetensors"), device=str(device))
    return cfg, model, data, device, metadata


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
            if args.format == "documents-jsonl":
                from .document_data import prepare_documents
                result = prepare_documents(paths, args.output, kind=args.tokenizer, vocab_size=args.vocab_size)
            else:
                result = prepare_data(paths, args.output, kind=args.tokenizer, vocab_size=args.vocab_size,
                                      val_fraction=args.val_fraction, seed=args.seed)
        elif args.command == "inspect":
            cfg = load_config(args.config)
            with torch.device("meta") if args.meta else nullcontext():
                model = build_backbone(cfg)
                objective = build_objective(cfg, model)
            estimate = estimate_research_flops(model, objective, cfg.data.seq_len) if cfg.research else estimate_transformer_flops(cfg.model, cfg.data.seq_len)
            result = {"parameters_calculated": count_parameters(model),
                      "parameters_formula": transformer_parameter_formula(cfg.model) if type(model) is TransformerLM else None,
                      "total_parameters_calculated": sum(p.numel() for p in {id(p):p for root in (model,objective) for p in root.parameters()}.values()),
                      "trainable_parameters_calculated": sum(p.numel() for root in (model, objective) for p in root.parameters() if p.requires_grad),
                      "flops_estimate": asdict(estimate) if estimate else None}
        elif args.command in ("train", "experiment"):
            cfg = load_config(args.config)
            if getattr(args, "max_steps", None) is not None:
                cfg.train.max_steps = args.max_steps
            if args.output_dir:
                cfg.train.output_dir = str(Path(args.output_dir).resolve())
            if args.data:
                cfg.data.path = str(Path(args.data).resolve())
            if args.seed is not None:
                cfg.train.seed = args.seed
            cfg = ExperimentConfig.from_dict(cfg.to_dict())
            if args.command == "experiment" and not args.execute:
                result = {"execution": "not_started", "config": cfg.to_dict(),
                          "next_command": "Use experiment --execute or the explicit train command for this one config."}
                print(json.dumps(result, indent=2, ensure_ascii=False))
                return
            with Trainer(cfg, resume=getattr(args, "resume", None)) as trainer:
                metrics = trainer.fit(getattr(args, "until_step", None))
                result = {**metrics, "checkpoint": str(trainer.last_checkpoint)}
        elif args.command == "plan-budget":
            from .research.budget import plan_budget
            result = plan_budget(**yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))) if args.config else plan_budget(args.credits, args.rate, args.reserve, args.session_minutes)
        elif args.command == "report":
            from .research.reporting import generate_report
            result = generate_report(args.runs, args.output, target_loss=args.target_loss)
        elif args.command == "profile":
            if args.execute:
                from .research.profiling import profile_step
                result = profile_step(load_config(args.config), args.output,
                                      warmup_steps=args.warmup_steps, measure_steps=args.measure_steps)
            else:
                result = {"execution": "not_started", "config": args.config, "output": args.output,
                          "next_command": "Add --execute to perform one optimizer step and export its trace."}
        else:
            if args.command == "evaluate":
                destinations = [Path(p).resolve() for p in (args.output, args.windows_output) if p]
                if len(set(destinations)) != len(destinations):
                    raise ValueError("evaluation JSON and window JSONL must have distinct paths")
                if any(p.exists() for p in destinations):
                    raise FileExistsError("evaluation output already exists")
            cfg, model, data, device, metadata = _inference(args)
            if args.command == "evaluate":
                if args.split not in data.arrays:
                    raise ValueError(f"dataset has no reserved {args.split} split")
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                with ExitStack() as stack:
                    callback = None
                    if args.windows_output:
                        path = Path(args.windows_output)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        stream = stack.enter_context(path.open("x", encoding="utf-8"))
                        def callback(row):
                            row["target_start_offset"] = row["window_index"] * cfg.data.seq_len + 1
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                    result = evaluate(model, CausalLanguageModeling(),
                                      validation_batches(data.arrays[args.split], cfg.data.seq_len,
                                                         cfg.data.eval_batch_size, args.max_batches, device),
                                      device=device, precision=cfg.train.precision,
                                      deterministic=cfg.train.deterministic, window_callback=callback)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                result.update(evaluation_split=args.split, evaluation_wall_seconds=time.perf_counter()-started,
                              validation_scope="all_full_windows" if args.max_batches is None else "first_batches",
                              checkpoint=str(Path(args.checkpoint).resolve()),
                              checkpoint_model_sha256=metadata["files"]["model.safetensors"],
                              dataset_sha256=data.fingerprint, seq_len=cfg.data.seq_len,
                              precision=cfg.train.precision, device=str(device),
                              deterministic=cfg.train.deterministic,
                              cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                              windows_output=str(Path(args.windows_output).resolve()) if args.windows_output else None)
                if args.output:
                    path = Path(args.output)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("x", encoding="utf-8") as stream:
                        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            elif args.command == "generate":
                ids = data.tokenizer.encode(args.prompt) or [data.tokenizer.eos_id]
                output = generate(
                    model, torch.tensor([ids], device=device), max_seq_len=cfg.model.max_seq_len,
                    max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
                    eos_id=data.tokenizer.eos_id, vocab_size=data.tokenizer.vocab_size, seed=args.seed,
                    precision=cfg.train.precision, deterministic=cfg.train.deterministic,
                )
                print(data.tokenizer.decode(output[0].tolist()))
                return
            else:
                path = Path(args.output)
                if path.exists():
                    raise FileExistsError("inference evaluation output already exists")
                from .research.evaluators import evaluate_reasoning, benchmark_inference
                if args.command == "reasoning":
                    seen = json.loads(Path(args.seen_examples).read_text(encoding="utf-8")) if args.seen_examples else None
                    result = evaluate_reasoning(model, data.tokenizer, seed=args.seed, examples=args.examples,
                        difficulty=args.difficulty, max_new_tokens=args.max_new_tokens,
                        precision=cfg.train.precision, deterministic=cfg.train.deterministic,
                        seen_examples=seen, iterations=args.iterations)
                elif args.command == "diagnose":
                    from .research.diagnostics import diagnose_model
                    diag_bs = getattr(args, "batch_size", 1)
                    batch = next(validation_batches(data.arrays["val"], cfg.data.seq_len, diag_bs, 1, device))
                    actual_bs = int(batch["input_ids"].shape[0])
                    batch_source = (
                        "validation_first_window_b1"
                        if actual_bs == 1
                        else f"validation_first_{actual_bs}_windows_b{actual_bs}"
                    )
                    result = diagnose_model(
                        model,
                        batch["input_ids"],
                        targets=batch["targets"],
                        batch_source=batch_source,
                        max_samples=args.samples,
                        precision=cfg.train.precision,
                        deterministic=cfg.train.deterministic,
                    )
                else:
                    result = benchmark_inference(model, data.tokenizer, args.prompt,
                        max_new_tokens=args.max_new_tokens, warmup_tokens=args.warmup_tokens,
                        precision=cfg.train.precision, deterministic=cfg.train.deterministic)
                result["checkpoint"] = str(Path(args.checkpoint).resolve()) if getattr(args, "checkpoint", None) else None
                result["config_source"] = str(Path(args.config).resolve()) if getattr(args, "config", None) else None
                result["checkpoint_model_sha256"] = metadata["files"]["model.safetensors"]
                path.parent.mkdir(parents=True, exist_ok=True)
                write_json(path, result)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (ValueError, FileNotFoundError, FileExistsError, FloatingPointError, RuntimeError) as exc:
        arg_parser.exit(1, f"llm2: {exc}\n")
