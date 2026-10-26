"""Single-device trainer; backbone, objective, optimizer and evaluation stay separate."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import importlib.metadata
from pathlib import Path
import time
import uuid

import torch
import yaml

from .checkpoint import load_checkpoint, save_checkpoint
from .config import ExperimentConfig
from .data import PreparedData, RandomBatchStream, validation_batches, write_json
from .evaluation import evaluate
from .metrics import MetricLogger, estimate_transformer_flops
from .model import TransformerLM, count_parameters
from .objectives import CausalLanguageModeling
from .optim import build_optimizer, build_scheduler
from .runtime import (attention_context, autocast_context, capture_rng, environment,
                      restore_rng, save_source_snapshot, setup_runtime, source_provenance)


class Trainer:
    def __init__(self, config, *, model=None, objective=None, data=None, resume=None,
                 flops_estimator=None, model_factory=None, objective_factory=None):
        # Revalidate even when a caller has mutated a dataclass programmatically.
        self.config = ExperimentConfig.from_dict(config.to_dict())
        cfg = self.config
        self.device = setup_runtime(cfg.train)
        if (model is not None and model_factory is not None) or (objective is not None and objective_factory is not None):
            raise ValueError("provide an instance or a factory, not both")
        self.data = data if data is not None else PreparedData(cfg.data.path)
        if cfg.model.vocab_size < self.data.tokenizer.vocab_size:
            raise ValueError("model vocabulary is smaller than the prepared tokenizer")
        if min(len(x) for x in self.data.arrays.values()) <= cfg.data.seq_len:
            raise ValueError("both splits need at least seq_len + 1 tokens")
        if model is None:
            model = model_factory() if model_factory is not None else TransformerLM(cfg.model)
        if objective is None:
            objective = objective_factory() if objective_factory is not None else CausalLanguageModeling()
        self.model, self.objective = model.to(self.device), objective.to(self.device)
        if isinstance(self.model, TransformerLM) and self.model.config != cfg.model:
            raise ValueError("Transformer instance and experiment model configuration disagree")
        self.optimizer = build_optimizer(self.model, self.objective, cfg.optimizer)
        self.scheduler = build_scheduler(self.optimizer, cfg.optimizer)
        self.sampler = RandomBatchStream(self.data.arrays["train"], cfg.data.seq_len,
                                        cfg.data.batch_size, cfg.train.seed + 1)
        self.parameters = [p for group in self.optimizer.param_groups for p in group["params"]]
        self.model_parameters = count_parameters(self.model)
        self.total_parameters = sum(p.numel() for p in {id(p): p for module in (self.model, self.objective)
                                                       for p in module.parameters()}.values())
        if flops_estimator is None and type(self.model) is TransformerLM and type(self.objective) is CausalLanguageModeling:
            flops_estimator = lambda: estimate_transformer_flops(cfg.model, cfg.data.seq_len)
        estimate = flops_estimator() if flops_estimator is not None else None
        self.flops = asdict(estimate) if estimate is not None else None
        self.environment_info = environment()
        self.provenance = source_provenance()
        self.counters = {"step": 0, "tokens_processed": 0, "sequences_processed": 0,
                         "training_flops_estimate": 0 if self.flops is not None else None,
                         "train_wall_seconds": 0.0,
                         "gpu_time_seconds": 0.0 if self.device.type == "cuda" else None,
                         "peak_vram_allocated_bytes": 0 if self.device.type == "cuda" else None,
                         "peak_vram_reserved_bytes": 0 if self.device.type == "cuda" else None}
        self.output_dir = Path(cfg.train.output_dir)
        self.last_checkpoint = None
        if resume is not None:
            self.counters, _ = load_checkpoint(
                resume, model=self.model, objective=self.objective, optimizer=self.optimizer,
                scheduler=self.scheduler, sampler=self.sampler, config=cfg,
                data_fingerprint=self.data.fingerprint, environment_info=self.environment_info,
                provenance=self.provenance, flops=self.flops,
            )
            self.last_checkpoint = Path(resume).resolve()
            for existing in (self.output_dir / "checkpoints").glob("step-*"):
                if int(existing.name.split("-")[-1]) > self.counters["step"]:
                    raise ValueError("a later checkpoint exists; use a new output_dir to branch")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            if self.output_dir.exists() and any(self.output_dir.iterdir()):
                raise FileExistsError("output_dir is not empty; resume or choose a new directory")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        saved_rng = capture_rng()
        run_metadata = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "config": cfg.to_dict(), "dataset_sha256": self.data.fingerprint,
                        "environment": self.environment_info, "provenance": self.provenance,
                        "model_parameters_calculated": self.model_parameters,
                        "total_parameters_calculated": self.total_parameters,
                        "trainable_parameters_calculated": sum(p.numel() for p in self.parameters),
                        "flops_estimate": self.flops,
                        "resume_from": str(resume) if resume else None,
                        "start_step": self.counters["step"]}
        name = "run.json" if resume is None else f"resume-{uuid.uuid4().hex}.json"
        write_json(self.output_dir / name, run_metadata)
        (self.output_dir / name.replace(".json", ".yaml")).write_text(
            yaml.safe_dump(cfg.to_dict(), sort_keys=True), encoding="utf-8")
        if not (self.output_dir / "source.zip").exists():
            save_source_snapshot(self.output_dir / "source.zip", self.provenance)
            freeze = sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
                             if d.metadata["Name"]}, key=str.casefold)
            (self.output_dir / "environment.freeze.txt").write_text("\n".join(freeze) + "\n", encoding="utf-8")
        # Keep raw logs from failed attempts but remove superseded metrics from the active stream.
        if resume is not None:
            self._rewind_metrics()
        self.logger = MetricLogger(self.output_dir,
                                   purge_step=self.counters["step"] + 1 if resume else None)
        self.train_model = torch.compile(self.model, backend=cfg.train.compile_backend) if cfg.train.compile else self.model
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        restore_rng(saved_rng)
        self.closed = False
        self._at_step_boundary = True

    def _rewind_metrics(self):
        path = self.output_dir / "metrics.jsonl"
        if not path.exists():
            return
        lines = path.read_text(encoding="utf-8").splitlines()
        kept = []
        for line in lines:
            try:
                if json.loads(line)["step"] <= self.counters["step"]:
                    kept.append(line)
            except (ValueError, KeyError):
                pass  # A crash may leave a partially written last line.
        if len(kept) != len(lines):
            path.rename(self.output_dir / f"metrics-before-resume-{uuid.uuid4().hex}.jsonl")
            path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    def _update_peak_memory(self):
        if self.device.type == "cuda":
            for name, getter in (("allocated", torch.cuda.max_memory_allocated),
                                 ("reserved", torch.cuda.max_memory_reserved)):
                key = f"peak_vram_{name}_bytes"
                self.counters[key] = max(self.counters[key], getter(self.device))

    def train_step(self):
        cfg = self.config
        if not self._at_step_boundary:
            raise RuntimeError("an incomplete step requires recovery from a checkpoint")
        if self.counters["step"] >= cfg.train.max_steps:
            raise ValueError("max_steps already reached")
        self._at_step_boundary = False
        self.model.train()
        self.objective.train()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event.record()
        started = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum, tokens, sequences = 0.0, 0, 0
        with attention_context(cfg.train.deterministic):
            for _ in range(cfg.train.grad_accum_steps):
                batch = self.sampler.next(self.device)
                with autocast_context(self.device, cfg.train.precision):
                    result = self.objective(self.train_model, batch)
                if result.loss.ndim != 0 or not torch.isfinite(result.loss) or result.token_count <= 0:
                    raise FloatingPointError("non-finite loss, non-scalar loss or invalid token count")
                (result.loss / cfg.train.grad_accum_steps).backward()
                loss_sum += float(result.loss.detach()) * result.token_count
                tokens += result.token_count
                sequences += batch["input_ids"].shape[0]
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters, cfg.train.grad_clip,
                                                      error_if_nonfinite=True)
            lr_used = self.optimizer.param_groups[0]["lr"]
            self.optimizer.step()
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            end_event.record()
            torch.cuda.synchronize(self.device)
            self.counters["gpu_time_seconds"] += start_event.elapsed_time(end_event) / 1000
        elapsed = time.perf_counter() - started
        self.counters["step"] += 1
        self.counters["tokens_processed"] += tokens
        self.counters["sequences_processed"] += sequences
        self.counters["train_wall_seconds"] += elapsed
        if self.flops is not None:
            self.counters["training_flops_estimate"] += self.flops["training_per_sequence"] * sequences
        self._update_peak_memory()
        self._at_step_boundary = True
        return {"training_loss": loss_sum / tokens, "lr": lr_used, "grad_norm": float(grad_norm),
                "step_tokens": tokens, "step_train_wall_seconds": elapsed,
                "step_tokens_per_second": tokens / elapsed,
                "tokens_per_second": self.counters["tokens_processed"] / self.counters["train_wall_seconds"],
                "model_parameters_calculated": self.model_parameters,
                "total_parameters_calculated": self.total_parameters,
                **{k: v for k, v in self.counters.items() if k != "step"}}

    def validate(self, max_batches=None):
        cfg = self.config
        result = evaluate(self.model, self.objective,
                          validation_batches(self.data.arrays["val"], cfg.data.seq_len,
                                             cfg.data.eval_batch_size, max_batches, self.device),
                          device=self.device, precision=cfg.train.precision,
                          deterministic=cfg.train.deterministic)
        self._update_peak_memory()
        return result

    def save(self):
        if not self._at_step_boundary:
            raise RuntimeError("checkpoints are allowed only at complete optimizer-step boundaries")
        step = self.counters["step"]
        path = (self.output_dir / "checkpoints" / f"step-{step:08d}").resolve()
        if self.last_checkpoint == path and path.exists():
            return path
        self._update_peak_memory()
        self.last_checkpoint = save_checkpoint(
            path, model=self.model, objective=self.objective, optimizer=self.optimizer,
            scheduler=self.scheduler, sampler=self.sampler, counters=self.counters,
            config=self.config, data_fingerprint=self.data.fingerprint,
            environment_info=self.environment_info, provenance=self.provenance, flops=self.flops,
        )
        return self.last_checkpoint

    def fit(self, until_step=None):
        target = self.config.train.max_steps if until_step is None else until_step
        if not self.counters["step"] <= target <= self.config.train.max_steps:
            raise ValueError("invalid stopping step")
        latest = None
        while self.counters["step"] < target:
            metrics = self.train_step()
            step = self.counters["step"]
            should_eval = step % self.config.train.eval_interval == 0 or step == target
            if should_eval:
                metrics.update(self.validate(self.config.train.eval_batches))
                metrics.update({k: v for k, v in self.counters.items() if k.startswith("peak_vram")})
                latest = metrics
            if step % self.config.train.log_interval == 0 or should_eval:
                self.logger.log(step, **metrics)
            if step % self.config.train.checkpoint_interval == 0 or step == target:
                self.save()
        if latest is None:
            latest = self.validate(self.config.train.eval_batches)
            self.logger.log(self.counters["step"], **latest)
        write_json(self.output_dir / "summary.json", {**latest, **self.counters,
                                                     "checkpoint": str(self.last_checkpoint)})
        return latest

    def close(self):
        if not self.closed:
            self.logger.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

