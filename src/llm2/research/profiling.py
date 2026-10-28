"""Warm timing samples plus a separate instrumented step, in a disposable run."""

from pathlib import Path
import statistics

import torch

from ..data import write_json
from ..trainer import Trainer


def profile_step(config, output, *, warmup_steps=0, measure_steps=0):
    output = Path(output)
    if output.exists():
        raise FileExistsError("profiler output already exists")
    if type(warmup_steps) is not int or type(measure_steps) is not int or min(warmup_steps, measure_steps) < 0:
        raise ValueError("profile step counts must be nonnegative integers")
    if warmup_steps + measure_steps + 1 > config.train.max_steps:
        raise ValueError("profiling needs warmup + measured + one trace step within max_steps")
    cfg = type(config).from_dict(config.to_dict())
    cfg.train.output_dir = str(output / "run")
    activities = [torch.profiler.ProfilerActivity.CPU]
    if cfg.train.device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with Trainer(cfg) as trainer:
        warmup_samples = [trainer.train_step() for _ in range(warmup_steps)]
        samples = [trainer.train_step() for _ in range(measure_steps)]
        with torch.profiler.profile(
            activities=activities, record_shapes=True, profile_memory=True, with_flops=True
        ) as profiler:
            metrics = trainer.train_step()
        profiler.export_chrome_trace(str(output / "trace.json"))
        events = profiler.key_averages()
        warmup_durations = [sample["step_train_wall_seconds"] for sample in warmup_samples]
        durations = [sample["step_train_wall_seconds"] for sample in samples]
        sample_tok_s = [sample["step_tokens_per_second"] for sample in samples]
        measured_tokens = sum(sample["step_tokens"] for sample in samples)
        measured_seconds = sum(durations)
        cuda_launch_calls = sum(
            event.count for event in events if event.key == "cudaLaunchKernel"
        )
        result = {
            "step_metrics": metrics,
            "warmup_steps": warmup_steps,
            "warmup_timing": {
                "warmup_steps": warmup_steps,
                "first_step_seconds": warmup_durations[0] if warmup_durations else None,
                "last_warmup_step_seconds": warmup_durations[-1] if warmup_durations else None,
                "total_warmup_seconds": sum(warmup_durations) if warmup_durations else 0.0,
                "compile_enabled": bool(cfg.train.compile),
            },
            "unprofiled_samples": samples,
            "timing": {
                "measured_steps": measure_steps,
                "mean_step_seconds": statistics.mean(durations) if durations else None,
                "median_step_seconds": statistics.median(durations) if durations else None,
                "stdev_step_seconds": statistics.stdev(durations) if len(durations) >= 2 else (0.0 if durations else None),
                "min_step_seconds": min(durations) if durations else None,
                "max_step_seconds": max(durations) if durations else None,
                "mean_tokens_per_second": measured_tokens / measured_seconds if durations else None,
                "median_tokens_per_second": statistics.median(sample_tok_s) if sample_tok_s else None,
                "scope": "train_step; warmup and instrumented trace excluded",
            },
            "cuda_launch_kernel_calls": cuda_launch_calls,
            "observed_attention_operators": sorted(
                {event.key for event in events if "scaled_dot_product" in event.key}
            ),
            "operators": [
                {
                    "name": e.key,
                    "calls": e.count,
                    "self_cpu_time_us": e.self_cpu_time_total,
                    "self_device_time_us": getattr(e, "self_device_time_total", 0),
                    "supported_flops": e.flops or 0,
                }
                for e in sorted(events, key=lambda e: e.self_cpu_time_total, reverse=True)
            ],
            "profiler_supported_ops_flops": sum(event.flops or 0 for event in events),
            "analytical_step_flops_estimate": metrics["step_flops_estimate"],
            "limitations": (
                "Profiler FLOPs omit unsupported/fused operators. Memory statistics may omit pre-trace allocations. "
                "GPU event intervals are not summed kernel time. Warm timings are short observations, not a scientific campaign."
            ),
        }
        write_json(output / "profile.json", result)
    return result
