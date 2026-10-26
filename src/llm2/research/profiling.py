"""One optimizer-step trace; FLOPs provided by profiler cover only supported operators."""

from dataclasses import replace
from pathlib import Path

import torch

from ..data import write_json
from ..trainer import Trainer


def profile_step(config, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("profiler output already exists")
    cfg = type(config).from_dict(config.to_dict())
    cfg.train.output_dir = str(output / "run")
    activities = [torch.profiler.ProfilerActivity.CPU]
    if cfg.train.device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with Trainer(cfg) as trainer:
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True,
                                    with_flops=True) as profiler:
            metrics = trainer.train_step()
        profiler.export_chrome_trace(str(output / "trace.json"))
        supported_flops = sum(event.flops or 0 for event in profiler.key_averages())
        result = {"step_metrics": metrics, "profiler_supported_ops_flops": supported_flops,
                  "analytical_step_flops_estimate": metrics["step_flops_estimate"],
                  "limitations": "Profiler FLOPs omit unsupported/fused operators; this trace is a calibration aid, not a complete measured FLOP total."}
        write_json(output / "profile.json", result)
    return result
