"""Language-only milestones during concurrent training; separate final diagnostics."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch
from torch import nn
from ..cli import main as cli_main, _inference
from ..config import load_config
from ..checkpoint import read_checkpoint_metadata
from ..data import validation_batches, write_json
from ..evaluation import evaluate
from ..objectives import CausalLanguageModeling


def languages(config, *, device="cuda"):
    cfg = load_config(config); run = Path(cfg.train.output_dir)
    records = []
    for step in [cfg.train.max_steps*i//4 for i in range(1,5)]:
        checkpoint = run/"checkpoints"/f"step-{step:08d}"
        read_checkpoint_metadata(checkpoint)
        destination = run/"evaluations"/checkpoint.name
        destination.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        cli_main(["evaluate", "--checkpoint", str(checkpoint), "--device", device, "--precision", "fp32",
            "--split", "val", "--output", str(destination/"language.json"),
            "--windows-output", str(destination/"windows.jsonl")])
        value = json.loads((destination/"language.json").read_text())
        if value["evaluation_split"] != "val" or value["validation_scope"] != "all_full_windows":
            raise ValueError("full validation required")
        records.append({"step": step, "checkpoint": str(checkpoint),
            "checkpoint_model_sha256": value["checkpoint_model_sha256"],
            "evaluation": str(destination/"language.json"), "validation_loss": value["validation_loss"],
            "validation_tokens": value["validation_tokens"], "wall_seconds": time.monotonic()-started})
        gc.collect()
        if device == "cuda": torch.cuda.empty_cache()
    result = {"checkpoints": records, "split": "val", "test_evaluated": False}
    write_json(run/"evaluation_bundle.json", result)
    return result


class IterationView(nn.Module):
    def __init__(self, model, iterations):
        super().__init__(); self.backbone = model; self.iterations = iterations
    def forward(self, ids, **kwargs):
        return self.backbone(ids, iterations=self.iterations, **kwargs)


def depth_probe(checkpoint, output, *, device="cuda"):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    cfg, model, data, target, _ = _inference(argparse.Namespace(checkpoint=str(checkpoint),
        config=None, data=None, device=device, precision="fp32"))
    meta = read_checkpoint_metadata(checkpoint); rows = []
    for k in (1,2,4,8):
        with (output/f"k-{k}-windows.jsonl").open("x", encoding="utf-8") as stream:
            def window(row):
                row["target_start_offset"] = row["window_index"]*cfg.data.seq_len+1
                stream.write(json.dumps(row)+"\n")
            started = time.monotonic()
            result = evaluate(IterationView(model, k), CausalLanguageModeling(),
                validation_batches(data.arrays["val"], cfg.data.seq_len, cfg.data.eval_batch_size, None, target),
                device=target, precision="fp32", deterministic=cfg.train.deterministic, window_callback=window)
        rows.append({"iterations_override": k, "trained_iterations": cfg.research.iterations,
            "wall_seconds": time.monotonic()-started, **result})
    result = {"checkpoint_model_sha256": meta["files"]["model.safetensors"],
        "dataset_sha256": data.fingerprint, "split": "val", "test_evaluated": False,
        "rows": rows, "limitations": "K=2 training; sensitivity probe is not evidence of variable-depth training benefit"}
    write_json(output/"depth.json", result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config"); p.add_argument("--checkpoint"); p.add_argument("--output")
    p.add_argument("--device", default="cuda", choices=("cpu","cuda"))
    args = p.parse_args(argv)
    if args.checkpoint:
        if not args.output: p.error("depth probe needs --output")
        return depth_probe(args.checkpoint, args.output, device=args.device)
    if not args.config: p.error("language bundle needs --config")
    return languages(args.config, device=args.device)

if __name__ == "__main__": main()
