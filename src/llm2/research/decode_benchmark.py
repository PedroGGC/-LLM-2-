"""Frozen-weight generation benchmark launched from the operator checkout."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from tempfile import TemporaryDirectory
import time
import traceback
from zipfile import ZipFile

import llm2
import torch
from safetensors.torch import load_model

from llm2.config import ExperimentConfig
from llm2.data import write_json
from llm2.research.checkpoint_review import verified_source
from llm2.research.cycle4 import archive_files
from llm2.research.factory import build_backbone
from llm2.research.resource_telemetry import EnergyMeter, energy_result, host_memory
from llm2.research.session import SessionBudget, SessionBudgetExceeded, SessionCampaign
from llm2.research.stories_data import StoriesData, prepare
from llm2.research.stories_worker import score
from llm2.runtime import attention_context, environment, source_provenance

if __package__:
    from .decode_cache import RecurrentDecode, TransformerDecode, generate, verify_logits
else:
    from decode_cache import RecurrentDecode, TransformerDecode, generate, verify_logits

LENGTHS = (256, 1024, 7936)
NEW_BYTES = 256
ROOT = Path(llm2.__file__).resolve().parents[2]
VARIANTS = ("context_untied2", "transformer_medium")
SEEDS = (341, 342)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def locate_weights(explicit, work, inputs):
    if explicit:
        source = Path(explicit)
        if not source.exists():
            raise FileNotFoundError(f"Pesos da 0033 não encontrados em {source}")
        return source
    original = work / "llm2-campaign-0033"
    folders = [original / "runs" / f"{variant}-seed-{seed}" / "checkpoints/step-00006000"
               for variant in VARIANTS for seed in SEEDS]
    if all((folder / name).is_file() for folder in folders for name in ("metadata.json", "model.safetensors")):
        return original
    filename = "llm2-campaign-0033-checkpoints.zip"
    if (work / filename).is_file():
        return work / filename
    candidates = sorted(inputs.rglob(filename))
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise ValueError("Mais de um ZIP de checkpoints da 0033. Informe checkpoints na célula.")
    raise FileNotFoundError(
        "Pesos da 0033 ausentes. Preserve a pasta da 0033 ou adicione "
        "llm2-campaign-0033-checkpoints.zip como input privado. O ZIP de resultados contém as métricas."
    )


def read_weight(source, job):
    folder = job["folder"]
    if source.is_dir():
        def payload(name, limit):
            path = source / folder / name
            if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
                raise ValueError("checkpoint path escapes its directory")
            with path.open("rb") as stream:
                return stream.read(limit + 1)
        metadata = payload("metadata.json", 1024**2)
        weight = payload("model.safetensors", 8 * 1024**2)
    else:
        with ZipFile(source) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("duplicate ZIP members")
            def payload(name, limit):
                with archive.open(f"{folder}/{name}") as stream:
                    return stream.read(limit + 1)
            metadata = payload("metadata.json", 1024**2)
            weight = payload("model.safetensors", 8 * 1024**2)
    if len(metadata) > 1024**2 or hashlib.sha256(metadata).hexdigest() != job["metadata_sha256"]:
        raise ValueError(f"Metadados do checkpoint divergentes: {folder}")
    if len(weight) > 8 * 1024**2 or hashlib.sha256(weight).hexdigest() != job["weight_sha256"]:
        raise ValueError(f"SHA-256 dos pesos divergente: {folder}")
    return weight


def load_frozen_model(source, job, device):
    raw = read_weight(source, job)
    cfg = ExperimentConfig.from_dict(job["config"])
    model = build_backbone(cfg)
    if getattr(model, "architecture_version", 1) != job["architecture_version"]:
        raise ValueError("architecture version differs from frozen checkpoint")
    with TemporaryDirectory(prefix="llm2-decode-weight-") as folder:
        path = Path(folder) / "model.safetensors"
        path.write_bytes(raw)
        load_model(model, path, strict=True)
    if model.embedding.weight is not model.lm_head.weight or any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("unexpected embedding sharing or precision")
    return model.to(device).eval()


def write_result(path, value):
    temporary = path.with_suffix(".tmp")
    write_json(temporary, value)
    os.replace(temporary, path)


def benchmark(decoder, prefix, new_bytes):
    device = prefix.device
    warm, expected = generate(decoder, prefix, new_bytes)
    sample = bytes(expected.flatten().cpu().tolist())
    del expected
    baseline = None
    if device.type == "cuda":
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    meter = EnergyMeter(device)
    timings = []
    try:
        initial = meter.energy_mj()
        began = time.perf_counter()
        count = 0
        while count < 7 or (initial is not None and time.perf_counter() - began < 3 and count < 256):
            timing, tokens = generate(decoder, prefix, new_bytes)
            del tokens
            if count < 7:
                timings.append(timing)
            count += 1
        energy = energy_result(initial, meter.energy_mj(), [], count * new_bytes, time.perf_counter() - began)
    finally:
        meter.close()
    peak = torch.cuda.max_memory_allocated(device) if baseline is not None else None
    median = {key: statistics.median(row[key] for row in timings)
              for key in ("first_byte_seconds", "decode_seconds", "total_seconds")}
    return {
        **median, "prefix_bytes": prefix.shape[1], "new_bytes": new_bytes, "samples": timings,
        "decode_bytes_per_second": (new_bytes - 1) / median["decode_seconds"],
        "peak_vram_bytes": peak, "baseline_vram_bytes": baseline,
        "incremental_peak_bytes": peak - baseline if peak is not None else None,
        "state_bytes": warm["state_bytes"], "generated_hex": sample.hex(),
        "energy": {"method": energy["method"], "board_joules": energy["gpu_joules"],
                   "joules_per_generated_byte": energy["gpu_joules_per_training_target"],
                   "generated_bytes": energy["measured_training_targets"],
                   "window_seconds": energy["window_seconds"]},
    }


def evaluate(model, data, device, result_path, result, *, lengths=LENGTHS, new_bytes=NEW_BYTES):
    decoder = TransformerDecode(model) if result["variant"] == "transformer_medium" else RecurrentDecode(model)
    full_length = max(row["bytes"] for row in data.docs["val"])
    result["validation"] = [
        {"context": length, **score(model, data, "val", length, batch, device)}
        for length, batch in ((256, 32), (full_length, 4))
    ]
    if not all(math.isfinite(item["nll"]) for item in result["validation"]):
        raise ValueError("nonfinite validation NLL")
    result["nll_difference_from_0033"] = result["validation"][0]["nll"] - result["original_nll_t256"]
    if abs(result["nll_difference_from_0033"]) > .002:
        raise ValueError("T256 NLL did not reproduce the frozen 0033 checkpoint")
    write_result(result_path, result)
    for length in lengths:
        rng = torch.Generator().manual_seed(17634 + length)
        tokens = torch.randint(0, 256, (1, length + new_bytes), generator=rng).to(device)
        prefix, continuation = tokens[:, :length], tokens[:, length:]
        with attention_context(device.type == "cpu"):
            parity = verify_logits(decoder, prefix, continuation)
            profile = benchmark(decoder, prefix, new_bytes)
        result["profiles"].append({"parity": parity, **profile})
        write_result(result_path, result)
        print(f"{result['variant']} seed {result['seed']} prefixo {length}. "
              f"Primeiro byte {profile['first_byte_seconds'] * 1000:.2f} ms. "
              f"Decodificação {profile['decode_bytes_per_second']:.1f} bytes/s.", flush=True)
    result.update(status="complete", host_memory=host_memory())
    write_result(result_path, result)


def run_worker(args, ref):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(device):
        raise RuntimeError("benchmark worker requires one isolated T4")
    job = ref["jobs"][args.job]
    model = load_frozen_model(Path(args.weights), job, device)
    data = StoriesData(args.data)
    if data.fingerprint != ref["dataset_sha256"]:
        raise ValueError("validation dataset differs from the frozen checkpoint")
    result = {
        "variant": job["variant"], "seed": job["seed"], "status": "running", "profiles": [],
        "original_nll_t256": job["nll_t256"], "weight_sha256": job["weight_sha256"],
        "dataset_sha256": data.fingerprint, "environment": environment(),
        "attention": {"mode": "PyTorch eager SDPA auto", "flash": torch.backends.cuda.flash_sdp_enabled(),
                      "efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                      "math": torch.backends.cuda.math_sdp_enabled(),
                      "cudnn": torch.backends.cuda.cudnn_sdp_enabled()},
    }
    evaluate(model, data, device, Path(args.output) / (args.job + ".json"), result)


def comparisons(rows):
    complete = {(row["variant"], row["seed"]): row for row in rows if row["status"] == "complete"}
    pairs = []
    for seed in SEEDS:
        if any((variant, seed) not in complete for variant in VARIANTS):
            continue
        model, reference = (complete[variant, seed] for variant in VARIANTS)
        for a, b in zip(model["profiles"], reference["profiles"], strict=True):
            if a["prefix_bytes"] != b["prefix_bytes"] or a["new_bytes"] != b["new_bytes"]:
                raise ValueError("incomparable workloads")
            pairs.append({
                "seed": seed, "prefix_bytes": a["prefix_bytes"],
                "first_byte_speedup": b["first_byte_seconds"] / a["first_byte_seconds"],
                "decode_speedup": b["decode_seconds"] / a["decode_seconds"],
                "total_speedup": b["total_seconds"] / a["total_seconds"],
            })
    longest = [p for p in pairs if p["prefix_bytes"] == LENGTHS[-1]]
    return {
        "pairs": pairs,
        "twofold_long_generation": len(longest) == len(SEEDS) and all(p["total_speedup"] >= 2 for p in longest),
        "scope": "FP32 eager greedy on synthetic prefixes. Long-context language quality is separate.",
    }


def run_campaign(args, ref):
    budget = SessionBudget(args.seconds, 300, elapsed=time.monotonic() - args.started_at)
    campaign = SessionCampaign(ROOT, args.output, device="cuda", gpus=args.gpus, budget=budget)
    campaign.manifest.update(
        cycle="cached_decode_0033_v1", optimizer_steps=0, source_campaign=ref["campaign"],
        dataset_sha256=ref["dataset_sha256"], reference=ref, operator=read_json(args.operator_metadata),
        limitations="Synthetic generation timing, TinyStories development quality. Two paired seeds. Shared host.",
    )
    error = None
    try:
        weights = locate_weights(args.weights, campaign.output.parent, Path("/kaggle/input"))
        for job in ref["jobs"].values():
            read_weight(weights, job)
        print(f"Quatro checkpoints verificados em {weights}.", flush=True)
        data = prepare(args.cache, budget=budget)
        if data.fingerprint != ref["dataset_sha256"]:
            raise ValueError("prepared dataset differs from 0033")
        campaign.manifest["weight_source"] = str(weights)
        campaign.manifest["status"] = "running"
        campaign.save()
        for index, seed in enumerate(SEEDS):
            with ThreadPoolExecutor(max_workers=2) as workers:
                tasks = []
                for slot, variant in enumerate(VARIANTS):
                    job = f"{variant}-seed-{seed}"
                    command = [__file__, "--reference", args.reference, "--output", args.output,
                               "--weights", weights, "--data", data.root, "--job", job]
                    tasks.append(workers.submit(campaign.command, command, job + ".log",
                                                gpu=args.gpus[(slot + index) % 2]))
                for task in tasks:
                    task.result()
        campaign.manifest["status"] = "complete"
    except SessionBudgetExceeded as exc:
        campaign.manifest.update(status="partial_budget", error=str(exc))
    except Exception as exc:
        error = exc
        campaign.manifest.update(status="failed", error=str(exc), traceback=traceback.format_exc())
    finally:
        campaign.abort()
        rows = [read_json(path) for job in ref["jobs"]
                if (path := campaign.output / (job + ".json")).is_file()]
        campaign.manifest.update(runs=rows, session_budget=budget.snapshot())
        campaign.save()
        summary = comparisons(rows)
        write_json(campaign.output / "comparison.json", summary)
        result = campaign.output.parent / (campaign.output.name + "-results.zip")
        files = list(campaign.output.glob("*.json")) + list((campaign.output / "logs").glob("*.log"))
        archive_files(result, files, base=campaign.output)
        print(f"Status {campaign.manifest['status']}. Resultado {result.name}.", flush=True)
    if error is not None:
        raise RuntimeError("Diagnóstico interrompido. Consulte o ZIP de resultados.") from error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--data")
    parser.add_argument("--job")
    parser.add_argument("--cache")
    parser.add_argument("--operator-metadata")
    parser.add_argument("--gpus", nargs=2)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--started-at", type=float)
    args = parser.parse_args()
    ref = read_json(args.reference)
    if ref["revision"] != source_provenance()["git"]["commit"]:
        raise ValueError("frozen worker revision differs")
    verified_source(ROOT, ref["source_sha256"])
    if args.job:
        run_worker(args, ref)
    else:
        run_campaign(args, ref)


if __name__ == "__main__":
    main()
