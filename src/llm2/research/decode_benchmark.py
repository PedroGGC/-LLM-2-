"""Frozen-weight generation benchmark launched from the operator checkout."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
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

from llm2.checkpoint import read_checkpoint_metadata
from llm2.config import ExperimentConfig
from llm2.data import sha256_file, write_json
from llm2.research.checkpoint_review import verified_source
from llm2.research.cycle4 import archive_files, checkpoint_files
from llm2.research.factory import build_backbone
from llm2.research.resource_telemetry import EnergyMeter, energy_result, host_memory
from llm2.research.session import SessionBudget, SessionBudgetExceeded, SessionCampaign
from llm2.research.state_readout_worker import retain_final
from llm2.research.state_study import discard_compiler_cache, storage_budget
from llm2.research.stories_data import StoriesData, prepare
from llm2.research.stories_worker import StoryTrainer, score
from llm2.runtime import attention_context, environment, source_provenance

if __package__:
    from .decode_cache import RecurrentDecode, TransformerDecode, generate, verify_logits
    from .decode_optimization import CompiledDecode, LocalTransformerDecode, cold_step_seconds, profile_steps
else:
    from decode_cache import RecurrentDecode, TransformerDecode, generate, verify_logits
    from decode_optimization import CompiledDecode, LocalTransformerDecode, cold_step_seconds, profile_steps

LENGTHS = (256, 1024, 7936)
NEW_BYTES = 256
ROOT = Path(llm2.__file__).resolve().parents[2]
VARIANTS = ("context_untied2", "transformer_medium")
SEEDS = (341, 342)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def locate_weights(explicit, work, inputs, *, campaign="llm2-campaign-0033"):
    if explicit:
        source = Path(explicit)
        if not source.exists():
            raise FileNotFoundError(f"Pesos de {campaign} não encontrados em {source}")
        return source
    original = work / campaign
    folders = [original / "runs" / f"{variant}-seed-{seed}" / "checkpoints/step-00006000"
               for variant in VARIANTS for seed in SEEDS]
    if all((folder / name).is_file() for folder in folders for name in ("metadata.json", "model.safetensors")):
        return original
    filename = f"{campaign}-checkpoints.zip"
    if (work / filename).is_file():
        return work / filename
    candidates = sorted(inputs.rglob(filename))
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise ValueError(f"Mais de um ZIP de {campaign}. Informe checkpoints na célula.")
    raise FileNotFoundError(
        f"Pesos de {campaign} ausentes. Preserve a pasta da campanha ou adicione "
        f"{filename} como input privado. O ZIP de resultados contém as métricas."
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


def rebuild_config(original, directory, data):
    cfg = ExperimentConfig.from_dict(original["config"])
    cfg.train.output_dir = str(directory)
    cfg.data.path = str(data)
    return cfg


def rebuild_job(output, job_id, ref, data, *, smoke=False):
    original = ref["jobs"][job_id]
    directory = output / "runs" / job_id
    cfg = rebuild_config(original, directory, data)
    with StoryTrainer(cfg, smoke=smoke) as trainer:
        if trainer.data.fingerprint != ref["dataset_sha256"]:
            raise ValueError("rebuild dataset differs from the frozen protocol")
        trainer.fit()
        if trainer.counters["step"] != cfg.train.max_steps:
            raise RuntimeError("rebuild missed the fixed training horizon")
        final = trainer.last_checkpoint
        meta = read_checkpoint_metadata(final)
        nll = trainer.evaluations[-1]["nll"]
        if not math.isfinite(nll):
            raise ValueError("nonfinite rebuilt checkpoint NLL")
        retain_final(directory, final)
        result = {
            **original, "config": cfg.to_dict(), "folder": final.relative_to(output).as_posix(),
            "architecture_version": meta["architecture_version"], "nll_t256": nll,
            "original_nll_t256": original["nll_t256"], "original_weight_sha256": original["weight_sha256"],
            "weight_sha256": meta["files"]["model.safetensors"],
            "metadata_sha256": sha256_file(final / "metadata.json"), "evaluations": trainer.evaluations,
        }
        write_result(directory / "rebuilt.json", result)


def verified_rebuild_record(output, job_id, ref, data):
    original = ref["jobs"][job_id]
    directory = output / "runs" / job_id
    record = read_json(directory / "rebuilt.json")
    cfg = rebuild_config(original, directory, data)
    final = directory / "checkpoints" / f"step-{cfg.train.max_steps:08d}"
    meta = read_checkpoint_metadata(final)
    if (meta["config"] != cfg.to_dict() or record["config"] != meta["config"]
            or record["folder"] != final.relative_to(output).as_posix()
            or record["variant"] != original["variant"] or record["seed"] != original["seed"]
            or meta["dataset_sha256"] != ref["dataset_sha256"]
            or meta["provenance"]["source_sha256"] != ref["source_sha256"]
            or meta["provenance"]["git"]["commit"] != ref["revision"]
            or record["architecture_version"] != original["architecture_version"]
            or meta["architecture_version"] != record["architecture_version"]
            or record["weight_sha256"] != meta["files"]["model.safetensors"]):
        raise ValueError("rebuilt checkpoint provenance mismatch")
    summary = read_json(directory / "summary.json")
    if summary["step"] != cfg.train.max_steps or summary["validation_loss"] != record["nll_t256"]:
        raise ValueError("rebuilt checkpoint validation record mismatch")
    read_weight(output, record)
    return record


def paired_jobs(campaign, gpus, commands, *, training=None):
    for index, seed in enumerate(SEEDS):
        with ThreadPoolExecutor(max_workers=2) as workers:
            tasks = []
            for slot, variant in enumerate(VARIANTS):
                job = f"{variant}-seed-{seed}"
                gpu = gpus[(slot + index) % 2]
                progress = None if training is None else {
                    "max_steps": training[job]["config"]["train"]["max_steps"],
                    "model": variant, "seed": seed, "physical_gpu": gpu,
                    "checkpoint_interval": training[job]["config"]["train"]["checkpoint_interval"],
                }
                stage = "rebuild" if training is not None else "decode"
                tasks.append(workers.submit(campaign.command, commands(job), f"{stage}-{job}.log",
                                            gpu=gpu, job=progress))
            for task in tasks:
                task.result()


@contextmanager
def private_compiler_cache(campaign):
    previous = os.environ.get("LLM2_COMPILER_CACHE_ROOT")
    os.environ["LLM2_COMPILER_CACHE_ROOT"] = str(campaign.output / "compiler-cache")
    try:
        with storage_budget(campaign):
            yield
    finally:
        if previous is None:
            os.environ.pop("LLM2_COMPILER_CACHE_ROOT", None)
        else:
            os.environ["LLM2_COMPILER_CACHE_ROOT"] = previous


def rebuild_reference(campaign, args, original, data):
    output = campaign.output
    data_path = data.root

    def command(job):
        return [__file__, "--reference", args.reference, "--output", output, "--data", data_path,
                "--train-job", job]

    with private_compiler_cache(campaign):
        paired_jobs(campaign, args.gpus, command, training=original["jobs"])
    new_reference = {
        **original, "campaign": output.name, "rebuild_of": original["campaign"],
        "jobs": {job: verified_rebuild_record(output, job, original, data_path) for job in original["jobs"]},
    }
    path = output / "decode-reference.json"
    write_result(path, new_reference)
    return new_reference, path


def rebuilt_checkpoint_files(campaign, job_ids):
    files, steps, errors = [], {}, {}
    for job in job_ids:
        run = campaign.output / "runs" / job
        try:
            payload = checkpoint_files(campaign, [{"directory": str(run)}])
            weights = next((path for path in payload if path.name == "model.safetensors"), None)
            if weights is None:
                continue
            steps[job] = int(weights.parent.name.removeprefix("step-"))
            files.extend(payload)
        except (OSError, ValueError, KeyError) as exc:
            errors[job] = str(exc)
    reference = campaign.output / "decode-reference.json"
    if reference.is_file() and files:
        files.append(reference)
    return files, steps, errors


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


def validate_frozen(model, data, device, result):
    full_length = max(row["bytes"] for row in data.docs["val"])
    result["validation"] = [
        {"context": length, **score(model, data, "val", length, batch, device)}
        for length, batch in ((256, 32), (full_length, 4))
    ]
    if not all(math.isfinite(item["nll"]) for item in result["validation"]):
        raise ValueError("nonfinite validation NLL")
    result["nll_difference_from_checkpoint"] = result["validation"][0]["nll"] - result["checkpoint_nll_t256"]
    if abs(result["nll_difference_from_checkpoint"]) > .002:
        raise ValueError("T256 NLL did not reproduce the saved checkpoint")


@torch.inference_mode()
def paired_input(length, new_bytes, device):
    rng = torch.Generator().manual_seed(17634 + length)
    tokens = torch.randint(0, 256, (1, length + new_bytes), generator=rng).to(device)
    return tokens[:, :length], tokens[:, length:]


def evaluate(model, data, device, result_path, result, *, lengths=LENGTHS, new_bytes=NEW_BYTES):
    decoder = TransformerDecode(model) if result["variant"] == "transformer_medium" else RecurrentDecode(model)
    validate_frozen(model, data, device, result)
    write_result(result_path, result)
    for length in lengths:
        prefix, continuation = paired_input(length, new_bytes, device)
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


def evaluate_optimized(model, data, device, result_path, result, local_nll, *,
                       lengths=LENGTHS, new_bytes=NEW_BYTES, compile_backend="inductor"):
    validate_frozen(model, data, device, result)
    result["mode_errors"] = {}
    write_result(result_path, result)

    def measure(name, adapter, *, compiled=False):
        rows = []
        try:
            decoder = CompiledDecode(adapter, backend=compile_backend) if compiled else adapter
            for length in lengths:
                prefix, continuation = paired_input(length, new_bytes, device)
                with attention_context(device.type == "cpu"):
                    cold = cold_step_seconds(decoder, prefix, continuation) if compiled else None
                    parity = verify_logits(decoder, prefix, continuation)
                    timed = benchmark(decoder, prefix, new_bytes)
                    operations = None
                    if length == lengths[-1]:
                        try:
                            operations = profile_steps(decoder, prefix, continuation)
                        except Exception as exc:
                            operations = {"error": f"{type(exc).__name__}: {exc}"[:4000]}
                rows.append({"mode": name, "parity": parity, "first_step_seconds": cold,
                             "backend": getattr(decoder, "backend", compile_backend) if compiled else "eager",
                             "profiler": operations, **timed})
                print(f"{name} seed {result['seed']} prefixo {length}: "
                      f"{timed['total_seconds'] * 1000:.2f} ms totais.", flush=True)
        except Exception as exc:
            if not compiled:
                raise
            result["mode_errors"][name] = f"{type(exc).__name__}: {exc}"[:4000]
            print(f"{name} excluído após erro de compilação, paridade ou medição.", flush=True)
        else:
            result["profiles"].extend(rows)
        write_result(result_path, result)

    if result["variant"] == "transformer_medium":
        measure("transformer_global_eager", TransformerDecode(model))
        local = LocalTransformerDecode(model)
        length = result["validation"][1]["context"]
        val = score(model, data, "val", length, 4, device)
        if not math.isfinite(val["nll"]) or abs(val["nll"] - local_nll) > .002:
            raise ValueError("local attention did not reproduce the 0035 CPU validation")
        result["local_validation"] = {"context": length, "reference_nll": local_nll, **val}
        measure("transformer_local_eager", local)
        measure("transformer_local_inductor", local, compiled=True)
    else:
        recurrent = RecurrentDecode(model)
        measure("recurrent_eager", recurrent)
        measure("recurrent_inductor", recurrent, compiled=True)
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
        "checkpoint_nll_t256": job["nll_t256"], "weight_sha256": job["weight_sha256"],
        "dataset_sha256": data.fingerprint, "environment": environment(),
        "attention": {"mode": "PyTorch eager SDPA auto", "flash": torch.backends.cuda.flash_sdp_enabled(),
                      "efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                      "math": torch.backends.cuda.math_sdp_enabled(),
                      "cudnn": torch.backends.cuda.cudnn_sdp_enabled()},
    }
    output = Path(args.output) / (args.job + ".json")
    if args.optimize:
        evaluate_optimized(model, data, device, output, result, ref["local_validation_nll"][str(job["seed"])])
    else:
        evaluate(model, data, device, output, result)


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


def optimized_comparisons(rows):
    pairs = []
    for seed in SEEDS:
        for length in LENGTHS:
            measured = {p["mode"]: p for row in rows if row["seed"] == seed and row["status"] == "complete"
                        for p in row["profiles"] if p["prefix_bytes"] == length}
            recurrent = [p for name, p in measured.items() if name.startswith("recurrent_")]
            transformer = [p for name, p in measured.items() if name.startswith("transformer_")]
            if not recurrent or not transformer:
                continue
            model = min(recurrent, key=lambda p: p["total_seconds"])
            reference = min(transformer, key=lambda p: p["total_seconds"])
            pairs.append({
                "seed": seed, "prefix_bytes": length, "all_modes_measured": len(measured) == 5,
                "recurrent_mode": model["mode"], "reference_mode": reference["mode"],
                "total_speedup": reference["total_seconds"] / model["total_seconds"],
                "decode_speedup": reference["decode_seconds"] / model["decode_seconds"],
            })
    long = [p for p in pairs if p["prefix_bytes"] == LENGTHS[-1]]
    return {
        "pairs": pairs,
        "twofold_long_generation": len(long) == len(SEEDS) and all(
            p["all_modes_measured"] and p["total_speedup"] >= 2 for p in long),
        "scope": "best qualified mode per model family, FP32 batch 1, synthetic greedy generation",
    }


def run_campaign(args, ref):
    budget = SessionBudget(args.seconds, 300, elapsed=time.monotonic() - args.started_at)
    campaign = SessionCampaign(ROOT, args.output, device="cuda", gpus=args.gpus, budget=budget)
    rebuild = args.rebuild
    optimize = getattr(args, "optimize", False)
    original = ref
    campaign.manifest.update(
        cycle="decode_step_optimization_v1" if optimize else "cached_decode_rebuild_v1" if rebuild else "cached_decode_0033_v1",
        optimizer_steps=0, source_campaign=ref["campaign"],
        dataset_sha256=ref["dataset_sha256"], reference=ref, operator=read_json(args.operator_metadata),
        limitations="Synthetic generation timing, TinyStories development quality. Two paired seeds. Shared host.",
    )
    error = None
    try:
        reference_path = args.reference
        if not rebuild:
            weights = locate_weights(args.weights, campaign.output.parent, Path("/kaggle/input"), campaign=ref["campaign"])
            if optimize and weights.is_file() and sha256_file(weights) != ref["checkpoint_archive_sha256"]:
                raise ValueError("checkpoint ZIP differs from the audited 0035 archive")
            for job in ref["jobs"].values():
                read_weight(weights, job)
            print(f"Quatro checkpoints verificados em {weights}.", flush=True)
        data = prepare(args.cache, budget=budget)
        if data.fingerprint != ref["dataset_sha256"]:
            raise ValueError("prepared dataset differs from 0033")
        if rebuild:
            campaign.manifest["status"] = "rebuilding"
            campaign.save()
            ref, reference_path = rebuild_reference(campaign, args, original, data)
            weights = campaign.output
            campaign.manifest["reference"] = ref
            print("Quatro pesos novos verificados. Iniciando geração em processos novos.", flush=True)
        campaign.manifest["weight_source"] = str(weights)
        campaign.manifest["status"] = "running"
        campaign.save()
        def command(job):
            call = [__file__, "--reference", reference_path, "--output", args.output,
                    "--weights", weights, "--data", data.root, "--job", job]
            return [*call, "--optimize"] if optimize else call
        if optimize:
            with private_compiler_cache(campaign):
                paired_jobs(campaign, args.gpus, command)
        else:
            paired_jobs(campaign, args.gpus, command)
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
        summary = optimized_comparisons(rows) if optimize else comparisons(rows)
        if optimize and campaign.manifest["status"] == "complete" and any(row["mode_errors"] for row in rows):
            campaign.manifest["status"] = "partial_compile"
        write_json(campaign.output / "comparison.json", summary)
        campaign.manifest.update(runs=rows, session_budget=budget.snapshot())
        if rebuild or optimize:
            try:
                discard_compiler_cache(campaign)
            except (OSError, RuntimeError, ValueError) as exc:
                campaign.manifest["cache_cleanup_error"] = str(exc)
        if rebuild:
            checkpoint_paths, steps, errors = rebuilt_checkpoint_files(campaign, original["jobs"])
            campaign.manifest.update(archived_steps_by_job=steps, optimizer_steps=sum(steps.values()),
                                     checkpoint_errors=errors)
            if checkpoint_paths:
                campaign.save()
                try:
                    archive_files(campaign.output.parent / (campaign.output.name + "-checkpoints.zip"),
                                  checkpoint_paths, base=campaign.output, stored=True)
                except (OSError, ValueError) as exc:
                    error = error or exc
                    campaign.manifest.update(status="failed", checkpoint_export_error=str(exc))
        campaign.save()
        result = campaign.output.parent / (campaign.output.name + "-results.zip")
        files = list(campaign.output.glob("*.json")) + list((campaign.output / "logs").glob("*.log"))
        files.extend(path for path in (campaign.output / "runs").rglob("*")
                     if path.is_file() and path.suffix in (".json", ".jsonl", ".yaml"))
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
    parser.add_argument("--train-job")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--cache")
    parser.add_argument("--operator-metadata")
    parser.add_argument("--gpus", nargs=2)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--started-at", type=float)
    args = parser.parse_args()
    if args.optimize and (args.rebuild or args.train_job):
        parser.error("decode optimization uses frozen weights")
    ref = read_json(args.reference)
    if ref["revision"] != source_provenance()["git"]["commit"]:
        raise ValueError("frozen worker revision differs")
    verified_source(ROOT, ref["source_sha256"])
    if args.train_job:
        if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
            raise RuntimeError("rebuild worker requires one isolated T4")
        rebuild_job(Path(args.output).resolve(), args.train_job, ref, args.data)
    elif args.job:
        run_worker(args, ref)
    else:
        run_campaign(args, ref)


if __name__ == "__main__":
    main()
