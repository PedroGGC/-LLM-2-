"""Explicit operational extension over an unchanged, qualified scientific checkout.

Execute as a file with PYTHONPATH pointing at the scientific worker. This file
does not import llm2 from its own operator checkout. Both sources are exported.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import sys
import threading
import time
import zipfile

SCIENTIFIC_REVISION = "1c8ce2abb8f47d3167454b3503598854b7756dad"
PLATFORM_SHA256 = "cf281d13a522724eabfe84e1e1a59b4244b5093971e9067867030db5b1894da3"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SlotCaches:
    """One family per physical GPU; never remove a different lane's cache."""
    def __init__(self, campaign, remove_cache, tree_bytes):
        self.campaign, self.remove_cache, self.tree_bytes = campaign, remove_cache, tree_bytes
        self.locks = {str(slot): threading.Lock() for slot in campaign.gpus}
        self.active, self.families = set(), {}

    def run(self, slot, family, function):
        slot = str(slot)
        with self.locks[slot]:
            path = self.campaign.output / "compiler-cache" / f"gpu-{slot}"
            previous = self.families.get(slot)
            if previous != family and path.exists():
                row = self.remove_cache(self.campaign.output.parent, path,
                                        guard=lambda: [slot] if slot in self.active else [])
                with self.campaign.lock:
                    self.campaign.manifest.setdefault("operator_cache_evictions", []).append(
                        {**row, "slot": slot, "previous_family": previous, "next_family": family})
                print(f"Cache GPU{slot}: troca de variante, {row['bytes']/1024**2:.1f} MiB removidos.", flush=True)
            self.families[slot] = family
            self.active.add(slot)
            began = time.monotonic()
            try:
                return function()
            finally:
                self.active.discard(slot)
                with self.campaign.lock:
                    self.campaign.manifest.setdefault("operator_commands", []).append({
                        "slot": slot, "family": family, "reused_family": previous == family,
                        "wall_seconds": time.monotonic()-began, "cache_bytes_after": self.tree_bytes(path)})
                    self.campaign.save()


def family_for(study, arguments):
    if "--config" not in arguments:
        return "preflight"
    cfg = study.yaml.safe_load(Path(arguments[arguments.index("--config")+1]).read_text(encoding="utf-8"))
    value = {"model": cfg["model"], "research": cfg["research"],
             "precision": cfg["train"]["precision"], "context": cfg["data"]["seq_len"],
             "batch": cfg["data"]["batch_size"]}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def campaign_class(study, metadata, operator_root):
    base = study.SessionCampaign
    class OperationalCampaign(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.manifest["operator"] = metadata
            self.manifest["operator_policy"] = "variant-scoped caches; grouped tuning; pairwise train then isolated profiles"
            self.slot_caches = SlotCaches(self, study.remove_cache, study.tree_bytes)
            report = self.output / "report"
            report.mkdir()
            with zipfile.ZipFile(report/"operator-source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, expected in metadata["files_sha256"].items():
                    path = operator_root/name
                    if path.is_symlink() or not path.resolve().is_relative_to(operator_root) or digest(path) != expected:
                        raise ValueError("operator source/snapshot mismatch")
                    archive.write(path, name)
            study.write_json(self.output/"operator.json", metadata)
            self.save()

        def command(self, arguments, log_name, *, gpu=None, job=None, timeout=None):
            call = lambda: super(OperationalCampaign, self).command(arguments, log_name, gpu=gpu, job=job, timeout=timeout)
            if gpu is None:
                return call()
            return self.slot_caches.run(gpu, family_for(study, arguments), call)
    return OperationalCampaign


def profile_reserve(profiles, *, smoke=False):
    if smoke:
        return 0.
    # Cold on every GPU/variant is intentionally conservative even though four
    # profiles can retain a hot cache after their training pair. Include imports,
    # measurement, validation and up to30s of energy telemetry, not just warmup.
    return max(1200., 1.35*sum(p["warmup_seconds"] + 30*p["median_step_seconds"]
                              + p["full_validation_seconds"] + 30. + 60. for p in profiles.values()))


def valid_frozen_tuning(study, value, campaign, data, rule):
    if (value.get("protocol_id") != study.PROTOCOL_ID or value.get("selected_rule") != rule
            or value.get("dataset_sha256") != data.fingerprint
            or value.get("provenance", {}).get("source_sha256") != campaign.manifest["provenance"]["source_sha256"]):
        raise ValueError("frozen tuning source/protocol/dataset/rule mismatch")
    rates = value.get("learning_rates", {})
    if set(rates) != set(study.VARIANTS) or any(v not in (.0002, .0004) for v in rates.values()):
        raise ValueError("invalid frozen rates")
    return rates


def experiment(study, campaign, args):
    if args.engineering_smoke:
        rule = "mlstm"
    else:
        value = json.loads(Path(args.platform_report).read_text(encoding="utf-8"))
        rule = study.validate_report(value)  # Original strict source/environment checks.
        if rule != "mlstm":
            raise ValueError("this Stage B release uses the rule qualified in campaign0025")
        study.write_json(campaign.output/"platforms.json", value)
    campaign.manifest["selected_rule"] = rule
    data = study.PreparedData(args.data) if args.engineering_smoke and args.data else study.PreparedData(study.prepare(campaign, args))
    campaign.manifest.update(dataset_sha256=data.fingerprint, dataset_manifest=data.manifest, dataset_path=str(data.root))
    campaign.save()
    tune_path = study.tuning_dataset(data.root, campaign.output/"tuning-data")
    frozen = json.loads(Path(args.tuning_report).read_text(encoding="utf-8")) if args.tuning_report else None
    rates = valid_frozen_tuning(study, frozen, campaign, data, rule) if frozen else {}
    tuning = list(frozen.get("tuning", [])) if frozen else []
    profiles, pilot_configs = {}, {}
    # Isolation is maintained; only task order changes. Rate sweeps immediately
    # follow their own pilot to reuse the active variant's compiled code.
    for variant in study.VARIANTS:
        for slot in args.gpus:
            cfg = study.configuration(variant, rule, data.root, campaign.output/"pilots"/f"{variant}-{slot}", 241,
                                      smoke=args.engineering_smoke, device=args.device)
            profiles[variant, slot] = study.worker(campaign, args, cfg, f"pilot-{variant}-{slot}", "pilot", slot)
            pilot_configs[variant, slot] = cfg
            campaign.manifest["pilot_profiles"] = [{"variant":v, "slot":s, **p} for (v,s),p in profiles.items()]
            campaign.save()
        if frozen:
            continue
        candidates = []
        for rate in (.0002, .0004):
            cfg = study.configuration(variant, rule, tune_path, campaign.output/"tuning"/f"{variant}-{rate}", 901,
                                      lr=rate, smoke=args.engineering_smoke, device=args.device,
                                      steps=4 if args.engineering_smoke else 750)
            estimate = profiles[variant,args.gpus[0]]
            campaign.budget.require(0 if args.engineering_smoke else
                1.35*(estimate["warmup_seconds"] + 750*estimate["median_step_seconds"] + estimate["full_validation_seconds"] + 60))
            result = study.worker(campaign, args, cfg, f"tune-{variant}-{rate}", "tune", args.gpus[0])
            nll = result["evaluations"][-1]["validation_loss"]
            candidates.append((nll, rate))
            tuning.append({"variant":variant, "lr":rate, "validation_loss":nll})
            campaign.manifest["tuning"] = tuning
            campaign.save()
        rates[variant] = min(candidates)[1]
        campaign.manifest["learning_rates"] = dict(rates)
        campaign.save()
    campaign.manifest.update(learning_rates=rates, tuning=tuning)
    campaign.save()

    def concurrent(slot):
        return slot, study.worker(campaign, args, pilot_configs["recurrent",slot], f"concurrent-{slot}", "pilot", slot,
                                  output=campaign.output/"pilots"/f"concurrent-{slot}")
    with ThreadPoolExecutor(max_workers=2) as pool:
        parallel = dict(pool.map(concurrent, args.gpus))
    contention = max(1., max(parallel[s]["median_step_seconds"]/profiles["recurrent",s]["median_step_seconds"] for s in args.gpus))
    jobs = study.ordered_jobs(rule, data.root, campaign.output, args.gpus, device=args.device,
                              smoke=args.engineering_smoke, seeds=args.seeds)
    for job in jobs:
        p = profiles[job["variant"],job["gpu_slot"]]
        job["estimated_job_seconds"] = 1.35*(contention*p["median_step_seconds"]*job["max_steps"]
            + p["warmup_seconds"] + 2.6*p["full_validation_seconds"] + (0 if args.engineering_smoke else 60))
    post = profile_reserve(profiles, smoke=args.engineering_smoke)
    admitted, pending = study.schedule_batches(jobs, args.gpus, remaining_seconds=campaign.budget.work_remaining,
                                               post_seconds=post)
    loads=[sum(j["estimated_job_seconds"] for j in admitted if j["gpu_slot"]==s) for s in args.gpus]
    remaining=max(loads,default=0.)+post
    campaign.manifest.update(runs=jobs, admitted_job_ids=[j["id"] for j in admitted], pending_job_ids=[j["id"] for j in pending],
        contention_factor=contention, isolated_profiles=[], forecast_margin=1.35, post_reserve_seconds=post,
        planned_tuning_seconds=0, tuning_completed_before_admission=True, requested_seeds=list(args.seeds),
        remaining_protocol_seeds=[s for s in study.SEEDS if s not in args.seeds],
        forecast_total_seconds=campaign.budget.elapsed+remaining+campaign.budget.reserve,
        forecast_remaining_seconds=remaining+campaign.budget.reserve)
    campaign.save()
    print(f"Admitidos {len(admitted)}/{len(jobs)} treinos; perfis finais reservados: {post/60:.1f} min.", flush=True)
    print(f"Previsão conservadora total: {campaign.manifest['forecast_total_seconds']/60:.1f} min "
          "(inclui preparação/triagem e exportação); meta120/teto180 min.", flush=True)
    if not admitted:
        raise study.SessionBudgetExceeded("no whole seed fits; pilots/tuning exported; horizons unchanged")
    needed = sum(j["parameters"]*16*4 for j in admitted) + (0 if args.engineering_smoke else 2*1024**3)
    if shutil.disk_usage(campaign.output).free < needed:
        raise OSError("insufficient space including 2GiB reserve")

    done = 0
    queues = {s:[j for j in admitted if j["gpu_slot"]==s] for s in args.gpus}
    measured = set()
    def train(job):
        campaign.budget.require(job["estimated_job_seconds"] + (0 if args.engineering_smoke else 60))
        cfg = study.configuration(job["variant"], rule, data.root, job["directory"], job["seed"], lr=rates[job["variant"]],
                                  smoke=args.engineering_smoke, device=args.device)
        job.update(status="training", config_resolved=cfg.to_dict())
        campaign.save()
        result = study.worker(campaign, args, cfg, job["id"], "train", job["gpu_slot"], job=job)
        if (result["source_sha256"] != campaign.manifest["provenance"]["source_sha256"]
                or result["dataset_sha256"] != data.fingerprint):
            raise ValueError("worker source/dataset mismatch")
        if result["stop_reason"] != "max_steps" or result["step"] != job["max_steps"]:
            raise study.SessionBudgetExceeded("fixed horizon not completed")
        job.update(status="complete",checkpoint=result["checkpoint"],checkpoint_model_sha256=result["checkpoint_model_sha256"],
            evaluations=result["evaluations"],counters=result["counters"],flops_estimate=result["flops_estimate"],
            worker_environment=result["environment"])
        campaign.save()
        return job

    def profile(job, slot):
        cfg = study.configuration(job["variant"], rule, data.root, job["directory"], job["seed"], lr=rates[job["variant"]],
                                  smoke=args.engineering_smoke, device=args.device)
        result = study.worker(campaign, args, cfg, f"post-{job['variant']}-{slot}", "pilot", slot,
                              output=campaign.output/"post"/f"{job['variant']}-{slot}", weights=job["checkpoint"],
                              energy_seconds=0 if args.engineering_smoke else 30.)
        campaign.manifest["isolated_profiles"].append({"variant":job["variant"],"slot":slot,**result})
        campaign.save()

    while any(queues.values()):
        pair = [queues[s].pop(0) for s in args.gpus if queues[s]]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(train, job) for job in pair]
            try:
                completed = [f.result() for f in as_completed(futures)]
            except BaseException:
                campaign.abort()
                raise
        done += len(completed)
        if done == (len(admitted)+1)//2:
            bundles = study.publish(campaign, args, suffix="midway")
            campaign.manifest["midway_archives"] = [str(p) for p in bundles]
            campaign.save()
        # Both training workers have stopped. All profiles are isolated. Do the
        # native GPU before cross-GPU profiles so its trained variant stays hot.
        first = []
        new_variants = set(measured)
        for job in pair:
            if job["variant"] not in new_variants:
                first.append(job)
                new_variants.add(job["variant"])
        for job in first:
            profile(job, job["gpu_slot"])
        for job in first:
            for slot in args.gpus:
                if slot != job["gpu_slot"]:
                    profile(job, slot)
            measured.add(job["variant"])
    if pending:
        campaign.manifest["status"] = "partial_planned_batches"


def run(worker_source, metadata, operator_root, arguments, *, started=None):
    """Bind explicit lifecycle/ordering hooks; all scientific calls stay original."""
    worker_source, operator_root = Path(worker_source).resolve(), Path(operator_root).resolve()
    from llm2.research import state_study as study
    if study.ROOT.resolve() != worker_source:
        raise ValueError("wrong scientific package imported")
    saved = study.SessionCampaign, study.experiment, study.analyze
    study.SessionCampaign = campaign_class(study, metadata, operator_root)
    study.experiment = lambda campaign,args: experiment(study, campaign, args)
    def analyze(*args, **kwargs):
        return {**saved[2](*args, **kwargs), "operator":metadata}
    study.analyze = analyze
    try:
        if started is not None and "--bootstrap-elapsed-seconds" in arguments:
            index=arguments.index("--bootstrap-elapsed-seconds")+1
            arguments=list(arguments)
            arguments[index]=str(float(arguments[index])+time.monotonic()-started)
        return study.main(arguments)
    finally:
        study.SessionCampaign, study.experiment, study.analyze = saved


def main(argv=None):
    started=time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-source",required=True)
    parser.add_argument("--operator-metadata",required=True)
    parser.add_argument("--operator-root",required=True)
    args, arguments = parser.parse_known_args(argv)
    if "--execute" not in arguments:
        return {"status":"not_started"}
    if "--phase" not in arguments or arguments[arguments.index("--phase")+1] != "experiment":
        raise ValueError("Stage B operator only accepts --phase experiment")
    if "--engineering-smoke" not in arguments:
        if "--platform-report" not in arguments or digest(arguments[arguments.index("--platform-report")+1]) != PLATFORM_SHA256:
            raise ValueError("original campaign0025 platform report required")
    worker = Path(args.worker_source).resolve()
    # When invoked as a file no llm2 package has been imported from this release.
    sys.path.insert(0,str(worker/"src"))
    metadata=json.loads(Path(args.operator_metadata).read_text(encoding="utf-8"))
    if metadata.get("scientific_revision") != SCIENTIFIC_REVISION:
        raise ValueError("qualified scientific revision required")
    return run(worker,metadata,args.operator_root,arguments,started=started)


if __name__ == "__main__":
    main()
