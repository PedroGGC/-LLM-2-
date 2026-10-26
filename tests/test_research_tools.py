import json
from pathlib import Path

import pytest
import torch

from llm2.cli import main
from llm2.config import ResearchConfig
from llm2.curriculum.generators import generate_problem
from llm2.data import TextTokenizer, write_json
from llm2.model import TransformerLM
from llm2.research.budget import plan_budget
from llm2.research.evaluators import evaluate_reasoning, benchmark_inference
from llm2.research.profiling import profile_step
from llm2.research.reporting import generate_report, learning_flops
from llm2.trainer import Trainer
from test_training import training_config


def test_budget_planner_and_dry_run_cli(capsys):
    assert learning_flops({"training_flops_estimate": None}) is None
    budget = plan_budget()
    assert budget["usable_credits"] == 160
    assert budget["reserved_credits"] == 40
    assert budget["planned_hours"] == pytest.approx(160/5.3)
    with pytest.raises(ValueError):
        plan_budget(safety_fraction=0)
    main(["plan-budget"])
    assert json.loads(capsys.readouterr().out)["is_estimate"]
    main(["experiment", "--config", "configs/cpu_smoke.yaml"])
    assert json.loads(capsys.readouterr().out)["execution"] == "not_started"
    main(["profile", "--config", "configs/cpu_smoke.yaml", "--output", "unused"])
    assert json.loads(capsys.readouterr().out)["execution"] == "not_started"


def test_inference_measurement_and_generated_answer_eval(training_config):
    model = TransformerLM(training_config.model)
    tokenizer = TextTokenizer()
    result = benchmark_inference(model, tokenizer, "A", max_new_tokens=2)
    assert result["generated_tokens"] == 2
    assert 0 < result["time_to_first_token_seconds"] <= result["total_latency_seconds"]
    assert result["tokens_per_second"] > 0
    assert result["generation_flops_estimate"] > 0
    assert result["peak_vram_allocated_bytes"] is None
    assert result["weight_bytes_calculated"] == sum(p.numel()*p.element_size() for p in model.parameters())
    training_config.model.max_seq_len = 128
    model = TransformerLM(training_config.model)
    seen = [generate_problem("arithmetic", 1, 1).to_dict()]
    quality = evaluate_reasoning(model, tokenizer, examples=1, max_new_tokens=1, seen_examples=seen)
    assert quality["kind"] == "generated_answer_exact_match"
    assert quality["groups"]["recent_training_instances"]["examples"] == 1
    assert quality["groups"]["unseen_test_and_families"]["examples"] == 9
    assert 0 <= quality["groups"]["unseen_test_and_families"]["exact_match"] <= 1


def test_profile_and_reporting_are_real_files(training_config, tmp_path):
    cfg = training_config
    with Trainer(cfg) as trainer:
        trainer.fit(until_step=2)
        result = benchmark_inference(trainer.model, trainer.data.tokenizer, "A", max_new_tokens=2)
        write_json(trainer.output_dir / "benchmark.json", result)
    report = generate_report([cfg.train.output_dir], tmp_path / "report", target_loss=100)
    assert report["comparison_controls"]["same_dataset"]
    assert (tmp_path / "report/series.csv").read_text().startswith("run,step,")
    for name in ("loss_vs_flops", "quality_vs_flops", "parameters", "memory", "throughput"):
        assert "<svg" in (tmp_path / f"report/{name}.svg").read_text()
    with pytest.raises(FileExistsError):
        generate_report([cfg.train.output_dir], tmp_path / "report")
    result["checkpoint_model_sha256"] = "stale"
    write_json(trainer.output_dir / "benchmark.json", result)
    with pytest.raises(ValueError, match="stale evaluation"):
        generate_report([cfg.train.output_dir], tmp_path / "stale-report")
    traced = profile_step(cfg, tmp_path / "profile")
    assert traced["profiler_supported_ops_flops"] > 0
    assert (tmp_path / "profile/trace.json").is_file()


def test_research_cli_inference_and_legacy_weights(training_config, tmp_path, capsys):
    cfg = training_config
    cfg.research = ResearchConfig(architecture="recursive", jepa_weight=0.1, mtp_tokens=1, mtp_weight=0.1)
    with Trainer(cfg) as trainer:
        trainer.fit(until_step=1)
        path = str(trainer.last_checkpoint)
    main(["evaluate", "--checkpoint", path, "--max-batches", "1"])
    assert json.loads(capsys.readouterr().out)["validation_perplexity"] > 0
    main(["generate", "--checkpoint", path, "--prompt", "A", "--max-new-tokens", "1"])
    assert capsys.readouterr().out.startswith("A")
    main(["benchmark", "--checkpoint", path, "--prompt", "A", "--max-new-tokens", "1", "--output", str(tmp_path / "benchmark.json")])
    assert json.loads(capsys.readouterr().out)["internal_iterations"] == 2
