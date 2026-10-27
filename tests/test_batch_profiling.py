from pathlib import Path
import json
import pytest
import torch
import yaml

from llm2.data import PreparedData, RandomBatchStream, prepare_data
from llm2.research.campaign import (
    campaign_config,
    parse_batch_matrix,
    parser,
    run_batch_profile,
)
from llm2.research.presentation import batch_comparison_text
from test_training import training_config


def test_campaign_config_batch_overrides(training_config, tmp_path):
    cfg_default = campaign_config(
        training_config, "baseline_small", training_config.data.path,
        tmp_path / "default", seed=42, steps=10, device="cpu", max_seconds=0
    )
    assert cfg_default.data.batch_size == training_config.data.batch_size
    assert cfg_default.train.grad_accum_steps == training_config.train.grad_accum_steps

    cfg_custom = campaign_config(
        training_config, "baseline_small", training_config.data.path,
        tmp_path / "custom", seed=42, steps=10, device="cpu", max_seconds=0,
        batch_size=16, grad_accum_steps=2
    )
    assert cfg_custom.data.batch_size == 16
    assert cfg_custom.train.grad_accum_steps == 2

    with pytest.raises(ValueError, match="positive integer"):
        campaign_config(training_config, "baseline_small", training_config.data.path,
                        tmp_path / "err", seed=42, steps=10, device="cpu", max_seconds=0,
                        batch_size=-1)


def test_parse_batch_matrix():
    matrix = parse_batch_matrix(["8x4", "16X2", "32*1"])
    assert matrix == [(8, 4), (16, 2), (32, 1)]

    with pytest.raises(ValueError, match="Formato de lote inválido"):
        parse_batch_matrix(["invalid"])

    with pytest.raises(ValueError, match="positivos"):
        parse_batch_matrix(["0x4"])


def test_batch_comparison_text():
    sample_rows = [
        {
            "model": "baseline_small",
            "batch_size": 8,
            "grad_accum_steps": 4,
            "tokens_per_second": 150000.0,
            "mean_step_seconds": 0.0546,
            "peak_vram_allocated_bytes": 200 * 1024 * 1024,
        },
        {
            "model": "flrm_compressed_grouped_no_jepa_k1",
            "batch_size": 16,
            "grad_accum_steps": 2,
            "tokens_per_second": 120000.0,
            "mean_step_seconds": 0.0683,
            "peak_vram_allocated_bytes": 350 * 1024 * 1024,
        },
    ]
    text = batch_comparison_text(sample_rows)
    assert "Transformer pequeno" in text
    assert "FLRM agrupado sem JEPA (K=1)" in text
    assert "8 × 4" in text
    assert "16 × 2" in text
    assert "150.000" in text


def test_random_batch_stream_invariance_across_batches():
    seed = 1234
    tokens = torch.arange(5000, dtype=torch.int64).numpy()
    seq_len = 32

    # 4 microbatches of 8
    stream1 = RandomBatchStream(tokens, seq_len, batch_size=8, seed=seed)
    targets1 = []
    for _ in range(4):
        b = stream1.next()
        targets1.extend(b["targets"].reshape(-1).tolist())

    # 2 microbatches of 16
    stream2 = RandomBatchStream(tokens, seq_len, batch_size=16, seed=seed)
    targets2 = []
    for _ in range(2):
        b = stream2.next()
        targets2.extend(b["targets"].reshape(-1).tolist())

    # 1 microbatch of 32
    stream3 = RandomBatchStream(tokens, seq_len, batch_size=32, seed=seed)
    targets3 = stream3.next()["targets"].reshape(-1).tolist()

    assert targets1 == targets2 == targets3


def test_run_batch_profile_cpu_smoke(training_config, tmp_path):
    output_dir = tmp_path / "campaign_output"
    config_file = tmp_path / "test_base.yaml"
    config_file.write_text(yaml.safe_dump(training_config.to_dict()), encoding="utf-8")

    args = parser().parse_args([
        "--execute",
        "--config", str(config_file),
        "--device", "cpu",
        "--gpus", "0",
        "--data", str(training_config.data.path),
        "--output", str(output_dir),
        "--models", "baseline_small",
        "--batch-matrix", "2x2", "4x1",
        "--warmup-steps", "0",
        "--measure-steps", "1",
        "--seeds", "42",
        "--profile-batches",
    ])

    res = run_batch_profile(args)
    assert res["status"] == "complete"
    assert len(res["results"]) == 2
    assert (output_dir / "batch_profiles.json").is_file()
    assert (output_dir / "batch_profile_report.md").is_file()
    assert Path(res["archive"]).is_file()
