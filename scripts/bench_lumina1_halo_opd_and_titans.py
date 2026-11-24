"""Prepared (DO NOT RUN automatically) GPU verification harness for:
  1. Lumina 1: HALO Layer Sensitivity Probe + Global/KV-INT4 + Top-K KL + OPD + QuaRot W4A16 (Kaggle 2x T4 / Colab).
  2. Titans / TTT: Surprise-gated 2-layer MLP memory vs Gated DeltaNet-2 microbenchmark (Colab L4 / A100).

Usage (only when explicitly authorized):
  python scripts/bench_lumina1_halo_opd_and_titans.py --mode halo_opd --teacher <hf_model_id>
  python scripts/bench_lumina1_halo_opd_and_titans.py --mode titans_colab --seq-len 2048
"""

from __future__ import annotations

import argparse
import json
import time

import torch


def run_halo_opd_smoke(teacher_id: str, num_global: int = 9, kv_quant: str = "int4") -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llm2.architectures.quarot import quantize_model_quarot_int4
    from llm2.research.distill_convert import transplant_teacher_weights
    from llm2.research.lumina_3b_pipeline import (
        _extract_rope_theta,
        build_lumina_3b_config_from_hf,
        build_student_for_surgery,
        probe_halo_layer_sensitivity,
        run_on_policy_distill,
    )
    from llm2.research.ringue_eval import _measure_native_decode, _model_storage_bytes

    dev0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dev1 = torch.device("cuda:1" if torch.cuda.device_count() >= 2 else dev0.type)
    tok = AutoTokenizer.from_pretrained(teacher_id)
    teacher = AutoModelForCausalLM.from_pretrained(teacher_id, torch_dtype=torch.float16).to(dev0).eval()
    teacher.requires_grad_(False)

    probe_txt = (
        "In mathematics and computer science, hybrid linear attention combines recurrent delta memory "
        "with global softmax checkpoints to preserve exact long-range needle retrieval."
    )
    probe_ids = torch.tensor([tok.encode(probe_txt, add_special_tokens=False)[:64]], dtype=torch.long, device=dev0)
    halo = probe_halo_layer_sensitivity(teacher, probe_ids, num_global_layers=num_global, always_keep=(0,), use_amp=True)

    m_cfg, r_cfg = build_lumina_3b_config_from_hf(
        teacher.config,
        swa_levels=halo["selected_global_levels"],
        swa_global=True,
        kv_quant=kv_quant,
        out_gate_rank=256,
        assoc_channel=False,
        swa_rope=True,
        gradient_checkpointing=True,
    )
    student = build_student_for_surgery(m_cfg, r_cfg, dtype=torch.float16)
    transplant_teacher_weights(teacher, student, quiet_init_gates=True)

    opd_prompts = [probe_ids[:, :24].cpu() for _ in range(4)]
    opd_res = run_on_policy_distill(
        teacher,
        student,
        iter(opd_prompts),
        dev0=dev0,
        dev1=dev1,
        max_steps=4,
        rollout_tokens=16,
        kl_topk=128,
        lr=5e-5,
        precision="fp16" if dev1.type == "cuda" else "fp32",
        freeze_deltanet_proj=True,
        lora_rank=8,
    )

    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    student = student.to(dev0, dtype=torch.float16).eval()
    fp16_dec = _measure_native_decode(student, probe_ids[:, :32].to(dev0), 32, dev0, compile_fractal=False)
    q_model, q_info = quantize_model_quarot_int4(student, group_size=32, act_bits=16, inplace=True, use_triton=True)
    q4_dec = _measure_native_decode(q_model.eval(), probe_ids[:, :32].to(dev0), 32, dev0, compile_fractal=False)

    return {
        "teacher": teacher_id,
        "rope_theta_extracted": _extract_rope_theta(m_cfg),
        "student_params": student.parameter_formula(),
        "halo_selected_global_levels": halo["selected_global_levels"],
        "state_mib_16k": round(q_model.state_bytes_per_sequence(seq_len=16384) / (1024**2), 2),
        "opd_smoke": opd_res,
        "fp16_decode": fp16_dec,
        "q4_storage_gib": round(_model_storage_bytes(q_model) / (1024**3), 3),
        "q4_decode_w4a16": q4_dec,
        "q4_replaced": q_info["replaced_count"],
    }


def run_titans_colab_ablation(d_model: int = 1024, n_heads: int = 8, seq_len: int = 2048, steps: int = 20) -> dict:
    """Head-to-head Titans/TTT MLP memory vs Gated DeltaNet-2 on Colab L4/A100 (BF16)."""
    from llm2.architectures.fractal_delta import FractalDeltaCell
    from llm2.architectures.titans_ttt import TitansMemoryCell
    from llm2.config import ModelConfig, ResearchConfig

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (dev.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    titans = TitansMemoryCell(d_model=d_model, n_heads=n_heads, mem_hidden_mult=2, chunk_size=32).to(dev)
    cfg = ModelConfig(vocab_size=256, d_model=d_model, n_layers=1, n_heads=n_heads, hidden_dim=d_model * 2, max_seq_len=seq_len)
    res = ResearchConfig(architecture="fractal_delta", scale_conditioning=False, share_scales=False, fractal_decoupled_gates=True)
    delta = FractalDeltaCell(cfg, res).to(dev)

    x = torch.randn(2, seq_len, d_model, device=dev, dtype=dtype)
    report: dict = {"device": str(dev), "dtype": str(dtype), "seq_len": seq_len, "d_model": d_model}

    for name, mod in (("titans_ttt", titans), ("gated_deltanet2", delta)):
        opt = torch.optim.AdamW(mod.parameters(), lr=1e-3)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        last_loss = 0.0
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=dtype, enabled=(dev.type == "cuda")):
                out = mod(x)[0]
                loss = (out - x).square().mean()
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().item())
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        dt = time.perf_counter() - t0
        report[name] = {
            "steps": steps,
            "tok_per_s": round((2 * seq_len * steps) / max(dt, 1e-9), 1),
            "peak_vram_mib": round(torch.cuda.max_memory_allocated(dev) / (1024**2), 1) if dev.type == "cuda" else None,
            "state_mib_per_seq": round(mod.state_bytes_per_seq() / (1024**2), 2) if hasattr(mod, "state_bytes_per_seq") else None,
            "final_recon_loss": round(last_loss, 5),
        }
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("halo_opd", "titans_colab"), required=True)
    ap.add_argument("--teacher", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--seq-len", type=int, default=2048)
    args = ap.parse_args()
    out = run_halo_opd_smoke(args.teacher) if args.mode == "halo_opd" else run_titans_colab_ablation(seq_len=args.seq_len)
    print(json.dumps(out, indent=2))
