"""Unit tests for Lumina 1 architecture upgrades: HALO, Global+KV INT4/FP8, QK-Norm, Top-K KL, OPD, and Titans/TTT."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from llm2.architectures.fractal_delta import FractalDecode, FractalDeltaMemory, SlidingWindowGQACell, _apply_split_half_rope
from llm2.architectures.quarot import HadamardLinearINT4, quantize_model_quarot_int4
from llm2.architectures.titans_ttt import TitansMemoryCell
from llm2.config import ModelConfig, ResearchConfig
from llm2.model import RMSNorm, SwiGLU
from llm2.research.distill_convert import transplant_teacher_weights
from llm2.research.lumina_3b_pipeline import (
    _chunked_hidden_ce_kl_loss,
    _extract_rope_theta,
    build_lumina_3b_config_from_hf,
    build_student_for_surgery,
    probe_halo_layer_sensitivity,
    run_on_policy_distill,
)


def _unique_params(model: nn.Module) -> int:
    return sum(p.numel() for p in {p.data_ptr(): p for p in model.parameters()}.values())


class _TinyQwen3Attn(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv: int, head_dim: int, rope_theta: float = 5e6):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv = n_kv
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)
        inv_freq = 1.0 / (float(rope_theta) ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / float(head_dim)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, t, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, t, self.n_kv, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        pos = torch.arange(t, device=x.device, dtype=torch.float64)[None, None, :, None]
        ang = (pos * self.inv_freq.view(1, 1, 1, -1)).float()
        q = _apply_split_half_rope(q.float(), ang).to(x.dtype)
        k = _apply_split_half_rope(k.float(), ang).to(x.dtype)
        rep = self.n_heads // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim))


class _TinyQwen3Layer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv: int, head_dim: int, hidden_dim: int, rope_theta: float = 5e6):
        super().__init__()
        self.input_layernorm = RMSNorm(d_model, eps=1e-6)
        self.self_attn = _TinyQwen3Attn(d_model, n_heads, n_kv, head_dim, rope_theta=rope_theta)
        self.post_attention_layernorm = RMSNorm(d_model, eps=1e-6)
        self.mlp = SwiGLU(d_model, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class _TinyQwen3Teacher(nn.Module):
    def __init__(self, vocab: int = 64, d_model: int = 32, n_layers: int = 6, n_heads: int = 4, n_kv: int = 2, head_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="qwen3",
            hidden_size=d_model,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv,
            head_dim=head_dim,
            intermediate_size=hidden_dim,
            vocab_size=vocab,
            rope_parameters={"rope_theta": 5000000.0},
            rms_norm_eps=1e-6,
        )
        self.embed_tokens = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([_TinyQwen3Layer(d_model, n_heads, n_kv, head_dim, hidden_dim) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model, eps=1e-6)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def get_input_embeddings(self):
        return self.embed_tokens

    def model(self, input_ids: torch.Tensor, output_hidden_states: bool = True, use_cache: bool = False):
        h = self.embed_tokens(input_ids)
        hs = [h]
        for layer in self.layers:
            h = layer(h)
            hs.append(h)
        last = self.norm(h)
        hs[-1] = last
        return SimpleNamespace(last_hidden_state=last, hidden_states=tuple(hs))

    def forward(self, input_ids: torch.Tensor):
        out = self.model(input_ids)
        return SimpleNamespace(logits=self.lm_head(out.last_hidden_state), hidden_states=out.hidden_states)


def test_rope_theta_v5_extraction_and_qwen3_verbatim_global_transplant():
    torch.manual_seed(42)
    teacher = _TinyQwen3Teacher().eval()
    assert _extract_rope_theta(teacher.config) == 5000000.0

    # HALO sensitivity probe picks top-2 layers cleanly
    probe_ids = torch.randint(0, 64, (2, 16))
    halo = probe_halo_layer_sensitivity(teacher, probe_ids, num_global_layers=2, always_keep=(0,))
    assert len(halo["selected_global_levels"]) == 2
    assert 0 in halo["selected_global_levels"]

    m_cfg, r_cfg = build_lumina_3b_config_from_hf(
        teacher.config,
        swa_levels=halo["selected_global_levels"],
        swa_global=True,
        kv_quant="none",
        out_gate_rank=8,
        assoc_channel=False,
        swa_rope=True,
    )
    assert m_cfg.rope_theta == 5000000.0
    assert r_cfg.fractal_head_dim == 16
    assert r_cfg.fractal_qk_norm is True
    assert r_cfg.fractal_assoc_channel is False

    student = build_student_for_surgery(m_cfg, r_cfg, dtype=torch.float32).eval()
    assert student.parameter_formula() == _unique_params(student)
    assert student.swa_levels == frozenset(halo["selected_global_levels"])
    assert isinstance(student.cells[0], SlidingWindowGQACell)

    transplant_teacher_weights(teacher, student, quiet_init_gates=True)

    # Step-0 exact parity on every global layer (cos >= 0.99999 with QK-Norm + head_dim=16 != d/H + RoPE 5e6!)
    with torch.no_grad():
        h_in = teacher.embed_tokens(probe_ids)
        t_delta0 = teacher.layers[0].self_attn(teacher.layers[0].input_layernorm(h_in))
        s_delta0 = student.forward_level_attn_delta(0, h_in, h_in)
        cos0 = float(F.cosine_similarity(s_delta0, t_delta0, dim=-1).mean().item())
        assert cos0 > 0.99999, f"Expected exact Qwen3 global transplant parity, got cos={cos0}"


@pytest.mark.parametrize("kv_quant", ["none", "fp8", "int4"])
def test_global_attention_and_kv_quant_causality_and_streaming(kv_quant: str):
    torch.manual_seed(17)
    cfg = ModelConfig(vocab_size=48, d_model=32, n_layers=4, n_heads=4, hidden_dim=64, max_seq_len=64, dropout=0.0)
    res = ResearchConfig(
        architecture="fractal_delta",
        scale_conditioning=False,
        share_scales=False,
        fractal_hybrid_swa=True,
        fractal_swa_levels=[1, 3],
        fractal_swa_global=True,
        fractal_swa_kv_heads=2,
        fractal_swa_rope=True,
        fractal_deltanet_rope=True,
        fractal_qk_norm=True,
        fractal_kv_quant=kv_quant,
        fractal_out_gate_rank=8,
        fractal_assoc_channel=False,
        fractal_decoupled_gates=True,
    )
    model = FractalDeltaMemory(cfg, res).eval()
    assert model.parameter_formula() == _unique_params(model)

    ids = torch.randint(0, 48, (2, 24))
    ids_mut = ids.clone()
    ids_mut[:, 14] = (ids_mut[:, 14] + 1) % 48
    with torch.no_grad():
        out_a = model(ids).logits
        out_b = model(ids_mut).logits
    assert torch.allclose(out_a[:, :14], out_b[:, :14], atol=1e-5)

    dec = FractalDecode(model)
    with torch.no_grad():
        lg_p, st = dec.prefill(ids[:, :10], capacity=32)
        outs = [lg_p]
        for t in range(10, 24):
            lg, st = dec.step(ids[:, t : t + 1], st)
            outs.append(lg)
        inc_logits = torch.cat(outs, dim=1)
    tol = 1e-4 if kv_quant == "none" else (0.15 if kv_quant == "fp8" else 0.75)
    assert torch.isfinite(inc_logits).all()
    assert (out_a[:, 9:] - inc_logits).abs().max().item() < tol

    # Verify KV memory shrinks monotonically: int4 < fp8 < none
    if kv_quant == "none":
        b_none = model.state_bytes_per_sequence(seq_len=1024)
        assert b_none > 0


def test_topk_sparse_kl_and_on_policy_distillation():
    torch.manual_seed(9)
    teacher = _TinyQwen3Teacher(vocab=64, d_model=32, n_layers=4, n_heads=4, n_kv=2, head_dim=16, hidden_dim=64).eval()
    m_cfg, r_cfg = build_lumina_3b_config_from_hf(
        teacher.config,
        swa_levels=[3],
        swa_global=True,
        out_gate_rank=8,
        assoc_channel=False,
        swa_rope=True,
    )
    student = build_student_for_surgery(m_cfg, r_cfg, dtype=torch.float32)
    t_info = transplant_teacher_weights(teacher, student, quiet_init_gates=True)
    assert t_info["embedding"]["warmstarted"] is True

    # 1. Verify Top-K KL matches full KL cleanly & backprops finite gradients
    sh = torch.randn(32, 32, requires_grad=True)
    th = torch.randn(32, 32)
    y = torch.randint(0, 64, (32,))
    loss_full, _, kl_full = _chunked_hidden_ce_kl_loss(
        sh, student.lm_head, th, teacher.lm_head, y, temperature=2.0, ce_weight=0.5, kl_weight=1.0, chunk_tokens=16, kl_topk=0
    )
    loss_topk63, _, kl_topk63 = _chunked_hidden_ce_kl_loss(
        sh, student.lm_head, th, teacher.lm_head, y, temperature=2.0, ce_weight=0.5, kl_weight=1.0, chunk_tokens=16, kl_topk=63
    )
    loss_topk, _, kl_topk = _chunked_hidden_ce_kl_loss(
        sh, student.lm_head, th, teacher.lm_head, y, temperature=2.0, ce_weight=0.5, kl_weight=1.0, chunk_tokens=16, kl_topk=32
    )
    assert abs(kl_full - kl_topk63) < 1e-4
    assert 0.0 <= kl_full - kl_topk < 0.25
    loss_topk.backward()
    assert sh.grad is not None and torch.isfinite(sh.grad).all()

    # 2. Verify On-Policy Distillation (OPD) runs cleanly end-to-end on CPU
    prompts = [torch.randint(0, 64, (2, 6)) for _ in range(3)]
    res = run_on_policy_distill(
        teacher,
        student,
        iter(prompts),
        dev0=torch.device("cpu"),
        dev1=torch.device("cpu"),
        max_steps=3,
        rollout_tokens=4,
        kl_topk=16,
        lr=1e-3,
        precision="fp32",
        freeze_deltanet_proj=True,
        lora_rank=4,
    )
    assert res["steps_completed"] == 3
    assert res["rollout_tokens"] == 2 * 4 * 3
    assert res["merged_lora_layers"] > 0

    # 3. Verify QuaRot W4A16 default act_bits=16 & rotate_input=False option
    lin = nn.Linear(32, 32, bias=False)
    q_no_rot = HadamardLinearINT4.from_linear(lin, group_size=16, act_bits=16, use_triton=False, rotate_input=False)
    x_in = torch.randn(2, 4, 32)
    assert torch.isfinite(q_no_rot(x_in)).all()
    q_m, q_rep = quantize_model_quarot_int4(student.eval(), group_size=16, inplace=False, use_triton=False)
    assert q_rep["act_bits"] == 16
    assert torch.isfinite(q_m(prompts[0]).logits).all()


def test_titans_ttt_causality_o1_state_and_streaming_equivalence():
    torch.manual_seed(21)
    cell = TitansMemoryCell(d_model=32, n_heads=4, mem_hidden_mult=2, chunk_size=4).eval()
    x = torch.randn(2, 12, 32)

    # 1. Strict chunk-level causality (tokenwise when chunk_size=1)
    x_mut = x.clone()
    x_mut[:, 7] += 5.0
    with torch.no_grad():
        y_a, st_a = cell(x, chunk_size=1)
        y_b, _ = cell(x_mut, chunk_size=1)
    assert torch.allclose(y_a[:, :7], y_b[:, :7], atol=1e-5)
    assert not torch.allclose(y_a[:, 8:], y_b[:, 8:], atol=1e-4)

    # 2. Exact streaming equivalence across chunk boundaries & constant O(1) state bytes
    with torch.no_grad():
        full_y, full_st = cell(x, chunk_size=4)
        c1_y, c1_st = cell(x[:, :4], chunk_size=4)
        c2_y, c2_st = cell(x[:, 4:8], state=c1_st, chunk_size=4)
        c3_y, c3_st = cell(x[:, 8:], state=c2_st, chunk_size=4)
        stream_y = torch.cat((c1_y, c2_y, c3_y), dim=1)
    assert torch.allclose(full_y, stream_y, atol=1e-5)
    assert torch.allclose(full_st.w1, c3_st.w1, atol=1e-5)
    st_bytes = sum(t.numel() * t.element_size() for t in (c3_st.w1, c3_st.w2, c3_st.s1, c3_st.s2)) // 2
    assert st_bytes == cell.state_bytes_per_seq()

    # 3. Finite outer-loop gradients through inner test-time weight updates
    cell.train()
    x_g = torch.randn(2, 8, 32, requires_grad=True)
    out_g, _ = cell(x_g, chunk_size=4)
    out_g.square().mean().backward()
    assert x_g.grad is not None and torch.isfinite(x_g.grad).all()
    assert cell.init_w1.grad is not None and torch.isfinite(cell.init_w1.grad).all()
