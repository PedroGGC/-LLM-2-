from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from llm2.architectures.fractal_delta import (
    FractalDecode,
    FractalDeltaCell,
    FractalDeltaMemory,
    SlidingWindowGQACell,
    SWAState,
    _apply_split_half_rope,
)
from llm2.architectures.quarot import quantize_model_quarot_int4
from llm2.config import ModelConfig, ResearchConfig
from llm2.model import RMSNorm, SwiGLU
from llm2.research.decode_optimization import SegmentMemoryCache
from llm2.research.distill_convert import LoRALinear, apply_surgery_freeze, transplant_teacher_weights
from llm2.research.lumina_3b_pipeline import (
    _chunked_hidden_ce_kl_loss,
    build_lumina_3b_config_from_hf,
    build_student_for_surgery,
    load_lumina_checkpoint,
    run_subfase1_dual_gpu,
    run_subfase2_dual_gpu,
    save_lumina_checkpoint,
)


def _unique_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in {p.data_ptr(): p for p in model.parameters()}.values())


def _measure_state_nbytes(state) -> int:
    total = 0
    for lvl in state.levels:
        if isinstance(lvl, SWAState):
            for t in (lvl.k_sinks, lvl.v_sinks, lvl.pos_sinks, lvl.k_tail, lvl.v_tail, lvl.pos_tail, lvl.seen):
                total += t.numel() * t.element_size()
        else:
            total += lvl.memory.numel() * lvl.memory.element_size() + lvl.tail.numel() * lvl.tail.element_size()
    total += state.assoc_memory.numel() * state.assoc_memory.element_size()
    total += state.local_tail.numel() * state.local_tail.element_size()
    if state.phase_pos is not None:
        total += state.phase_pos.numel() * state.phase_pos.element_size()
    return total


class _MockHFGQAAttn(nn.Module):
    def __init__(self, d: int, n_heads: int, n_kv: int, rope_theta: float, use_rope: bool):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv = n_kv
        self.hd = d // n_heads
        self.groups = n_heads // n_kv
        self.use_rope = use_rope
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, n_kv * self.hd, bias=False)
        self.v_proj = nn.Linear(d, n_kv * self.hd, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, self.hd, 2, dtype=torch.float64) / float(self.hd)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv, self.hd).transpose(1, 2)
        if self.use_rope:
            pos = torch.arange(t, device=x.device, dtype=torch.float64)[None, :]
            angles = (pos[:, None, :, None] * self.inv_freq.view(1, 1, 1, -1)).float()
            q = _apply_split_half_rope(q.float(), angles).to(x.dtype)
            k = _apply_split_half_rope(k.float(), angles).to(x.dtype)
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(attn.transpose(1, 2).contiguous().view(b, t, d))


class _MockHFSmolLM3Layer(nn.Module):
    def __init__(self, d: int, n_heads: int, n_kv: int, ff: int, rope_theta: float, use_rope: bool):
        super().__init__()
        self.input_layernorm = RMSNorm(d, eps=1e-6)
        self.self_attn = _MockHFGQAAttn(d, n_heads, n_kv, rope_theta, use_rope)
        self.post_attention_layernorm = RMSNorm(d, eps=1e-6)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(d, ff, bias=False)
        self.mlp.up_proj = nn.Linear(d, ff, bias=False)
        self.mlp.down_proj = nn.Linear(ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        h = self.post_attention_layernorm(x)
        x = x + self.mlp.down_proj(F.silu(self.mlp.gate_proj(h)) * self.mlp.up_proj(h))
        return x


class _MockHFSmolLM3Inner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        self.layers = nn.ModuleList(
            [
                _MockHFSmolLM3Layer(
                    cfg.hidden_size,
                    cfg.num_attention_heads,
                    cfg.num_key_value_heads,
                    cfg.intermediate_size,
                    cfg.rope_theta,
                    use_rope=((i + 1) % cfg.no_rope_layer_interval != 0),
                )
                for i in range(cfg.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = True, use_cache: bool = False):
        x = self.embed_tokens(input_ids)
        hs = [x]
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            hs.append(self.norm(x) if idx == len(self.layers) - 1 else x)
        return SimpleNamespace(last_hidden_state=hs[-1], hidden_states=tuple(hs))


class _MockHFSmolLM3ForCausalLM(nn.Module):
    def __init__(self, *, vocab=64, d=64, n_layers=8, n_heads=8, n_kv=2, ff=128):
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=vocab,
            hidden_size=d,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv,
            intermediate_size=ff,
            rope_theta=5000000.0,
            rms_norm_eps=1e-6,
            no_rope_layer_interval=4,
        )
        self.model = _MockHFSmolLM3Inner(self.config)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = True, use_cache: bool = False):
        out = self.model(input_ids, output_hidden_states=output_hidden_states, use_cache=use_cache)
        return SimpleNamespace(logits=self.lm_head(out.last_hidden_state), hidden_states=out.hidden_states)


def test_true_3to1_hybrid_parameter_formula_and_o1_state_bytes():
    torch.manual_seed(7)
    cfg = ModelConfig(vocab_size=64, d_model=64, n_layers=8, n_heads=8, hidden_dim=128, max_seq_len=64, dropout=0.0)
    res = ResearchConfig(
        architecture="fractal_delta",
        scale_conditioning=False,
        share_scales=False,
        fractal_window=12,
        fractal_hybrid_stride=4,
        fractal_hybrid_swa=True,
        fractal_swa_kv_heads=2,
        fractal_swa_sinks=4,
        fractal_swa_rope=False,
        fractal_deltanet_rope=True,
        fractal_decoupled_gates=True,
        fractal_assoc_delta=True,
    )
    model = FractalDeltaMemory(cfg, res).eval()
    assert model.swa_levels == frozenset({3, 7})
    assert isinstance(model.cells[0], FractalDeltaCell)
    assert isinstance(model.cells[3], SlidingWindowGQACell)
    assert isinstance(model.cells[7], SlidingWindowGQACell)
    assert model.parameter_formula() == _unique_param_count(model)

    expected_bytes = model.state_bytes_per_sequence()
    for seq_len in (6, 20, 48):
        ids = torch.randint(0, 64, (1, seq_len))
        out = model(ids)
        assert _measure_state_nbytes(out.state) == expected_bytes


@pytest.mark.parametrize("swa_rope", [False, True])
def test_swa_gqa_causality_and_exact_streaming_decode_equivalence(swa_rope: bool):
    torch.manual_seed(11)
    cfg = ModelConfig(vocab_size=48, d_model=32, n_layers=4, n_heads=4, hidden_dim=64, max_seq_len=64, dropout=0.0)
    res = ResearchConfig(
        architecture="fractal_delta",
        scale_conditioning=False,
        share_scales=False,
        fractal_window=8,
        fractal_hybrid_stride=4,
        fractal_hybrid_swa=True,
        fractal_swa_kv_heads=2,
        fractal_swa_sinks=4,
        fractal_swa_rope=swa_rope,
        fractal_deltanet_rope=True,
        fractal_decoupled_gates=True,
    )
    model = FractalDeltaMemory(cfg, res).eval()
    ids = torch.randint(0, 48, (2, 28))  # 28 > sinks(4) + window(8), exercises eviction + sink pinning!

    # 1. Strict causality
    ids_mut = ids.clone()
    ids_mut[:, 15] = (ids_mut[:, 15] + 1) % 48
    with torch.no_grad():
        full_a = model(ids).logits
        full_b = model(ids_mut).logits
    assert torch.allclose(full_a[:, :15], full_b[:, :15], atol=1e-5)

    # 2. Exact equivalence: full forward vs 3-chunk streaming vs prefill + 1-token incremental decode
    with torch.no_grad():
        c1 = model(ids[:, :3])  # < sinks (tests partial sink fill!)
        c2 = model(ids[:, 3:14], state=c1.state)  # spans sink boundary into window!
        c3 = model(ids[:, 14:], state=c2.state)  # exceeds window (tests eviction!)
        chunked_logits = torch.cat((c1.logits, c2.logits, c3.logits), dim=1)
    assert torch.allclose(full_a, chunked_logits, atol=1e-4), f"chunked max diff: {(full_a - chunked_logits).abs().max().item()}"

    dec = FractalDecode(model)
    with torch.no_grad():
        pref_logits, st = dec.prefill(ids[:, :2], capacity=32)  # start inside sinks (t=2 < S=4)
        step_outs = [pref_logits]
        for t in range(2, 28):
            lg, st = dec.step(ids[:, t : t + 1], st)
            step_outs.append(lg)
        inc_logits = torch.cat(step_outs, dim=1)
    assert torch.allclose(full_a[:, 1:], inc_logits, atol=1e-4), f"decode max diff: {(full_a[:, 1:] - inc_logits).abs().max().item()}"

    # 3. Verify return_all_states + slice_step_state + GRM state cloning on SWA levels
    with torch.no_grad():
        _, _, st_base = dec.prefill_hidden(ids[:, :6], capacity=32)
        lg_blk, _, st_all = dec.step_hidden(ids[:, 6:10], st_base, return_all_states=True)
        st_idx2 = FractalDecode.slice_step_state(st_all, 2)
        lg_after, _ = dec.step(ids[:, 9:10], st_idx2)
    assert torch.allclose(lg_blk[:, 3:4], lg_after, atol=1e-4)

    cache = SegmentMemoryCache()
    cache.record_segment(st_base, torch.randn(2, 6, 32), tag="s0")
    cache.record_anchor(st_base)
    fused, _ = cache.retrieve_grm_state(torch.randn(2, 2, 32), active_state=st, top_k=1)
    blended = cache.blend_with_anchors(st, anchor_weight=0.2)
    with torch.no_grad():
        assert torch.isfinite(dec.step(ids[:, 0:1], fused)[0]).all()
        assert torch.isfinite(dec.step(ids[:, 0:1], blended)[0]).all()


def test_chunked_hidden_ce_kl_matches_unchunked_exactly():
    torch.manual_seed(3)
    n_tok, d, v = 19, 32, 48
    s_head = nn.Linear(d, v, bias=False)
    t_head = nn.Linear(d, v, bias=False)
    h_s1 = torch.randn(n_tok, d, requires_grad=True)
    h_s2 = h_s1.detach().clone().requires_grad_(True)
    h_t = torch.randn(n_tok, d)
    targets = torch.randint(0, v, (n_tok,))
    targets[3] = -100
    targets[11] = -100

    # Reference unchunked FP32
    s_log = s_head(h_s1).float()
    t_log = t_head(h_t).detach().float()
    vm = targets != -100
    ref_ce = F.cross_entropy(s_log, targets, ignore_index=-100)
    ref_kl = F.kl_div(
        F.log_softmax(s_log[vm] / 2.0, dim=-1),
        F.log_softmax(t_log[vm] / 2.0, dim=-1),
        reduction="batchmean",
        log_target=True,
    ) * 4.0
    ref_total = 0.5 * ref_ce + 1.0 * ref_kl
    ref_total.backward()

    chk_total, chk_ce, chk_kl = _chunked_hidden_ce_kl_loss(
        h_s2, s_head, h_t, t_head, targets, temperature=2.0, ce_weight=0.5, kl_weight=1.0, chunk_tokens=5
    )
    chk_total.backward()

    assert abs(chk_ce - float(ref_ce.item())) < 1e-5
    assert abs(chk_kl - float(ref_kl.item())) < 1e-5
    assert torch.allclose(h_s1.grad, h_s2.grad, atol=1e-5)


def test_smollm3_style_gqa_3to1_surgery_subfase1_2_lora_merge_and_ckpt(tmp_path):
    torch.manual_seed(19)
    teacher = _MockHFSmolLM3ForCausalLM(vocab=64, d=64, n_layers=8, n_heads=8, n_kv=2, ff=128).eval()
    m_cfg, r_cfg = build_lumina_3b_config_from_hf(
        teacher.config, window=16, sinks=4, hybrid_stride=4, max_seq_len=32, gradient_checkpointing=True
    )
    assert r_cfg.fractal_hybrid_swa is True
    assert r_cfg.fractal_swa_rope is False  # Auto-detected NoPE on every 4th layer!
    assert r_cfg.fractal_deltanet_rope is True
    assert r_cfg.fractal_swa_kv_heads == 2

    student = build_student_for_surgery(m_cfg, r_cfg, dtype=torch.float32)
    assert student.parameter_formula() == _unique_param_count(student)
    t_info = transplant_teacher_weights(teacher, student)
    assert t_info["swa_cells_copied"] == 2
    assert t_info["qkv_cells_copied"] == 6
    assert t_info["ffn_cells_copied"] == 8

    # Verify Step-0 EXACT identity on transplanted NoPE SWA layers (indices 3 and 7)
    x_probe = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        t_out = teacher.model(x_probe, output_hidden_states=True)
        for swa_lvl in (3, 7):
            h_in = t_out.hidden_states[swa_lvl]
            t_delta = teacher.model.layers[swa_lvl].self_attn(teacher.model.layers[swa_lvl].input_layernorm(h_in))
            s_delta = student.forward_level_attn_delta(swa_lvl, h_in, t_out.hidden_states[0])
            cos = float(F.cosine_similarity(s_delta.float(), t_delta.float(), dim=-1).mean().item())
            assert cos > 0.9999, f"SWA level {swa_lvl} not exact at Step 0: cos={cos}"

    fixed_seq = torch.randint(0, 64, (2, 17), generator=torch.Generator().manual_seed(123))

    def _batch_gen():
        while True:
            yield fixed_seq[:, :-1], fixed_seq[:, 1:]

    cpu = torch.device("cpu")
    s1 = run_subfase1_dual_gpu(
        teacher, student, _batch_gen(), dev0=cpu, dev1=cpu, max_steps=10, lr=3e-3, warmup_steps=2, precision="fp32", log_interval=0
    )
    assert s1["final_metrics"]["attn_delta_loss"] < s1["initial_metrics"]["attn_delta_loss"]
    assert s1["final_metrics"]["mean_attn_cos"] > s1["initial_metrics"]["mean_attn_cos"]

    s2a = run_subfase2_dual_gpu(
        teacher,
        student,
        _batch_gen(),
        dev0=cpu,
        dev1=cpu,
        split_layer=0,
        max_steps=8,
        lr=1e-3,
        warmup_steps=2,
        logit_chunk_tokens=8,
        precision="fp32",
        lora_rank=0,
        compact_optimizer=True,
        log_interval=0,
    )
    assert s2a["final_metrics"]["total_loss"] < s2a["initial_metrics"]["total_loss"]

    ckpt_dir = tmp_path / "lumina_test_ckpt"
    assert (
        apply_surgery_freeze(student, freeze_deltanet_proj=True)["trainable_params"]
        < apply_surgery_freeze(student, freeze_deltanet_proj=False)["trainable_params"]
    )
    s2b = run_subfase2_dual_gpu(
        teacher,
        student,
        _batch_gen(),
        dev0=cpu,
        dev1=cpu,
        split_layer=0,
        max_steps=6,
        lr=5e-4,
        warmup_steps=1,
        logit_chunk_tokens=8,
        precision="fp32",
        lora_rank=4,
        freeze_deltanet_proj=True,
        compact_optimizer=True,
        log_interval=0,
        ckpt_dir=ckpt_dir,
    )
    assert s2b["final_metrics"]["total_loss"] < s2b["initial_metrics"]["total_loss"]
    assert s2b["merged_lora_layers"] > 0
    assert not any(isinstance(m, LoRALinear) for m in student.modules())
    assert student.parameter_formula() == _unique_param_count(student)

    reloaded, meta = load_lumina_checkpoint(ckpt_dir, device="cpu")
    assert meta["stage"] == "subfase2b_merged"
    assert reloaded.lm_head.weight is reloaded.embedding.weight
    student.eval()
    reloaded.eval()
    with torch.no_grad():
        assert torch.allclose(student(x_probe).logits, reloaded(x_probe).logits, atol=1e-5)

    # Verify QuaRot W4A16 quantization on True 3:1 Hybrid
    q_model, q_rep = quantize_model_quarot_int4(reloaded, group_size=16, inplace=False, use_triton=False)
    assert q_rep["shared_embedding_preserved"] is True
    assert any("cells.3.q_proj" in name for name in q_rep["replaced_layers"])
    with torch.no_grad():
        q_out = q_model(x_probe)
        assert torch.isfinite(q_out.logits).all()
        q_dec = FractalDecode(q_model.eval())
        q_lg, q_st = q_dec.prefill(x_probe[:, :8], capacity=32)
        q_lg2, _ = q_dec.step(x_probe[:, 8:9], q_st)
        assert torch.isfinite(q_lg).all() and torch.isfinite(q_lg2).all()
