"""Fused Triton inter-chunk recurrence scan for chunked DPLR memory.

Replaces the unrolled Python `for m in range(n_chunks):` loop in `dplr_chunked`
with a fused single-kernel scan keeping the recurrent state matrix `S` in SRAM
/ registers with strict IEEE FP32 accumulation (`input_precision="ieee"`).
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:
    triton = None  # type: ignore
    tl = None  # type: ignore
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _dplr_inter_chunk_fwd_kernel(
        kv_ptr,
        p_ptr,
        t_ptr,
        m_ptr,
        init_ptr,
        out_ptr,
        final_ptr,
        ckpt_ptr,
        N: tl.constexpr,
        C: tl.constexpr,
        V: tl.constexpr,
        K: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_K: tl.constexpr,
        stride_kv_bh,
        stride_kv_n,
        stride_kv_c,
        stride_kv_v,
        stride_p_bh,
        stride_p_n,
        stride_p_c,
        stride_p_k,
        stride_t_bh,
        stride_t_n,
        stride_t_k0,
        stride_t_k1,
        stride_m_bh,
        stride_m_n,
        stride_m_v,
        stride_m_k,
        stride_s_bh,
        stride_s_v,
        stride_s_k,
        stride_o_bh,
        stride_o_n,
        stride_o_c,
        stride_o_v,
        stride_ck_bh,
        stride_ck_n,
        stride_ck_v,
        stride_ck_k,
    ):
        bh = tl.program_id(0).to(tl.int64)
        offs_c = tl.arange(0, BLOCK_C)
        offs_v = tl.arange(0, BLOCK_V)
        offs_k = tl.arange(0, BLOCK_K)
        mask_c = offs_c < C
        mask_v = offs_v < V
        mask_k = offs_k < K
        mask_vk = mask_v[:, None] & mask_k[None, :]
        mask_cv = mask_c[:, None] & mask_v[None, :]
        mask_ck = mask_c[:, None] & mask_k[None, :]
        mask_kk = mask_k[:, None] & mask_k[None, :]

        s = tl.load(
            init_ptr + bh * stride_s_bh + offs_v[:, None] * stride_s_v + offs_k[None, :] * stride_s_k,
            mask=mask_vk,
            other=0.0,
        ).to(tl.float32)

        for n in range(N):
            tl.store(
                ckpt_ptr + bh * stride_ck_bh + n * stride_ck_n + offs_v[:, None] * stride_ck_v + offs_k[None, :] * stride_ck_k,
                s,
                mask=mask_vk,
            )
            kv = tl.load(
                kv_ptr + bh * stride_kv_bh + n * stride_kv_n + offs_c[:, None] * stride_kv_c + offs_v[None, :] * stride_kv_v,
                mask=mask_cv,
                other=0.0,
            ).to(tl.float32)
            p = tl.load(
                p_ptr + bh * stride_p_bh + n * stride_p_n + offs_c[:, None] * stride_p_c + offs_k[None, :] * stride_p_k,
                mask=mask_ck,
                other=0.0,
            ).to(tl.float32)
            out_n = kv + tl.dot(p, tl.trans(s), input_precision="ieee")
            tl.store(
                out_ptr + bh * stride_o_bh + n * stride_o_n + offs_c[:, None] * stride_o_c + offs_v[None, :] * stride_o_v,
                out_n,
                mask=mask_cv,
            )
            t_mat = tl.load(
                t_ptr + bh * stride_t_bh + n * stride_t_n + offs_k[:, None] * stride_t_k0 + offs_k[None, :] * stride_t_k1,
                mask=mask_kk,
                other=0.0,
            ).to(tl.float32)
            m_mat = tl.load(
                m_ptr + bh * stride_m_bh + n * stride_m_n + offs_v[:, None] * stride_m_v + offs_k[None, :] * stride_m_k,
                mask=mask_vk,
                other=0.0,
            ).to(tl.float32)
            s = tl.where(mask_vk, tl.dot(s, t_mat, input_precision="ieee") + m_mat, 0.0)

        tl.store(
            final_ptr + bh * stride_s_bh + offs_v[:, None] * stride_s_v + offs_k[None, :] * stride_s_k,
            s,
            mask=mask_vk,
        )

    @triton.jit
    def _dplr_inter_chunk_bwd_kernel(
        do_ptr,
        df_ptr,
        p_ptr,
        t_ptr,
        ckpt_ptr,
        dp_ptr,
        dt_ptr,
        dm_ptr,
        dinit_ptr,
        N: tl.constexpr,
        C: tl.constexpr,
        V: tl.constexpr,
        K: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_K: tl.constexpr,
        stride_o_bh,
        stride_o_n,
        stride_o_c,
        stride_o_v,
        stride_s_bh,
        stride_s_v,
        stride_s_k,
        stride_p_bh,
        stride_p_n,
        stride_p_c,
        stride_p_k,
        stride_t_bh,
        stride_t_n,
        stride_t_k0,
        stride_t_k1,
        stride_ck_bh,
        stride_ck_n,
        stride_ck_v,
        stride_ck_k,
        stride_m_bh,
        stride_m_n,
        stride_m_v,
        stride_m_k,
    ):
        bh = tl.program_id(0).to(tl.int64)
        offs_c = tl.arange(0, BLOCK_C)
        offs_v = tl.arange(0, BLOCK_V)
        offs_k = tl.arange(0, BLOCK_K)
        mask_c = offs_c < C
        mask_v = offs_v < V
        mask_k = offs_k < K
        mask_vk = mask_v[:, None] & mask_k[None, :]
        mask_cv = mask_c[:, None] & mask_v[None, :]
        mask_ck = mask_c[:, None] & mask_k[None, :]
        mask_kk = mask_k[:, None] & mask_k[None, :]

        ds = tl.load(
            df_ptr + bh * stride_s_bh + offs_v[:, None] * stride_s_v + offs_k[None, :] * stride_s_k,
            mask=mask_vk,
            other=0.0,
        ).to(tl.float32)

        for step in range(N):
            n = N - 1 - step
            s_n = tl.load(
                ckpt_ptr + bh * stride_ck_bh + n * stride_ck_n + offs_v[:, None] * stride_ck_v + offs_k[None, :] * stride_ck_k,
                mask=mask_vk,
                other=0.0,
            ).to(tl.float32)
            do_n = tl.load(
                do_ptr + bh * stride_o_bh + n * stride_o_n + offs_c[:, None] * stride_o_c + offs_v[None, :] * stride_o_v,
                mask=mask_cv,
                other=0.0,
            ).to(tl.float32)
            p_n = tl.load(
                p_ptr + bh * stride_p_bh + n * stride_p_n + offs_c[:, None] * stride_p_c + offs_k[None, :] * stride_p_k,
                mask=mask_ck,
                other=0.0,
            ).to(tl.float32)
            t_n = tl.load(
                t_ptr + bh * stride_t_bh + n * stride_t_n + offs_k[:, None] * stride_t_k0 + offs_k[None, :] * stride_t_k1,
                mask=mask_kk,
                other=0.0,
            ).to(tl.float32)

            tl.store(
                dm_ptr + bh * stride_m_bh + n * stride_m_n + offs_v[:, None] * stride_m_v + offs_k[None, :] * stride_m_k,
                ds,
                mask=mask_vk,
            )
            dt_n = tl.dot(tl.trans(s_n), ds, input_precision="ieee")
            tl.store(
                dt_ptr + bh * stride_t_bh + n * stride_t_n + offs_k[:, None] * stride_t_k0 + offs_k[None, :] * stride_t_k1,
                dt_n,
                mask=mask_kk,
            )
            dp_n = tl.dot(do_n, s_n, input_precision="ieee")
            tl.store(
                dp_ptr + bh * stride_p_bh + n * stride_p_n + offs_c[:, None] * stride_p_c + offs_k[None, :] * stride_p_k,
                dp_n,
                mask=mask_ck,
            )
            ds = tl.where(
                mask_vk,
                tl.dot(ds, tl.trans(t_n), input_precision="ieee") + tl.dot(tl.trans(do_n), p_n, input_precision="ieee"),
                0.0,
            )

        tl.store(
            dinit_ptr + bh * stride_s_bh + offs_v[:, None] * stride_s_v + offs_k[None, :] * stride_s_k,
            ds,
            mask=mask_vk,
        )


def _next_pow2_16(x: int) -> int:
    return 16 if x <= 16 else 1 << (x - 1).bit_length()


def can_use_triton_dplr(chunk_size: int, v_dim: int, k_dim: int, device: torch.device) -> bool:
    return bool(HAS_TRITON and device.type == "cuda" and max(chunk_size, v_dim, k_dim) <= 128)


def _inter_chunk_scan_pytorch_fwd(
    kv_local: torch.Tensor,
    read_proj: torch.Tensor,
    trans_mat: torch.Tensor,
    mem_local: torch.Tensor,
    init_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bh, n_chunks, c_len, v_dim = kv_local.shape
    _, _, _, k_dim = read_proj.shape
    out = torch.empty((bh, n_chunks, c_len, v_dim), device=kv_local.device, dtype=torch.float32)
    checkpoints = torch.empty((bh, n_chunks, v_dim, k_dim), device=kv_local.device, dtype=torch.float32)
    s = init_state.float()
    for n in range(n_chunks):
        checkpoints[:, n] = s
        out[:, n] = kv_local[:, n] + read_proj[:, n] @ s.transpose(-1, -2)
        s = s @ trans_mat[:, n] + mem_local[:, n]
    return out, s.contiguous(), checkpoints


def _inter_chunk_scan_pytorch_bwd(
    grad_out: torch.Tensor,
    grad_final: torch.Tensor,
    read_proj: torch.Tensor,
    trans_mat: torch.Tensor,
    checkpoints: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bh, n_chunks, _, _ = read_proj.shape
    d_read_proj = torch.empty_like(read_proj)
    d_trans_mat = torch.empty_like(trans_mat)
    d_mem_local = torch.empty_like(checkpoints)
    ds = grad_final.float()
    for n in range(n_chunks - 1, -1, -1):
        s_n = checkpoints[:, n]
        do_n = grad_out[:, n]
        d_mem_local[:, n] = ds
        d_trans_mat[:, n] = s_n.transpose(-1, -2) @ ds
        d_read_proj[:, n] = do_n @ s_n
        ds = ds @ trans_mat[:, n].transpose(-1, -2) + do_n.transpose(-1, -2) @ read_proj[:, n]
    return d_read_proj, d_trans_mat, d_mem_local, ds.contiguous()


@torch.library.custom_op("llm2::dplr_inter_chunk_scan_fwd", mutates_args=())
def _dplr_inter_chunk_scan_fwd(
    kv_local: torch.Tensor,
    read_proj: torch.Tensor,
    trans_mat: torch.Tensor,
    mem_local: torch.Tensor,
    init_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kv = kv_local.float().contiguous()
    p = read_proj.float().contiguous()
    t_mat = trans_mat.float().contiguous()
    m_mat = mem_local.float().contiguous()
    s0 = init_state.float().contiguous()
    bh, n_chunks, c_len, v_dim = kv.shape
    k_dim = p.shape[-1]
    if not can_use_triton_dplr(c_len, v_dim, k_dim, kv.device):
        return _inter_chunk_scan_pytorch_fwd(kv, p, t_mat, m_mat, s0)

    out = torch.empty_like(kv)
    final_state = torch.empty_like(s0)
    checkpoints = torch.empty((bh, n_chunks, v_dim, k_dim), device=kv.device, dtype=torch.float32)
    bc, bv, bk = _next_pow2_16(c_len), _next_pow2_16(v_dim), _next_pow2_16(k_dim)
    stages = 1 if max(bc, bv, bk) >= 128 else 2
    _dplr_inter_chunk_fwd_kernel[(bh,)](
        kv, p, t_mat, m_mat, s0, out, final_state, checkpoints,
        n_chunks, c_len, v_dim, k_dim, bc, bv, bk,
        kv.stride(0), kv.stride(1), kv.stride(2), kv.stride(3),
        p.stride(0), p.stride(1), p.stride(2), p.stride(3),
        t_mat.stride(0), t_mat.stride(1), t_mat.stride(2), t_mat.stride(3),
        m_mat.stride(0), m_mat.stride(1), m_mat.stride(2), m_mat.stride(3),
        s0.stride(0), s0.stride(1), s0.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        checkpoints.stride(0), checkpoints.stride(1), checkpoints.stride(2), checkpoints.stride(3),
        num_warps=4,
        num_stages=stages,
    )
    return out, final_state, checkpoints


@_dplr_inter_chunk_scan_fwd.register_fake
def _fwd_fake(
    kv_local: torch.Tensor,
    read_proj: torch.Tensor,
    trans_mat: torch.Tensor,
    mem_local: torch.Tensor,
    init_state: torch.Tensor,
):
    return (
        torch.empty_like(kv_local),
        torch.empty_like(init_state),
        torch.empty_like(mem_local),
    )


@torch.library.custom_op("llm2::dplr_inter_chunk_scan_bwd", mutates_args=())
def _dplr_inter_chunk_scan_bwd(
    grad_out: torch.Tensor,
    grad_final: torch.Tensor,
    read_proj: torch.Tensor,
    trans_mat: torch.Tensor,
    checkpoints: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    do = grad_out.float().contiguous()
    df = grad_final.float().contiguous()
    p = read_proj.float().contiguous()
    t_mat = trans_mat.float().contiguous()
    ckpts = checkpoints.float().contiguous()
    bh, n_chunks, c_len, k_dim = p.shape
    v_dim = ckpts.shape[-2]
    if not can_use_triton_dplr(c_len, v_dim, k_dim, do.device):
        return _inter_chunk_scan_pytorch_bwd(do, df, p, t_mat, ckpts)

    dp = torch.empty_like(p)
    dt = torch.empty_like(t_mat)
    dm = torch.empty_like(ckpts)
    dinit = torch.empty_like(df)
    bc, bv, bk = _next_pow2_16(c_len), _next_pow2_16(v_dim), _next_pow2_16(k_dim)
    stages = 1 if max(bc, bv, bk) >= 128 else 2
    _dplr_inter_chunk_bwd_kernel[(bh,)](
        do, df, p, t_mat, ckpts, dp, dt, dm, dinit,
        n_chunks, c_len, v_dim, k_dim, bc, bv, bk,
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        df.stride(0), df.stride(1), df.stride(2),
        p.stride(0), p.stride(1), p.stride(2), p.stride(3),
        t_mat.stride(0), t_mat.stride(1), t_mat.stride(2), t_mat.stride(3),
        ckpts.stride(0), ckpts.stride(1), ckpts.stride(2), ckpts.stride(3),
        dm.stride(0), dm.stride(1), dm.stride(2), dm.stride(3),
        num_warps=4,
        num_stages=stages,
    )
    return dp, dt, dm, dinit


@_dplr_inter_chunk_scan_bwd.register_fake
def _bwd_fake(
    grad_out: torch.Tensor,
    grad_final: torch.Tensor,
    read_proj: torch.Tensor,
    trans_mat: torch.Tensor,
    checkpoints: torch.Tensor,
):
    return (
        torch.empty_like(read_proj),
        torch.empty_like(trans_mat),
        torch.empty_like(checkpoints),
        torch.empty_like(grad_final),
    )


def _setup_context(ctx, inputs, output):
    _, read_proj, trans_mat, _, _ = inputs
    _, _, checkpoints = output
    ctx.save_for_backward(read_proj, trans_mat, checkpoints)


def _backward(ctx, grad_out, grad_final, grad_ckpt):
    read_proj, trans_mat, checkpoints = ctx.saved_tensors
    if grad_out is None:
        grad_out = torch.zeros(
            (read_proj.shape[0], read_proj.shape[1], read_proj.shape[2], checkpoints.shape[-2]),
            device=read_proj.device,
            dtype=torch.float32,
        )
    if grad_final is None:
        grad_final = torch.zeros_like(checkpoints[:, 0])
    dp, dt, dm, dinit = _dplr_inter_chunk_scan_bwd(grad_out, grad_final, read_proj, trans_mat, checkpoints)
    return grad_out, dp, dt, dm, dinit


_dplr_inter_chunk_scan_fwd.register_autograd(_backward, setup_context=_setup_context)


def dplr_inter_chunk_scan(
    kv_local: torch.Tensor,
    read_proj: torch.Tensor,
    mem_decay: torch.Tensor,
    state_trans: torch.Tensor,
    mem_local: torch.Tensor,
    init_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fused inter-chunk DPLR scan across (B, H, N, ...) tensors."""
    b, h, n_chunks, c_len, v_dim = kv_local.shape
    k_dim = read_proj.shape[-1]
    bh = b * h
    decay_vec = mem_decay.squeeze(-2).expand(b, h, n_chunks, k_dim)
    trans_mat = state_trans + torch.diag_embed(decay_vec)
    out_bh, final_bh, _ = _dplr_inter_chunk_scan_fwd(
        kv_local.reshape(bh, n_chunks, c_len, v_dim),
        read_proj.reshape(bh, n_chunks, c_len, k_dim),
        trans_mat.reshape(bh, n_chunks, k_dim, k_dim),
        mem_local.reshape(bh, n_chunks, v_dim, k_dim),
        init_state.reshape(bh, v_dim, k_dim),
    )
    return out_bh.reshape(b, h, n_chunks * c_len, v_dim), final_bh.reshape(b, h, v_dim, k_dim)
