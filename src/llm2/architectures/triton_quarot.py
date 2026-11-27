"""Fused Triton INT4 unpack + matvec/matmul kernels for QuaRot HadamardLinearINT4."""

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
    def _quarot_int4_matvec_grouped_kernel(
        x_ptr,
        w_ptr,
        s_ptr,
        bias_ptr,
        out_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        G_BYTES: tl.constexpr,
        GROUPS_PER_TILE: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_KB: tl.constexpr,
    ):
        """Bandwidth-optimized INT4 matvec: 1 scale load per group, single cross-warp reduction, L2 pinning for x."""
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        k_bytes = K // 2
        n_groups = k_bytes // G_BYTES

        acc_g = tl.zeros((BLOCK_N, GROUPS_PER_TILE), dtype=tl.float32)
        for g0 in range(0, n_groups, GROUPS_PER_TILE):
            offs_g = g0 + tl.arange(0, GROUPS_PER_TILE)
            mask_g = offs_g < n_groups
            s = tl.load(
                s_ptr + offs_n[:, None] * n_groups + offs_g[None, :],
                mask=mask_n[:, None] & mask_g[None, :],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

            offs_kb = g0 * G_BYTES + tl.arange(0, BLOCK_KB)
            mask_kb = offs_kb < k_bytes
            offs_k_even = offs_kb * 2

            x_even = tl.load(
                x_ptr + offs_k_even, mask=mask_kb, other=0.0, eviction_policy="evict_last"
            ).to(tl.float32)
            x_odd = tl.load(
                x_ptr + offs_k_even + 1, mask=mask_kb, other=0.0, eviction_policy="evict_last"
            ).to(tl.float32)

            mask_nb = mask_n[:, None] & mask_kb[None, :]
            # 0x88 decodes both nibbles to 8 - 8 = 0 for masked positions.
            b = tl.load(
                w_ptr + offs_n[:, None] * k_bytes + offs_kb[None, :],
                mask=mask_nb,
                other=0x88,
                eviction_policy="evict_first",
            )

            low = ((b & 0x0F).to(tl.int32) - 8).to(tl.float32)
            high = (((b >> 4) & 0x0F).to(tl.int32) - 8).to(tl.float32)
            prod = low * x_even[None, :] + high * x_odd[None, :]
            prod_g = tl.reshape(prod, (BLOCK_N, GROUPS_PER_TILE, G_BYTES))
            acc_g += tl.sum(prod_g, axis=2) * s

        acc = tl.sum(acc_g, axis=1)
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            acc += bias
        tl.store(out_ptr + offs_n, acc, mask=mask_n)

    @triton.jit
    def _quarot_int4_matvec_kernel(
        x_ptr,
        w_ptr,
        s_ptr,
        bias_ptr,
        out_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        G: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_KB: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        k_bytes = K // 2
        n_groups = K // G

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for kb0 in range(0, k_bytes, BLOCK_KB):
            offs_kb = kb0 + tl.arange(0, BLOCK_KB)
            mask_kb = offs_kb < k_bytes
            offs_k_even = offs_kb * 2
            offs_k_odd = offs_k_even + 1

            x_even = tl.load(
                x_ptr + offs_k_even, mask=mask_kb, other=0.0, eviction_policy="evict_last"
            ).to(tl.float32)
            x_odd = tl.load(
                x_ptr + offs_k_odd, mask=mask_kb, other=0.0, eviction_policy="evict_last"
            ).to(tl.float32)

            mask_nb = mask_n[:, None] & mask_kb[None, :]
            # 0x88 decodes both nibbles to 8 - 8 = 0 for masked positions.
            b = tl.load(
                w_ptr + offs_n[:, None] * k_bytes + offs_kb[None, :],
                mask=mask_nb,
                other=0x88,
                eviction_policy="evict_first",
            )
            g_idx = offs_k_even // G
            s = tl.load(
                s_ptr + offs_n[:, None] * n_groups + g_idx[None, :],
                mask=mask_nb,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

            low = ((b & 0x0F).to(tl.int32) - 8).to(tl.float32)
            high = (((b >> 4) & 0x0F).to(tl.int32) - 8).to(tl.float32)
            acc += tl.sum((low * x_even[None, :] + high * x_odd[None, :]) * s, axis=1)

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            acc += bias
        tl.store(out_ptr + offs_n, acc, mask=mask_n)

    @triton.jit
    def _quarot_int4_matmul_kernel(
        x_ptr,
        w_ptr,
        s_ptr,
        bias_ptr,
        out_ptr,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        G: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        USE_FP16_DOT: tl.constexpr,
        USE_BF16_DOT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_KB: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N
        k_bytes = K // 2
        n_groups = K // G

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        dot_dtype = tl.float16 if USE_FP16_DOT else (tl.bfloat16 if USE_BF16_DOT else tl.float32)
        for kb0 in range(0, k_bytes, BLOCK_KB):
            offs_kb = kb0 + tl.arange(0, BLOCK_KB)
            mask_kb = offs_kb < k_bytes
            offs_k_even = offs_kb * 2
            offs_k_odd = offs_k_even + 1

            mask_mk = mask_m[:, None] & mask_kb[None, :]
            x_even = tl.load(x_ptr + offs_m[:, None] * K + offs_k_even[None, :], mask=mask_mk, other=0.0).to(dot_dtype)
            x_odd = tl.load(x_ptr + offs_m[:, None] * K + offs_k_odd[None, :], mask=mask_mk, other=0.0).to(dot_dtype)

            mask_nb = mask_n[:, None] & mask_kb[None, :]
            b = tl.load(w_ptr + offs_n[:, None] * k_bytes + offs_kb[None, :], mask=mask_nb, other=0x88)
            g_idx = offs_k_even // G
            s = tl.load(s_ptr + offs_n[:, None] * n_groups + g_idx[None, :], mask=mask_nb, other=0.0).to(tl.float32)

            w_even = (((b & 0x0F).to(tl.int32) - 8).to(tl.float32) * s).to(dot_dtype)
            w_odd = ((((b >> 4) & 0x0F).to(tl.int32) - 8).to(tl.float32) * s).to(dot_dtype)

            if USE_FP16_DOT or USE_BF16_DOT:
                acc += tl.dot(x_even, tl.trans(w_even))
                acc += tl.dot(x_odd, tl.trans(w_odd))
            else:
                acc += tl.dot(x_even, tl.trans(w_even), input_precision="ieee")
                acc += tl.dot(x_odd, tl.trans(w_odd), input_precision="ieee")

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            acc += bias[None, :]
        mask_out = mask_m[:, None] & mask_n[None, :]
        tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=mask_out)


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _launch_quarot_matvec(
    x_2d: torch.Tensor,
    w_c: torch.Tensor,
    s_c: torch.Tensor,
    b_c: torch.Tensor,
    out: torch.Tensor,
    out_features: int,
    in_features: int,
    group_size: int,
    has_bias: bool,
) -> None:
    block_n = 8 if out_features >= 128 else 4
    k_bytes = in_features // 2
    block_kb = 128 if k_bytes >= 128 else (64 if k_bytes >= 64 else 32)
    g_bytes = group_size // 2
    grid = (triton.cdiv(out_features, block_n),)
    if _is_power_of_two(g_bytes) and g_bytes <= block_kb and (block_kb % g_bytes == 0):
        _quarot_int4_matvec_grouped_kernel[grid](
            x_2d,
            w_c,
            s_c,
            b_c,
            out,
            N=out_features,
            K=in_features,
            G_BYTES=g_bytes,
            GROUPS_PER_TILE=block_kb // g_bytes,
            HAS_BIAS=has_bias,
            BLOCK_N=block_n,
            BLOCK_KB=block_kb,
            num_warps=4,
        )
    else:
        _quarot_int4_matvec_kernel[grid](
            x_2d,
            w_c,
            s_c,
            b_c,
            out,
            N=out_features,
            K=in_features,
            G=group_size,
            HAS_BIAS=has_bias,
            BLOCK_N=block_n,
            BLOCK_KB=block_kb,
            num_warps=4,
        )


def can_use_triton_quarot(in_features: int, group_size: int, device: torch.device) -> bool:
    return bool(
        HAS_TRITON
        and device.type == "cuda"
        and in_features >= 32
        and in_features % 2 == 0
        and group_size >= 2
        and group_size % 2 == 0
        and in_features % group_size == 0
    )


def _dequantize_rotated_weight(
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    in_features: int,
    group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    out_features = packed_weight.shape[0]
    n_groups = in_features // group_size
    low = (packed_weight & 0x0F).to(torch.int8) - 8
    high = ((packed_weight >> 4) & 0x0F).to(torch.int8) - 8
    q = torch.stack((low, high), dim=-1).flatten(-2).to(torch.float32)
    scale = weight_scale.to(torch.float32).unsqueeze(-1)
    return (q.view(out_features, n_groups, group_size) * scale).view(out_features, in_features).to(dtype)


def _quarot_int4_pytorch_linear(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    w_rot = _dequantize_rotated_weight(packed_weight, weight_scale, x.shape[-1], group_size, x.dtype)
    b = None if bias is None else bias.to(x.dtype)
    return torch.nn.functional.linear(x, w_rot, b)


@torch.library.custom_op("llm2::quarot_int4_linear", mutates_args=())
def _quarot_int4_linear_op(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    if not can_use_triton_quarot(x.shape[-1], group_size, x.device):
        return _quarot_int4_pytorch_linear(x, packed_weight, weight_scale, bias, group_size)

    orig_shape = x.shape
    in_features = orig_shape[-1]
    out_features = packed_weight.shape[0]
    x_2d = x.reshape(-1, in_features).contiguous()
    m = x_2d.shape[0]
    w_c = packed_weight.contiguous()
    s_c = weight_scale.contiguous()
    has_bias = bias is not None
    b_c = bias.contiguous() if has_bias else w_c.new_empty((1,), dtype=x.dtype)
    out = torch.empty((m, out_features), device=x.device, dtype=x.dtype)

    with torch.cuda.device(x.device):
        if m == 1:
            _launch_quarot_matvec(
                x_2d, w_c, s_c, b_c, out, out_features, in_features, group_size, has_bias
            )
        else:
            block_m = 32 if m >= 32 else 16
            block_n = 32
            block_kb = 32
            grid = (triton.cdiv(m, block_m), triton.cdiv(out_features, block_n))
            _quarot_int4_matmul_kernel[grid](
                x_2d,
                w_c,
                s_c,
                b_c,
                out,
                M=m,
                N=out_features,
                K=in_features,
                G=group_size,
                HAS_BIAS=has_bias,
                USE_FP16_DOT=(x.dtype == torch.float16),
                USE_BF16_DOT=(x.dtype == torch.bfloat16),
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_KB=block_kb,
                num_warps=4,
            )
    return out.reshape(*orig_shape[:-1], out_features)


@_quarot_int4_linear_op.register_fake
def _quarot_int4_linear_fake(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], packed_weight.shape[0]), device=x.device, dtype=x.dtype)


def _quarot_int4_setup_context(ctx, inputs, output) -> None:
    x, packed_weight, weight_scale, bias, group_size = inputs
    ctx.save_for_backward(packed_weight, weight_scale)
    ctx.in_features = x.shape[-1]
    ctx.group_size = group_size
    ctx.has_bias = bias is not None


def _quarot_int4_backward(ctx, grad_output: torch.Tensor):
    packed_weight, weight_scale = ctx.saved_tensors
    grad_x = None
    grad_bias = None
    if ctx.needs_input_grad[0]:
        w_rot = _dequantize_rotated_weight(
            packed_weight, weight_scale, ctx.in_features, ctx.group_size, grad_output.dtype
        )
        grad_x = grad_output.matmul(w_rot)
    if ctx.has_bias and ctx.needs_input_grad[3]:
        grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(dim=0)
    return grad_x, None, None, grad_bias, None


_quarot_int4_linear_op.register_autograd(_quarot_int4_backward, setup_context=_quarot_int4_setup_context)


def quarot_int4_linear(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    *,
    use_triton: bool = True,
) -> torch.Tensor:
    if use_triton and can_use_triton_quarot(x.shape[-1], group_size, x.device):
        # Fast direct matvec launch for m==1 inference decode (avoids custom_op autograd dispatcher overhead).
        in_features = x.shape[-1]
        if not x.requires_grad and x.numel() == in_features:
            out_features = packed_weight.shape[0]
            x_2d = x.reshape(1, in_features)
            if not x_2d.is_contiguous():
                x_2d = x_2d.contiguous()
            w_c = packed_weight if packed_weight.is_contiguous() else packed_weight.contiguous()
            s_c = weight_scale if weight_scale.is_contiguous() else weight_scale.contiguous()
            has_bias = bias is not None
            b_c = (bias if bias.is_contiguous() else bias.contiguous()) if has_bias else s_c
            out = torch.empty((*x.shape[:-1], out_features), device=x.device, dtype=x.dtype)
            with torch.cuda.device(x.device):
                _launch_quarot_matvec(
                    x_2d, w_c, s_c, b_c, out, out_features, in_features, group_size, has_bias
                )
            return out
        return _quarot_int4_linear_op(x, packed_weight, weight_scale, bias, group_size)
    return _quarot_int4_pytorch_linear(x, packed_weight, weight_scale, bias, group_size)
