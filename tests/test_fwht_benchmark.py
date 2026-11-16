import time
import torch
import math

def fast_walsh_hadamard_transform_loops(x: torch.Tensor) -> torch.Tensor:
    b, t, d = x.shape
    h = 1
    while h < d:
        x = x.view(b, t, d // (2 * h), 2, h)
        a = x[:, :, :, 0, :]
        b_vec = x[:, :, :, 1, :]
        x = torch.stack([a + b_vec, a - b_vec], dim=3)
        h *= 2
    return x.view(b, t, d) / (d ** 0.5)

def benchmark_fwht():
    print("=== Empirical Benchmark: FWHT vs GEMM ===")
    d = 4096
    # 1. Theoretical FLOPs Comparison:
    # GEMM (Linear d -> d): 2 * d * d FLOPs (multiply-adds)
    # FWHT (d): d * log2(d) additions/subtractions + d divisions
    gemm_flops = 2 * d * d
    fwht_ops = d * int(math.log2(d)) # 4096 * 12 = 49,152 ops
    theoretical_ratio = (fwht_ops / gemm_flops) * 100
    print(f"Theoretical Complexity for d={d}:")
    print(f"  GEMM (d x d): {gemm_flops:,} FLOPs")
    print(f"  FWHT (d): {fwht_ops:,} Operations (O(d log d))")
    print(f"  Theoretical Ops Ratio: {theoretical_ratio:.3f}% ({theoretical_ratio:.2f}% of a single GEMM)")

    # 2. Runtime in Python/PyTorch (CPU):
    # Decoding regime: B=1, T=1
    x_dec = torch.randn(1, 1, d)
    lin = torch.nn.Linear(d, d, bias=False)

    # Warmup
    for _ in range(10):
        _ = lin(x_dec)
        _ = fast_walsh_hadamard_transform_loops(x_dec)

    N = 50
    t0 = time.perf_counter()
    for _ in range(N):
        _ = lin(x_dec)
    t_gemm = (time.perf_counter() - t0) / N * 1e6

    t0 = time.perf_counter()
    for _ in range(N):
        _ = fast_walsh_hadamard_transform_loops(x_dec)
    t_fwht = (time.perf_counter() - t0) / N * 1e6

    print(f"\nEmpirical Runtime on PyTorch (Decode B=1, T=1, d={d}):")
    print(f"  Linear layer GEMM: {t_gemm:.1f} us")
    print(f"  PyTorch naive FWHT (views/stacks loop): {t_fwht:.1f} us")
    print(f"  Naive PyTorch FWHT overhead: {(t_fwht / t_gemm)*100:.1f}%")

    print("\n--- Key Architectural Insight ---")
    print("1. In theoretical arithmetic, FWHT is indeed 49,152 ops vs 33.5M FLOPs (~0.147% of a GEMM).")
    print("2. BUT naive PyTorch implementation launches log2(d)=12 distinct Python loop iterations with tensor reshaping/stacking, incurring massive kernel launch and memory allocation overhead.")
    print("3. In QuaRot, overhead < 1.5% is ONLY achieved when using a custom FUSED C++/CUDA kernel (e.g. fast-hadamard-transform using GPU shared memory warp-shuffle).")
    print("4. The document in line 508 provides the naive Python loop as 'Pseudo-Código do FWHT Otimizado em PyTorch / Triton', but does NOT clarify that this Python loop would be a severe bottleneck without a fused Triton/CUDA kernel!")

if __name__ == "__main__":
    benchmark_fwht()
