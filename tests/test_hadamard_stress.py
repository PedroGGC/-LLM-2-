import torch
import math

def fast_walsh_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Aplica FWHT normalizado sobre a última dimensão d (potência de 2).
    Complexidade O(d log d), puramente aditivo."""
    b, t, d = x.shape
    h = 1
    while h < d:
        x = x.view(b, t, d // (2 * h), 2, h)
        a = x[:, :, :, 0, :]
        b_vec = x[:, :, :, 1, :]
        x = torch.stack([a + b_vec, a - b_vec], dim=3)
        h *= 2
    return x.view(b, t, d) / (d ** 0.5)

def test_dimensions():
    print("--- Test 1: Dimension compatibility ---")
    x_4096 = torch.randn(1, 1, 4096)
    try:
        y_4096 = fast_walsh_hadamard_transform(x_4096)
        print("d=4096: OK, shape =", y_4096.shape)
    except Exception as e:
        print("d=4096: FAILED ->", type(e).__name__, e)

    x_384 = torch.randn(1, 1, 384)
    try:
        y_384 = fast_walsh_hadamard_transform(x_384)
        print("d=384: OK, shape =", y_384.shape)
    except Exception as e:
        print("d=384: CRASHED ->", type(e).__name__, e)

def test_orthogonality_and_inversion():
    print("\n--- Test 2: Orthogonality & Exact Inversion ---")
    d = 256
    x = torch.randn(2, 4, d)
    y = fast_walsh_hadamard_transform(x)
    # Applying FWHT twice should recover x because H = H^T = H^-1
    x_rec = fast_walsh_hadamard_transform(y)
    err = (x - x_rec).abs().max().item()
    print(f"d={d}: Max reconstruction error after 2 FWHT calls: {err:.2e}")
    norm_orig = torch.norm(x, dim=-1)
    norm_rot = torch.norm(y, dim=-1)
    norm_err = (norm_orig - norm_rot).abs().max().item()
    print(f"d={d}: L2 norm conservation error: {norm_err:.2e}")

def test_rmsnorm_commutation():
    print("\n--- Test 3: RMSNorm Commutation with Rotation ---")
    d = 256
    x = torch.randn(1, 1, d)
    # Trained RMSNorm has non-uniform gamma weights!
    gamma = torch.randn(d).abs() + 0.1 # heterogeneous scales

    # Function 1: standard RMSNorm
    def rms_norm(val, g):
        rms = torch.sqrt(torch.mean(val ** 2, dim=-1, keepdim=True) + 1e-6)
        return (val / rms) * g

    # Path A: RMSNorm then rotate
    y_A = fast_walsh_hadamard_transform(rms_norm(x, gamma))

    # Path B: Rotate then RMSNorm
    y_B = rms_norm(fast_walsh_hadamard_transform(x), gamma)

    diff = (y_A - y_B).abs().max().item()
    rel_diff = diff / y_A.abs().max().item()
    print(f"Difference between RMSNorm(x*H) and RMSNorm(x)*H: max_abs={diff:.4f}, rel={rel_diff:.4%}")

    # What if gamma is absorbed into W?
    # Unscaled RMSNorm:
    def unscaled_rms_norm(val):
        rms = torch.sqrt(torch.mean(val ** 2, dim=-1, keepdim=True) + 1e-6)
        return val / rms

    y_A_unscaled = fast_walsh_hadamard_transform(unscaled_rms_norm(x))
    y_B_unscaled = unscaled_rms_norm(fast_walsh_hadamard_transform(x))
    unscaled_diff = (y_A_unscaled - y_B_unscaled).abs().max().item()
    print(f"Unscaled RMSNorm commutation: max_abs={unscaled_diff:.2e}")

def test_outlier_dispersion():
    print("\n--- Test 4: Outlier Dispersion ---")
    d = 4096
    # Scenario 1: Single coordinate outlier M = 1000, others ~ N(0, 1)
    x1 = torch.randn(1, 1, d)
    x1[0, 0, 42] = 1000.0
    y1 = fast_walsh_hadamard_transform(x1)
    max_orig1 = x1.abs().max().item()
    max_rot1 = y1.abs().max().item()
    ratio1 = max_orig1 / max_rot1
    print(f"Single spike (M=1000, d={d}): Max before={max_orig1:.1f}, Max after={max_rot1:.1f}, Reduction factor={ratio1:.2f}x (sqrt(d)={math.sqrt(d):.1f}x)")

    # Scenario 2: Multiple persistent outliers (e.g. 8 channels with M=500)
    x2 = torch.randn(1, 1, d)
    x2[0, 0, [10, 50, 100, 200, 500, 1000, 2000, 3000]] = 500.0
    y2 = fast_walsh_hadamard_transform(x2)
    max_orig2 = x2.abs().max().item()
    max_rot2 = y2.abs().max().item()
    ratio2 = max_orig2 / max_rot2
    print(f"8 spikes (M=500, d={d}): Max before={max_orig2:.1f}, Max after={max_rot2:.1f}, Reduction factor={ratio2:.2f}x")

    # Scenario 3: Correlated alignment with a Walsh basis vector (worst-case adversary)
    # Row 0 of Sylvester Hadamard is all +1/sqrt(d)
    x_worst = torch.ones(1, 1, d) * 10.0
    y_worst = fast_walsh_hadamard_transform(x_worst)
    print(f"Constant vector (all 10.0): Max before={x_worst.abs().max().item():.1f}, Max after={y_worst.abs().max().item():.1f} (Amplified by sqrt(d)!)")

if __name__ == "__main__":
    test_dimensions()
    test_orthogonality_and_inversion()
    test_rmsnorm_commutation()
    test_outlier_dispersion()
