import numpy as np

def conformal_residual_quantile(residuals: np.ndarray, alpha: float) -> float:
    n = int(residuals.size)
    k = int(np.ceil((n + 1) * alpha))
    q = min(1.0, max(0.0, k / n))
    return float(np.quantile(residuals, q, method="higher")), k

residuals = np.arange(1, 101) # 1 to 100
alpha = 0.9
val, k = conformal_residual_quantile(residuals, alpha)
print(f"alpha={alpha}, n={len(residuals)}, k={k}, val={val}")
# The k-th element is literally k (since it's 1-indexed and array is 1..100)
print(f"Expected: {k}")
