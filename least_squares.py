"""Least-squares fitting demo: noisy data -> polynomial fit -> plot."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scientific import least_squares_fit

rng = np.random.default_rng(42)
x = np.linspace(0, 10, 60)
y_true = 0.5 * x ** 2 - 3 * x + 2
y = y_true + rng.normal(0, 2, size=x.shape)

coeffs = least_squares_fit(x, y, degree=2)
y_fit = np.polyval(coeffs, x)
residual = np.sqrt(np.mean((y - y_fit) ** 2))

print(f"true coeffs:      [0.5, -3, 2]")
print(f"fitted coeffs:    {[round(c, 3) for c in coeffs]}")
print(f"RMSE of fit:      {residual:.3f}")

plt.figure(figsize=(9, 5))
plt.scatter(x, y, s=18, alpha=0.6, label="noisy data")
plt.plot(x, y_true, "g--", label="true curve")
plt.plot(x, y_fit, "r-", linewidth=2, label="least-squares fit")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Least-squares quadratic fit")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("least_squares.png", dpi=120)
print("saved least_squares.png")
