"""NumPy Scientific Computing Lab — vectorized arithmetic, broadcasting,
and core linear-algebra operations.
"""
import numpy as np


# ---------- vectorized arithmetic ----------

def add(a, b):
    return np.add(a, b)


def multiply(a, b):
    return np.multiply(a, b)


def power(a, b):
    return np.power(a, b)


def safe_divide(a, b):
    """Element-wise division returning inf/nan instead of raising on /0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.true_divide(a, b)


def stats(a):
    """Mean, std, min, max, and sum of an array in one call."""
    a = np.asarray(a, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "sum": float(np.sum(a)),
    }


# ---------- broadcasting demos ----------

def rowwise_normalize(matrix):
    """Subtract each row's mean (broadcasting over axis 1)."""
    m = np.asarray(matrix, dtype=float)
    return m - m.mean(axis=1, keepdims=True)


def pairwise_distances(points):
    """Euclidean distance matrix for a set of points, no Python loops."""
    p = np.asarray(points, dtype=float)
    diff = p[:, None, :] - p[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def outer_product_table(n=12):
    """Multiplication table via broadcasting: arange[:, None] * arange."""
    v = np.arange(1, n + 1)
    return v[:, None] * v[None, :]


# ---------- matrix operations ----------

def matmul(a, b):
    return np.matmul(a, b)


def inverse(a):
    a = np.asarray(a, dtype=float)
    if a.shape[0] != a.shape[1]:
        raise ValueError("inverse requires a square matrix")
    return np.linalg.inv(a)


def determinant(a):
    return float(np.linalg.det(a))


def eigen(a):
    """Returns (eigenvalues, eigenvectors) of a square matrix."""
    a = np.asarray(a, dtype=float)
    vals, vecs = np.linalg.eig(a)
    return vals, vecs


def solve_linear(a, b):
    """Solve Ax = b for x. Raises on singular matrices."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.linalg.solve(a, b)


def least_squares_fit(x, y, degree=1):
    """Polynomial least-squares fit; returns coefficients (highest power first)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.polyfit(x, y, degree)
