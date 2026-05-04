import random
from typing import List

A, B = 2.0, 5.0


def f(x: float) -> float:
    return x * x


def analytical_integral() -> float:
    return (B**3 - A**3) / 3.0


def pdf_power_k(x: float, k: int) -> float:
    if x < A or x > B:
        return 0.0
    c = (k + 1.0) / (B ** (k + 1) - A ** (k + 1))
    return c * (x**k)


def inv_cdf_power_k(u: float, k: int) -> float:
    ak, bk = A ** (k + 1), B ** (k + 1)
    return (ak + u * (bk - ak)) ** (1.0 / (k + 1))


def uniform(a: float, b: float, rng: random.Random) -> float:
    return a + rng.random() * (b - a)


def strata(h: float) -> List[tuple[float, float]]:
    out: List[tuple[float, float]] = []
    x = A
    while x < B - 1e-15:
        r = min(x + h, B)
        out.append((x, r))
        x = r
    return out


def counts_per_stratum(n: int, m: int) -> List[int]:
    base, rem = n // m, n % m
    return [base + (1 if i < rem else 0) for i in range(m)]


def phi(t: float) -> float:
    return f(A + (B - A) * t)
