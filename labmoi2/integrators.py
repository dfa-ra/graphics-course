import random

from utils import A, B, counts_per_stratum, f, inv_cdf_power_k, pdf_power_k, phi, strata, uniform


def plain_mc(n: int, rng: random.Random) -> float:
    w = B - A
    s = 0.0
    for _ in range(n):
        s += f(uniform(A, B, rng))
    return w * s / n


def stratified(n: int, h: float, rng: random.Random) -> float:
    st = strata(h)
    cnt = counts_per_stratum(n, len(st))
    total = 0.0
    for (lo, hi), nj in zip(st, cnt):
        if nj == 0:
            continue
        sj = sum(f(uniform(lo, hi, rng)) for _ in range(nj))
        total += (hi - lo) * (sj / nj)
    return total


def importance(n: int, k: int, rng: random.Random) -> float:
    s = 0.0
    for _ in range(n):
        x = inv_cdf_power_k(rng.random(), k)
        s += f(x) / pdf_power_k(x, k)
    return s / n


def _mis_split(n: int) -> tuple[int, int]:
    n1 = (n + 1) // 2
    return n1, n - n1


def mis_balance(n: int, rng: random.Random) -> float:
    """mean по q₁∝x + mean по q₂∝x³; w_i = q_i/(q₁+q₂)."""
    n1, n2 = _mis_split(n)
    s1 = 0.0
    for _ in range(n1):
        x = inv_cdf_power_k(rng.random(), 1)
        q1, q2 = pdf_power_k(x, 1), pdf_power_k(x, 3)
        s1 += (q1 / (q1 + q2)) * f(x) / q1
    s2 = 0.0
    for _ in range(n2):
        x = inv_cdf_power_k(rng.random(), 3)
        q1, q2 = pdf_power_k(x, 1), pdf_power_k(x, 3)
        s2 += (q2 / (q1 + q2)) * f(x) / q2
    return (s1 / n1 if n1 else 0.0) + (s2 / n2 if n2 else 0.0)


def mis_power(n: int, rng: random.Random) -> float:
    """β=2: w_i = q_i²/(q₁²+q₂²)."""
    n1, n2 = _mis_split(n)
    s1 = 0.0
    for _ in range(n1):
        x = inv_cdf_power_k(rng.random(), 1)
        q1, q2 = pdf_power_k(x, 1), pdf_power_k(x, 3)
        d = q1 * q1 + q2 * q2
        s1 += (q1 * q1 / d) * f(x) / q1
    s2 = 0.0
    for _ in range(n2):
        x = inv_cdf_power_k(rng.random(), 3)
        q1, q2 = pdf_power_k(x, 1), pdf_power_k(x, 3)
        d = q1 * q1 + q2 * q2
        s2 += (q2 * q2 / d) * f(x) / q2
    return (s1 / n1 if n1 else 0.0) + (s2 / n2 if n2 else 0.0)


def russian_roulette(n: int, r: float, rng: random.Random) -> float:
    w = B - A
    s = 0.0
    for _ in range(n):
        xi = rng.random()
        if xi <= r:
            s += (1.0 / r) * phi(xi / r)
    return w * s / n
