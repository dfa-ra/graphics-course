
import math
import random

import integrators
from utils import A, B, analytical_integral

SEED = 42
SAMPLE_SIZES = (100, 1000, 10000, 100000)
# Независимые прогоны Монте-Карло на одно и то же N; в таблице — среднее I_hat.
TRIALS = 5

COL = [22, 22, 8, 14, 12, 10, 12]


def mean_estimate(n: int, row_id: int, run) -> float:
    """run(rng) -> оценка интеграла; TRIALS раз с разными seed."""
    s = 0.0
    for t in range(TRIALS):
        rng = random.Random(SEED + int(n) * 1_000_003 + row_id * 97_981 + t * 8_388_607)
        s += run(rng)
    return s / TRIALS


def line(ch: str = "-") -> str:
    return ch * (sum(COL) + 3 * (len(COL) - 1))


def main() -> None:
    i_true = analytical_integral()
    assert abs(i_true - 39.0) < 1e-12

    for n in SAMPLE_SIZES:
        rows = [
            ("Analytic", "", i_true),
            (
                "Plain Monte Carlo",
                "",
                mean_estimate(n, 1, lambda r: integrators.plain_mc(n, r)),
            ),
            (
                "Stratified Monte Carlo",
                "h=1.0",
                mean_estimate(n, 2, lambda r: integrators.stratified(n, 1.0, r)),
            ),
            (
                "Stratified Monte Carlo",
                "h=0.5",
                mean_estimate(n, 3, lambda r: integrators.stratified(n, 0.5, r)),
            ),
            (
                "Importance Sampling",
                "p(x)~x",
                mean_estimate(n, 4, lambda r: integrators.importance(n, 1, r)),
            ),
            (
                "Importance Sampling",
                "p(x)~x^2",
                mean_estimate(n, 5, lambda r: integrators.importance(n, 2, r)),
            ),
            (
                "Importance Sampling",
                "p(x)~x^3",
                mean_estimate(n, 6, lambda r: integrators.importance(n, 3, r)),
            ),
            (
                "MIS",
                "balance heuristic",
                mean_estimate(n, 7, lambda r: integrators.mis_balance(n, r)),
            ),
            (
                "MIS",
                "power heuristic",
                mean_estimate(n, 8, lambda r: integrators.mis_power(n, r)),
            ),
            (
                "Russian Roulette",
                "R=0.5",
                mean_estimate(n, 9, lambda r: integrators.russian_roulette(n, 0.5, r)),
            ),
            (
                "Russian Roulette",
                "R=0.75",
                mean_estimate(n, 10, lambda r: integrators.russian_roulette(n, 0.75, r)),
            ),
            (
                "Russian Roulette",
                "R=0.95",
                mean_estimate(n, 11, lambda r: integrators.russian_roulette(n, 0.95, r)),
            ),
        ]

        print()
        print(f"=== Sample size N = {n} ===")
        print(line("="))
        hdr = " | ".join(
            f"{h:<{w}}"
            for h, w in zip(
                ["Method", "Params", "N", "I_hat", "|err|", "rel_err", "Δ_I (lab)"], COL
            )
        )
        print(hdr)
        print(line("-"))
        delta_lab = i_true / math.sqrt(n)
        for method, params, est in rows:
            err = abs(est - i_true)
            rel = err / i_true if i_true else float("inf")
            rel_s = f"{rel:>{COL[5]}.6f}" if not math.isinf(rel) else f"{'inf':>{COL[5]}}"
            print(
                f"{method:<{COL[0]}} | {params:<{COL[1]}} | {n:>{COL[2]}} | "
                f"{est:>{COL[3]}.6f} | {err:>{COL[4]}.6f} | {rel_s} | {delta_lab:>{COL[6]}.6f}"
            )
        print(line("="))


if __name__ == "__main__":
    main()
