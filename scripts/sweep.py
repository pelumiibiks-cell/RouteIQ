"""Grid search over the router's tunable constants.

    PYTHONPATH=. python scripts/sweep.py [--top N] [--quick]

The README has always described UTILITY_WEIGHTS as "isolated for easy tuning".
They were isolated and nothing tuned them, so every weight in the system was a
number someone picked once and never checked.

Scoring is banded accuracy on the **dev** split only, with mean absolute tier
error as the tie-break (a config that misses by one tier is better than one that
misses by two, and banded accuracy alone cannot see that). The best config is
then reported on the held-out **test** split exactly once, at the end. Nothing
in here reads the test split while searching -- that is the whole point of it.
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass

from app.analysis import complexity_scorer
from app.evaluation.runner import resolve_cases, run_benchmark
from app.routing import router


@dataclass
class Config:
    lead_weight: float
    multimodal_reasoning: float
    context_length: float
    reliability: float
    uncertainty_gate: float
    underpowered_weight: float

    def label(self) -> str:
        return (
            f"lead={self.lead_weight:.2f} mm={self.multimodal_reasoning:.2f} "
            f"ctx={self.context_length:.2f} rel={self.reliability:.2f} "
            f"gate={self.uncertainty_gate:.2f} under={self.underpowered_weight:.2f}"
        )


def apply(cfg: Config) -> None:
    ranks = complexity_scorer._RANK_WEIGHTS
    complexity_scorer._RANK_WEIGHTS = (cfg.lead_weight, ranks[1], ranks[2])
    complexity_scorer.DIMENSION_WEIGHTS["multimodal_reasoning_depth"] = cfg.multimodal_reasoning
    complexity_scorer.DIMENSION_WEIGHTS["context_length"] = cfg.context_length
    complexity_scorer.DIMENSION_WEIGHTS["reliability_requirement"] = cfg.reliability
    router._UNCERTAINTY_GATE = cfg.uncertainty_gate
    router.UTILITY_WEIGHTS["underpowered"] = cfg.underpowered_weight


def score(cases) -> tuple[float, float, float, float]:
    r = run_benchmark(cases=cases)
    return r.banded_accuracy, r.mean_abs_tier_error, r.routing_accuracy, r.overkill_rate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="coarser grid")
    args = ap.parse_args()

    step = 2 if args.quick else 1
    grid = list(
        itertools.product(
            [0.82, 0.88, 0.92, 0.98][::step],
            [0.85, 1.0, 1.15, 1.3, 1.45][::step],  # multimodal_reasoning_depth
            [0.30, 0.40, 0.50][::step],         # context_length
            [0.6, 0.7, 0.85][::step],           # reliability_requirement
            [0.25, 0.35, 0.45][::step],         # uncertainty gate
            [0.45, 0.7][::step],                # underpowered utility weight
        )
    )

    baseline = Config(
        complexity_scorer._RANK_WEIGHTS[0],
        complexity_scorer.DIMENSION_WEIGHTS["multimodal_reasoning_depth"],
        complexity_scorer.DIMENSION_WEIGHTS["context_length"],
        complexity_scorer.DIMENSION_WEIGHTS["reliability_requirement"],
        router._UNCERTAINTY_GATE,
        router.UTILITY_WEIGHTS["underpowered"],
    )

    dev = resolve_cases("dev")
    test = resolve_cases("test")

    print(f"Sweeping {len(grid)} configs against the dev split ({len(dev)} cases)...\n")
    results = []
    for combo in grid:
        cfg = Config(*combo)
        apply(cfg)
        banded, err, exact, over = score(dev)
        results.append((banded, -err, exact, cfg, over))

    # Tie-break declared up front, not chosen after seeing the numbers: among
    # configs the dev split cannot distinguish, prefer the one that never
    # overkills. That is the project's stated philosophy -- use the weakest
    # model likely to succeed -- so it is a product constraint, not a metric.
    results.sort(key=lambda r: (r[0], r[1], -r[4], r[2]), reverse=True)

    print(f"Top {args.top} on dev:")
    print(f"  {'banded':>7} {'exact':>7} {'err':>6} {'over':>6}  config")
    for banded, neg_err, exact, cfg, over in results[:args.top]:
        print(f"  {banded * 100:6.1f}% {exact * 100:6.1f}% {-neg_err:6.3f} {over * 100:5.1f}%  {cfg.label()}")

    best = results[0][3]
    print(f"\nBest dev config: {best.label()}")
    apply(best)
    b_banded, b_err, b_exact, b_over = score(dev)
    t_banded, t_err, t_exact, t_over = score(test)
    print(f"  dev  banded {b_banded * 100:.1f}%  exact {b_exact * 100:.1f}%  err {b_err:.3f}  overkill {b_over * 100:.1f}%")
    print(f"  test banded {t_banded * 100:.1f}%  exact {t_exact * 100:.1f}%  err {t_err:.3f}  overkill {t_over * 100:.1f}%")

    apply(baseline)
    c_banded, c_err, c_exact, c_over = score(dev)
    ct_banded, ct_err, ct_exact, ct_over = score(test)
    print(f"\nCurrent committed config: {baseline.label()}")
    print(f"  dev  banded {c_banded * 100:.1f}%  exact {c_exact * 100:.1f}%  err {c_err:.3f}  overkill {c_over * 100:.1f}%")
    print(f"  test banded {ct_banded * 100:.1f}%  exact {ct_exact * 100:.1f}%  err {ct_err:.3f}  overkill {ct_over * 100:.1f}%")

    gap = (b_banded - t_banded) * 100
    print(f"\nDev-to-test gap for the best config: {gap:.1f} points.")
    print("A large gap means the config is fitted to dev, not to routing.")


if __name__ == "__main__":
    main()
