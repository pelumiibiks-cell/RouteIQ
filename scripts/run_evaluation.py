"""Runs the evaluation benchmark and prints the report. Run with:

    PYTHONPATH=. python scripts/run_evaluation.py [dev|test|all]

`dev` (the default) is the tunable split. `test` is held out -- never tune
against it, only report it.
"""
from __future__ import annotations

import sys

from app.evaluation.runner import compare_to_always_strongest, resolve_cases, run_benchmark


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    cases = resolve_cases(split)
    report = run_benchmark(cases=cases)
    print(f"Split: {split} ({len(cases)} cases)")
    print("=== Router Benchmark ===")
    print(f"Total cases:          {report.total}")
    print(f"Banded accuracy:      {report.banded_accuracy * 100:.1f}%   (within acceptable_tiers)")
    print(f"Exact-tier accuracy:  {report.routing_accuracy * 100:.1f}%   (strict)")
    print(f"Underpowered rate:    {report.underpowered_rate * 100:.1f}%   (below the band)")
    print(f"Overkill rate:        {report.overkill_rate * 100:.1f}%   (above the band)")
    print(f"Mean |tier error|:    {report.mean_abs_tier_error:.2f}")
    print(f"Average cost:         ${report.average_cost:.6f}")
    print(f"Average latency:      {report.average_latency_ms:.0f} ms")
    print(f"Average confidence:   {report.average_confidence * 100:.1f}%")
    print()
    print("Tier confusion (expected -> got):")
    for exp, gots in sorted(report.tier_confusion.items()):
        print(f"  {exp}: {gots}")
    print()

    cost_cmp = compare_to_always_strongest(cases=cases, report=report)
    print("=== Cost vs. Always-Strongest-Model ===")
    print(f"Router total cost:            ${cost_cmp['router_total_cost']:.6f}")
    print(f"Always-strongest total cost:  ${cost_cmp['always_strongest_total_cost']:.6f}")
    print(f"Cost savings:                 {cost_cmp['cost_savings_pct']}%")
    print()

    print("Per-case detail:")
    for r in report.results:
        if r.correct:
            status = "OK"
        elif r.in_band:
            status = "in-band"
        else:
            status = "UNDER" if r.underpowered else "OVER"
        band = "".join(str(t) for t in (r.case.acceptable_tiers or []))
        print(f"  {r.case.id:30s} want=tier{r.case.expected_tier} band=[{band}] got=tier{r.selected_tier} difficulty={r.decision.complexity.overall:5.2f} [{status}]")


if __name__ == "__main__":
    main()
