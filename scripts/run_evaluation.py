"""Runs the full evaluation benchmark and prints the report. Run with:

    PYTHONPATH=. python scripts/run_evaluation.py
"""
from __future__ import annotations

from app.evaluation.runner import compare_to_always_strongest, run_benchmark


def main() -> None:
    report = run_benchmark()
    print("=== Router Benchmark ===")
    print(f"Total cases:          {report.total}")
    print(f"Routing accuracy:     {report.routing_accuracy * 100:.1f}%")
    print(f"Underpowered rate:    {report.underpowered_rate * 100:.1f}%")
    print(f"Overkill rate:        {report.overkill_rate * 100:.1f}%")
    print(f"Average cost:         ${report.average_cost:.6f}")
    print(f"Average latency:      {report.average_latency_ms:.0f} ms")
    print(f"Average confidence:   {report.average_confidence * 100:.1f}%")
    print()
    print("Tier confusion (expected -> got):")
    for exp, gots in sorted(report.tier_confusion.items()):
        print(f"  {exp}: {gots}")
    print()

    cost_cmp = compare_to_always_strongest()
    print("=== Cost vs. Always-Strongest-Model ===")
    print(f"Router total cost:            ${cost_cmp['router_total_cost']:.6f}")
    print(f"Always-strongest total cost:  ${cost_cmp['always_strongest_total_cost']:.6f}")
    print(f"Cost savings:                 {cost_cmp['cost_savings_pct']}%")
    print()

    print("Per-case detail:")
    for r in report.results:
        status = "OK" if r.correct else ("UNDER" if r.underpowered else "OVER")
        print(f"  {r.case.id:28s} expected=tier{r.case.expected_tier} got=tier{r.selected_tier} difficulty={r.decision.complexity.overall:5.2f} [{status}]")


if __name__ == "__main__":
    main()
