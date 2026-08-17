"""Runs the router against a diverse set of prompts and prints a table of
Prompt / Difficulty / Model / Effort / Confidence / Cost / Why -- the
end-to-end demonstration required by the project brief. Run with:

    PYTHONPATH=. python scripts/demo.py
"""
from __future__ import annotations

from app.routing.router import route

PROMPTS = [
    "Convert 25 USD to EUR.",
    "Translate this sentence into French: The weather is nice today.",
    "Extract all email addresses from: Contact john@example.com or jane@company.org.",
    "Summarize this paragraph in two sentences: The company reported record profits this quarter, driven by cloud demand and cost cuts.",
    "Classify this review as positive or negative: 'The food was cold and the service was slow.'",
    "Fix this Python bug:\n```python\ndef divide(a, b):\n    return a / b\nresult = divide(10, 0)\n```\nIt crashes with a ZeroDivisionError.",
    "Write a REST API endpoint in FastAPI that accepts a POST with name/email, validates the email, and returns 201.",
    "Explain what this SQL query does:\n```sql\nSELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id HAVING COUNT(*) > 5;\n```",
    "Design a distributed caching architecture for a high-traffic e-commerce platform, covering cache invalidation and consistency across regions.",
    "Find the subtle concurrency bugs in this distributed Python system. Explain the race conditions and propose a safe redesign:\n```python\nclass Counter:\n    def increment(self):\n        current = self.value\n        time.sleep(0.001)\n        self.value = current + 1\n```",
    "Analyze this 2,000-line Python repository, identify architectural problems, and propose a migration plan to a cleaner modular structure.",
    "Prove that for any integer n > 1, if n is not prime, it has a prime factor <= sqrt(n). Show your reasoning, then design an efficient primality test.",
    "Design a production-grade agentic system that can autonomously plan and execute multi-step research tasks, call tools, self-correct on failures, and maintain long-running state.",
    "Our distributed payments system had a cascading failure: a slow database replica caused connection pool exhaustion across 12 microservices, leading to a 40-minute outage. Reason through the causal chain and propose a comprehensive redesign.",
    "Prove P != NP.",
    "Make it better.",
    "Here is a long changelog with 150 entries.\n" + "\n".join(f"- v0.{i}: minor bugfix" for i in range(1, 30)) + "\nExtract just the version numbers into a list.",
    "I'm not a software architect, but can you rename this variable from 'x' to 'count': `x = 5`?",
    "Develop a sophisticated optimization strategy for a multi-warehouse logistics network with stochastic demand and conflicting cost/service-level objectives. Formulate the problem mathematically.",
]

VISION_PROMPT = ("What color is the car in this photo?", ["car.jpg"])


def main() -> None:
    print(f"{'PROMPT':60s} {'DIFF':>5s} {'MODEL':24s} {'EFFORT':8s} {'CONF':>5s} {'COST':>10s}  WHY")
    print("-" * 150)
    all_prompts = PROMPTS + [VISION_PROMPT[0]]
    for p in all_prompts:
        attachments = VISION_PROMPT[1] if p == VISION_PROMPT[0] else []
        d = route(p, attachments=attachments)
        short = (p[:57] + "...") if len(p) > 60 else p
        short = short.replace("\n", " ")
        why = "; ".join(d.positive_reasons[:2]) if d.positive_reasons else "low complexity across all dimensions"
        print(
            f"{short:60s} {d.complexity.overall:5.1f} {d.selected.model.name:24s} "
            f"{d.selected.effort:8s} {d.confidence:5.2f} ${d.selected.cost_latency.cost_usd:<9.5f} {why}"
        )

    print()
    tiers = [route(p, attachments=(VISION_PROMPT[1] if p == VISION_PROMPT[0] else [])).selected.model.tier for p in all_prompts]
    tier1_or_2 = sum(1 for t in tiers if t <= 2)
    tier3_or_4 = len(all_prompts) - tier1_or_2
    print(f"Routed to lightweight/balanced (tier 1-2): {tier1_or_2}/{len(all_prompts)}")
    print(f"Escalated to advanced/frontier (tier 3-4): {tier3_or_4}/{len(all_prompts)}")


if __name__ == "__main__":
    main()
