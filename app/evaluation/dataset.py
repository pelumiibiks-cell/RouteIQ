"""Evaluation dataset: prompts labeled with an expected model tier so the
router's choices can be scored against ground truth. Tiers:
  1 = Claude Haiku 4.5, 2 = Claude Sonnet 5, 3 = Claude Opus 5, 4 = Claude Fable 5
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalCase:
    id: str
    prompt: str
    context: str = ""
    attachments: list[str] | None = None
    expected_tier: int = 1
    difficulty_label: str = "easy"
    notes: str = ""


DATASET: list[EvalCase] = [
    # --- Easy ---
    EvalCase("easy_summarize", "Summarize this paragraph in two sentences: The company reported record profits this quarter, driven by strong demand for its cloud services and cost-cutting measures implemented last year.", expected_tier=1, difficulty_label="easy"),
    EvalCase("easy_convert", "Convert 25 USD to EUR.", expected_tier=1, difficulty_label="easy"),
    EvalCase("easy_extract_emails", "Extract all email addresses from this text: Contact john@example.com or jane.doe@company.org for details.", expected_tier=1, difficulty_label="easy"),
    EvalCase("easy_translate", "Translate this sentence into French: The weather is nice today.", expected_tier=1, difficulty_label="easy"),
    EvalCase("easy_classify", "Classify this review as positive or negative: 'The food was cold and the service was slow.'", expected_tier=1, difficulty_label="easy"),

    # --- Medium ---
    EvalCase(
        "medium_fix_bug",
        "Fix this Python bug:\n```python\ndef divide(a, b):\n    return a / b\n\nresult = divide(10, 0)\n```\nIt crashes with a ZeroDivisionError.",
        expected_tier=2, difficulty_label="medium",
    ),
    EvalCase(
        "medium_explain_sql", "Explain what this SQL query does:\n```sql\nSELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id HAVING COUNT(*) > 5;\n```",
        expected_tier=1, difficulty_label="medium",
        notes="Genuinely a light task (explaining one GROUP BY/HAVING query) -- a lightweight model handles this fine, despite being labeled 'medium' category-wise.",
    ),
    EvalCase(
        "medium_analyze_csv",
        "Analyze this CSV of monthly sales and tell me if there's a trend:\nmonth,sales\nJan,120\nFeb,135\nMar,128\nApr,150\nMay,162\nJun,171",
        expected_tier=1, difficulty_label="medium",
        notes="Six data points, an obviously monotonic-ish trend -- doesn't need a stronger model than lightweight.",
    ),
    EvalCase("medium_rest_endpoint", "Write a REST API endpoint in FastAPI that accepts a POST request with a JSON body containing 'name' and 'email', validates the email format, and returns a 201 response.", expected_tier=2, difficulty_label="medium"),
    EvalCase(
        "medium_short_story", "Write a short creative story (under 200 words) about a lighthouse keeper who discovers something strange washed ashore.",
        expected_tier=1, difficulty_label="medium",
        notes="Short, low-stakes creative writing -- lightweight tier is a reasonable choice; tier2 is not wrong either, but not required.",
    ),

    # --- Hard ---
    EvalCase(
        "hard_distributed_cache",
        "Design a distributed caching architecture for a high-traffic e-commerce platform. Consider cache invalidation, consistency, and failure modes across regions.",
        expected_tier=3, difficulty_label="hard",
    ),
    EvalCase(
        "hard_debug_concurrency",
        "Find the subtle concurrency bugs in this distributed Python system. Explain the race conditions and propose a safe redesign:\n```python\nclass Counter:\n    def __init__(self):\n        self.value = 0\n    def increment(self):\n        current = self.value\n        time.sleep(0.001)\n        self.value = current + 1\n```\nThis is called concurrently from 50 worker threads and the final count is wrong.",
        expected_tier=3, difficulty_label="hard",
    ),
    EvalCase(
        "hard_repo_migration",
        "Analyze this 2,000-line Python repository, identify architectural problems, and propose a migration plan to a cleaner modular structure. The repo currently mixes business logic, database access, and HTTP handling in the same files across dozens of modules.",
        expected_tier=3, difficulty_label="hard",
    ),
    EvalCase(
        "hard_ml_pipeline",
        "Design an ML training pipeline for a fraud-detection model that needs to retrain daily on streaming transaction data, handle class imbalance, and support online evaluation before promotion to production.",
        expected_tier=3, difficulty_label="hard",
    ),
    EvalCase(
        "hard_math_proof",
        "Solve this problem and show your reasoning: prove that for any integer n > 1, if n is not prime, then n has a prime factor less than or equal to sqrt(n). Then use this to design an efficient primality test.",
        expected_tier=3, difficulty_label="hard",
    ),

    # --- Very hard ---
    EvalCase(
        "vhard_agentic_system",
        "Design a production-grade agentic system that can autonomously plan and execute multi-step research tasks, call external tools, self-correct on tool failures, and maintain long-running state across sessions. Cover orchestration, memory, failure recovery, and safety guardrails.",
        expected_tier=4, difficulty_label="very_hard",
    ),
    EvalCase(
        "vhard_distributed_failure",
        "Our distributed payments system experienced a cascading failure: a single slow database replica caused connection pool exhaustion across 12 microservices, leading to a 40-minute outage. Reason through the likely causal chain, identify every contributing architectural weakness, and propose a comprehensive redesign with concrete safeguards at each layer.",
        expected_tier=4, difficulty_label="very_hard",
    ),
    EvalCase(
        "vhard_multimodal_dataset",
        "Analyze this large multimodal dataset combining product images, customer reviews, and sales time series to identify which visual product attributes most strongly predict returns. Attachments include product photos.",
        attachments=["product_images.zip"],
        expected_tier=4, difficulty_label="very_hard",
    ),
    EvalCase(
        "vhard_optimization_strategy",
        "Develop a sophisticated optimization strategy for a multi-warehouse logistics network with stochastic demand, variable transit times, and conflicting cost/service-level objectives. Formulate the problem mathematically and propose a solution approach.",
        expected_tier=4, difficulty_label="very_hard",
    ),

    # --- Adversarial / edge cases ---
    EvalCase(
        "adv_short_but_hard",
        "Prove P != NP.",
        expected_tier=4, difficulty_label="very_hard",
        notes="Short prompt, extremely difficult reasoning -- must not be judged easy by length.",
    ),
    EvalCase(
        "adv_long_but_trivial",
        "Here is a long changelog with hundreds of entries.\n" + "\n".join(f"- v0.{i}: minor bugfix" for i in range(1, 120)) + "\nExtract just the version numbers into a list.",
        expected_tier=1, difficulty_label="easy",
        notes="Long input but trivial extraction task -- must not be judged hard by length alone.",
    ),
    EvalCase(
        "adv_keyword_heavy_easy",
        "I'm not a software architect or anything, but can you just quickly rename this variable from 'x' to 'count' in this one-liner: `x = 5`?",
        expected_tier=1, difficulty_label="easy",
        notes="Mentions 'software architect' but the actual task is trivial -- must not be misled by keyword.",
    ),
    EvalCase(
        "adv_simple_vision",
        "What color is the car in this photo?",
        attachments=["car.jpg"],
        expected_tier=2, difficulty_label="medium",
        notes="Simple-looking task but requires vision -- lightweight (no-vision) model must be eliminated.",
    ),
    EvalCase(
        "adv_ambiguous",
        "Make it better.",
        expected_tier=2, difficulty_label="medium",
        notes="Highly ambiguous with no context -- should trigger high ambiguity score.",
    ),
]


def get_dataset() -> list[EvalCase]:
    return DATASET
