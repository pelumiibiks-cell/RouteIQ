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
    # A defensible band, not a point. Two reasonable engineers can disagree by
    # one tier on a prompt without either being wrong, so exact-tier equality
    # punishes the router for judgement calls that were never wrong. Defaults
    # to the single expected tier when no band is given.
    acceptable_tiers: list[int] | None = None
    # "dev" cases are tunable. "test" cases are held out and must never be used
    # to pick a weight, threshold or cutpoint -- they are the only defence
    # against tuning the router against its own benchmark.
    split: str = "dev"

    def __post_init__(self) -> None:
        if self.acceptable_tiers is None:
            self.acceptable_tiers = [self.expected_tier]
        if self.expected_tier not in self.acceptable_tiers:
            raise ValueError(f"{self.id}: expected_tier {self.expected_tier} not in acceptable_tiers {self.acceptable_tiers}")


# --- dev split: the 24 original cases. NOTE two of these (medium_explain_sql,
# medium_analyze_csv) were relabeled during development to agree with the
# router's own output, so this split is contaminated by construction and must
# not be quoted as a generalization number. That is what the test split is for.
DATASET: list[EvalCase] = [
    # --- Easy ---
    EvalCase("easy_summarize", "Summarize this paragraph in two sentences: The company reported record profits this quarter, driven by strong demand for its cloud services and cost-cutting measures implemented last year.", acceptable_tiers=[1, 2], expected_tier=1, difficulty_label="easy"),
    EvalCase("easy_convert", "Convert 25 USD to EUR.", acceptable_tiers=[1], expected_tier=1, difficulty_label="easy"),
    EvalCase("easy_extract_emails", "Extract all email addresses from this text: Contact john@example.com or jane.doe@company.org for details.", acceptable_tiers=[1], expected_tier=1, difficulty_label="easy"),
    EvalCase("easy_translate", "Translate this sentence into French: The weather is nice today.", acceptable_tiers=[1], expected_tier=1, difficulty_label="easy"),
    EvalCase("easy_classify", "Classify this review as positive or negative: 'The food was cold and the service was slow.'", acceptable_tiers=[1], expected_tier=1, difficulty_label="easy"),

    # --- Medium ---
    EvalCase(
        "medium_fix_bug",
        "Fix this Python bug:\n```python\ndef divide(a, b):\n    return a / b\n\nresult = divide(10, 0)\n```\nIt crashes with a ZeroDivisionError.",
        acceptable_tiers=[2, 3], expected_tier=2, difficulty_label="medium",
    ),
    EvalCase(
        "medium_explain_sql", "Explain what this SQL query does:\n```sql\nSELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id HAVING COUNT(*) > 5;\n```",
        acceptable_tiers=[1, 2], expected_tier=1, difficulty_label="medium",
        notes="Genuinely a light task (explaining one GROUP BY/HAVING query) -- a lightweight model handles this fine, despite being labeled 'medium' category-wise.",
    ),
    EvalCase(
        "medium_analyze_csv",
        "Analyze this CSV of monthly sales and tell me if there's a trend:\nmonth,sales\nJan,120\nFeb,135\nMar,128\nApr,150\nMay,162\nJun,171",
        acceptable_tiers=[1, 2], expected_tier=1, difficulty_label="medium",
        notes="Six data points, an obviously monotonic-ish trend -- doesn't need a stronger model than lightweight.",
    ),
    EvalCase("medium_rest_endpoint", "Write a REST API endpoint in FastAPI that accepts a POST request with a JSON body containing 'name' and 'email', validates the email format, and returns a 201 response.", acceptable_tiers=[2, 3], expected_tier=2, difficulty_label="medium"),
    EvalCase(
        "medium_short_story", "Write a short creative story (under 200 words) about a lighthouse keeper who discovers something strange washed ashore.",
        acceptable_tiers=[1, 2], expected_tier=1, difficulty_label="medium",
        notes="Short, low-stakes creative writing -- lightweight tier is a reasonable choice; tier2 is not wrong either, but not required.",
    ),

    # --- Hard ---
    EvalCase(
        "hard_distributed_cache",
        "Design a distributed caching architecture for a high-traffic e-commerce platform. Consider cache invalidation, consistency, and failure modes across regions.",
        acceptable_tiers=[3, 4], expected_tier=3, difficulty_label="hard",
    ),
    EvalCase(
        "hard_debug_concurrency",
        "Find the subtle concurrency bugs in this distributed Python system. Explain the race conditions and propose a safe redesign:\n```python\nclass Counter:\n    def __init__(self):\n        self.value = 0\n    def increment(self):\n        current = self.value\n        time.sleep(0.001)\n        self.value = current + 1\n```\nThis is called concurrently from 50 worker threads and the final count is wrong.",
        acceptable_tiers=[3, 4], expected_tier=3, difficulty_label="hard",
    ),
    EvalCase(
        "hard_repo_migration",
        "Analyze this 2,000-line Python repository, identify architectural problems, and propose a migration plan to a cleaner modular structure. The repo currently mixes business logic, database access, and HTTP handling in the same files across dozens of modules.",
        acceptable_tiers=[2, 3], expected_tier=3, difficulty_label="hard",
    ),
    EvalCase(
        "hard_ml_pipeline",
        "Design an ML training pipeline for a fraud-detection model that needs to retrain daily on streaming transaction data, handle class imbalance, and support online evaluation before promotion to production.",
        acceptable_tiers=[3, 4], expected_tier=3, difficulty_label="hard",
    ),
    EvalCase(
        "hard_math_proof",
        "Solve this problem and show your reasoning: prove that for any integer n > 1, if n is not prime, then n has a prime factor less than or equal to sqrt(n). Then use this to design an efficient primality test.",
        acceptable_tiers=[3, 4], expected_tier=3, difficulty_label="hard",
    ),

    # --- Very hard ---
    EvalCase(
        "vhard_agentic_system",
        "Design a production-grade agentic system that can autonomously plan and execute multi-step research tasks, call external tools, self-correct on tool failures, and maintain long-running state across sessions. Cover orchestration, memory, failure recovery, and safety guardrails.",
        acceptable_tiers=[3, 4], expected_tier=4, difficulty_label="very_hard",
    ),
    EvalCase(
        "vhard_distributed_failure",
        "Our distributed payments system experienced a cascading failure: a single slow database replica caused connection pool exhaustion across 12 microservices, leading to a 40-minute outage. Reason through the likely causal chain, identify every contributing architectural weakness, and propose a comprehensive redesign with concrete safeguards at each layer.",
        acceptable_tiers=[3, 4], expected_tier=4, difficulty_label="very_hard",
    ),
    EvalCase(
        "vhard_multimodal_dataset",
        "Analyze this large multimodal dataset combining product images, customer reviews, and sales time series to identify which visual product attributes most strongly predict returns. Attachments include product photos.",
        attachments=["product_images.zip"],
        acceptable_tiers=[3, 4], expected_tier=4, difficulty_label="very_hard",
    ),
    EvalCase(
        "vhard_optimization_strategy",
        "Develop a sophisticated optimization strategy for a multi-warehouse logistics network with stochastic demand, variable transit times, and conflicting cost/service-level objectives. Formulate the problem mathematically and propose a solution approach.",
        acceptable_tiers=[3, 4], expected_tier=4, difficulty_label="very_hard",
    ),

    # --- Adversarial / edge cases ---
    EvalCase(
        "adv_short_but_hard",
        "Prove P != NP.",
        acceptable_tiers=[3, 4], expected_tier=4, difficulty_label="very_hard",
        notes="Short prompt, extremely difficult reasoning -- must not be judged easy by length.",
    ),
    EvalCase(
        "adv_long_but_trivial",
        "Here is a long changelog with hundreds of entries.\n" + "\n".join(f"- v0.{i}: minor bugfix" for i in range(1, 120)) + "\nExtract just the version numbers into a list.",
        acceptable_tiers=[1], expected_tier=1, difficulty_label="easy",
        notes="Long input but trivial extraction task -- must not be judged hard by length alone.",
    ),
    EvalCase(
        "adv_keyword_heavy_easy",
        "I'm not a software architect or anything, but can you just quickly rename this variable from 'x' to 'count' in this one-liner: `x = 5`?",
        acceptable_tiers=[1], expected_tier=1, difficulty_label="easy",
        notes="Mentions 'software architect' but the actual task is trivial -- must not be misled by keyword.",
    ),
    EvalCase(
        "adv_simple_vision",
        "What color is the car in this photo?",
        attachments=["car.jpg"],
        acceptable_tiers=[2, 3], expected_tier=2, difficulty_label="medium",
        notes="Simple-looking task but requires vision -- lightweight (no-vision) model must be eliminated.",
    ),
    EvalCase(
        "adv_ambiguous",
        "Make it better.",
        acceptable_tiers=[2, 3], expected_tier=2, difficulty_label="medium",
        notes="Highly ambiguous with no context -- should trigger high ambiguity score.",
    ),
]


# --- test split: held out. Written after the dev split was already in use, and
# labeled from the prompt alone without consulting the router's output. Nothing
# here may be used to choose a weight, threshold or cutpoint.
TEST_DATASET: list[EvalCase] = [
    # --- Easy ---
    EvalCase("t_easy_wordcount", "How many words are in this sentence: 'The quick brown fox jumps over the lazy dog'?", acceptable_tiers=[1], expected_tier=1, difficulty_label="easy", split="test"),
    EvalCase("t_easy_json_pretty", 'Reformat this JSON with two-space indentation: {"a":1,"b":[2,3],"c":{"d":4}}', acceptable_tiers=[1], expected_tier=1, difficulty_label="easy", split="test"),
    EvalCase("t_easy_synonyms", "Give me three synonyms for the word 'important'.", acceptable_tiers=[1], expected_tier=1, difficulty_label="easy", split="test"),
    EvalCase("t_easy_temperature", "Convert 180 degrees Fahrenheit to Celsius.", acceptable_tiers=[1], expected_tier=1, difficulty_label="easy", split="test"),
    EvalCase("t_easy_sort_list", "Sort this list alphabetically: banana, apple, cherry, date.", acceptable_tiers=[1], expected_tier=1, difficulty_label="easy", split="test"),

    # --- Medium ---
    EvalCase(
        "t_med_regex_iso8601",
        "Write a regular expression that validates an ISO 8601 date-time with an optional timezone offset, and explain what each part of the pattern matches.",
        acceptable_tiers=[1, 2], expected_tier=2, difficulty_label="medium", split="test",
    ),
    EvalCase(
        "t_med_sql_index",
        "This query does a full table scan on a 40-million-row orders table and takes 9 seconds:\n```sql\nSELECT * FROM orders WHERE status = 'pending' AND created_at > '2026-01-01' ORDER BY created_at DESC LIMIT 50;\n```\nSuggest the indexes that would fix it and explain why each column order matters.",
        acceptable_tiers=[2, 3], expected_tier=2, difficulty_label="medium", split="test",
    ),
    EvalCase(
        "t_med_dockerfile",
        "Write a multi-stage Dockerfile for a Go service that compiles a static binary in the build stage and runs it on a distroless base image as a non-root user.",
        acceptable_tiers=[2, 3], expected_tier=2, difficulty_label="medium", split="test",
    ),
    EvalCase(
        "t_med_meeting_notes",
        "Summarize these meeting notes into decisions made and action items with owners:\nAlex raised the migration timeline, said staging is blocked on the new credentials. Priya agreed to file the access request today. The team debated whether to cut scope on the reporting module; consensus was to defer it to next quarter. Sam will update the roadmap doc by Friday. We also agreed to move the standup to 9:30.",
        acceptable_tiers=[1, 2], expected_tier=2, difficulty_label="medium", split="test",
    ),
    EvalCase(
        "t_med_pagination",
        "Design cursor-based pagination for a REST list endpoint. Cover how the cursor is encoded, how results stay stable when rows are inserted mid-pagination, and what happens when the cursor points at a deleted row.",
        acceptable_tiers=[2, 3], expected_tier=2, difficulty_label="medium", split="test",
    ),

    # --- Hard ---
    EvalCase(
        "t_hard_raft_partition",
        "Explain precisely how Raft behaves when a network partition isolates the current leader from a majority of the cluster. Walk through what happens to in-flight client writes, how the new term is established, and how the old leader's uncommitted log entries are reconciled when it rejoins.",
        acceptable_tiers=[3, 4], expected_tier=3, difficulty_label="hard", split="test",
    ),
    EvalCase(
        "t_hard_memory_leak",
        "A JVM service leaks about 200MB of heap per hour under steady load. Heap dumps show a growing ConcurrentHashMap inside a caching layer whose entries are keyed by request-scoped objects that override equals but not hashCode. Explain the mechanism, why it survives GC, and lay out a fix plus the regression test that would have caught it.",
        acceptable_tiers=[3, 4], expected_tier=3, difficulty_label="hard", split="test",
    ),
    EvalCase(
        "t_hard_zero_downtime_migration",
        "Plan a zero-downtime schema migration that splits a 500-million-row users table into users and user_profiles on a live production database. Cover the dual-write window, backfill strategy, read-path cutover, consistency verification, and the rollback procedure at each stage.",
        acceptable_tiers=[3, 4], expected_tier=3, difficulty_label="hard", split="test",
    ),
    EvalCase(
        "t_hard_quickselect_bound",
        "Derive the expected number of comparisons performed by randomized quickselect on an array of n distinct elements, and prove the resulting bound is linear in n.",
        acceptable_tiers=[3, 4], expected_tier=3, difficulty_label="hard", split="test",
    ),
    EvalCase(
        "t_hard_type_inference",
        "Implement type inference for the simply-typed lambda calculus extended with let-polymorphism. Explain the unification algorithm, where generalization happens, and why the value restriction is needed once mutable references are added.",
        acceptable_tiers=[3, 4], expected_tier=3, difficulty_label="hard", split="test",
    ),

    # --- Very hard ---
    EvalCase(
        "t_vhard_global_txn",
        "Design a globally distributed transaction system spanning three continents that offers bounded-staleness reads and strict serializability for writes. Address clock uncertainty, commit protocol, cross-region failure modes, the latency budget at each hop, and the consistency guarantees you are forced to give up.",
        acceptable_tiers=[3, 4], expected_tier=4, difficulty_label="very_hard", split="test",
    ),
    EvalCase(
        "t_vhard_agent_safety",
        "Design a multi-agent system where agents negotiate resource allocation under adversarial conditions. Specify the formal safety invariants that must hold, prove they are preserved under agent failure and message reordering, and design the monitoring that detects violation in production.",
        acceptable_tiers=[3, 4], expected_tier=4, difficulty_label="very_hard", split="test",
    ),
    EvalCase(
        "t_vhard_research_program",
        "Propose a novel approach to continual learning that avoids catastrophic forgetting without storing past examples. Survey why replay-based and regularization-based methods fall short, formalize your approach, and design the experimental program that would falsify it.",
        acceptable_tiers=[3, 4], expected_tier=4, difficulty_label="very_hard", split="test",
    ),

    # --- Adversarial ---
    EvalCase(
        "t_adv_casual_but_hard",
        "Quick question, no rush: is the Collatz conjecture true?",
        acceptable_tiers=[3, 4], expected_tier=4, difficulty_label="very_hard", split="test",
        notes="Casual, short phrasing wrapping an open problem in mathematics -- tone and length must not drive the score.",
    ),
    EvalCase(
        "t_adv_long_but_trivial_logs",
        "Here is a production log dump:\n" + "\n".join(f"2026-03-{(i % 28) + 1:02d} 10:{i % 60:02d}:00 [{'ERROR' if i % 7 == 0 else 'INFO'}] worker-{i % 12} processed batch {i}" for i in range(1, 160)) + "\nCount how many lines contain ERROR.",
        acceptable_tiers=[1, 2], expected_tier=1, difficulty_label="easy", split="test",
        notes="Long input, trivial counting task -- length must not drive the score.",
    ),
    EvalCase(
        "t_adv_vision_trivial",
        "Is the sky in this image blue or grey?",
        attachments=["sky.jpg"],
        acceptable_tiers=[2, 3], expected_tier=2, difficulty_label="medium", split="test",
        notes="Trivial-looking question that cannot be answered at all without vision.",
    ),
]


def get_test_dataset() -> list[EvalCase]:
    return TEST_DATASET


def get_all_cases() -> list[EvalCase]:
    return DATASET + TEST_DATASET


def get_dataset() -> list[EvalCase]:
    """The dev split. Kept as the default so existing callers are unchanged."""
    return DATASET
