# Adaptive Model Router

An intelligent gateway that sits between an application and multiple AI models. For every prompt it decides how hard the task actually is, what capabilities it needs, which registered model is the best fit, how much reasoning effort to spend, and whether that choice is overkill or underpowered — then returns a ranked, explained recommendation.

Philosophy: **use the weakest/cheapest model likely to solve the task reliably, and escalate only when the task genuinely demands it.** Not `easy -> small model, hard -> large model` by keyword; a real (if heuristic) analysis of the task.

## What it does

`POST /route` takes a prompt (plus optional context, attachments, and cost/latency/quality constraints) and returns:

- the selected model and reasoning effort
- a difficulty score and a 10-dimension complexity breakdown
- estimated cost, latency, and quality
- overkill risk and underpowered risk for the selection
- a confidence score
- a ranked list of alternatives with rejection reasons
- a human-readable explanation

## Why model routing matters

Every provider now ships a spread of models from cheap/fast to expensive/slow-but-smart. Sending every request to the strongest model wastes money and latency on tasks that a small model would nail; sending everything to the cheapest model produces unreliable output on genuinely hard tasks. A router earns its keep by making that call automatically and explainably, per request, instead of a human hardcoding it per endpoint.

## Architecture

```
Prompt
  -> Normalizer            (app/analysis/normalizer.py)
  -> Task Analyzer          (app/analysis/task_analyzer.py)      [Pass 1: cheap, Pass 2: deep]
  -> Complexity Scorer      (app/analysis/complexity_scorer.py)
  -> Candidate Generator    (app/routing/candidate_generator.py)
  -> Capability Matcher     (app/routing/capability_matcher.py)  [hard constraints + quality/risk estimates]
  -> Effort Selector        (app/routing/effort_selector.py)
  -> Cost/Latency Estimator (app/routing/cost_latency.py)
  -> Router / Utility Rank  (app/routing/router.py)
  -> Explanation            (app/routing/explanation.py)
```

Everything downstream of the Task Analyzer only depends on a plain `TaskAnalysis`/`ComplexityScore` object, and everything upstream of the Capability Matcher only depends on `ModelProfile`. That's the seam that lets you swap the analyzer (heuristic -> LLM-backed) or the model registry (YAML -> database) without touching the rest of the pipeline.

### Package layout

```
app/
  analysis/    normalizer, task analyzer (category + 19 requirement dims), complexity scorer
  registry/    ModelProfile + ModelRegistry (loads config/models.yaml)
  routing/     candidate generation, capability matching, effort selection, cost/latency, router, explanations
  providers/   ModelProvider interface + MockProvider (no network calls, deterministic-enough for demos/CI)
  evaluation/  labeled dataset, benchmark runner, metrics, model tournament
  api/         FastAPI routes (/route, /feedback, /models, /evaluate/*)
  models/      SQLAlchemy telemetry table
  main.py, database.py, schemas.py
config/models.yaml   model registry, editable without touching code
frontend/             static dashboard (no build step), served at /dashboard
tests/                unit, routing, effort, adversarial, and API tests
scripts/               demo.py and run_evaluation.py
```

## How task difficulty is evaluated

`task_analyzer.py` classifies the prompt into one or more of 20 categories (coding, debugging, software architecture, math reasoning, agentic, multimodal, etc.) and extracts 19 requirement dimensions (`reasoning_depth`, `context_length`, `coding_complexity`, `ambiguity`, `reliability_requirement`, ...). This is **not** a keyword lookup: every "signal" (domain vocabulary, code-block/data-block presence, numbered steps, discourse markers, ambiguity/precision phrase detection, prompt length) casts a weighted, sub-linear vote, and multiple independent signals have to agree before a dimension scores high. See the adversarial tests (`tests/test_edge_cases.py`) for the cases this is meant to defend against: short-but-hard prompts, long-but-trivial prompts, keyword-heavy-but-easy prompts, and simple-looking prompts that secretly require vision.

`complexity_scorer.py` turns those 19 dimensions into one 0-10 difficulty score. It deliberately does **not** average all ten weighted dimensions — that would let irrelevant dimensions (e.g. `mathematical_complexity == 0` on a pure planning task) drag the score down. Instead it takes a weighted blend of the *top three* elevated dimensions, then applies two non-linear modifiers: a bump when ambiguity or reliability requirements are high, and a floor so a genuinely ambiguous, under-specified prompt can't be scored "easy" just because no single dimension looks technically hard.

### Two-pass routing

Pass 1 (`quick_analyze`) computes category signals and a rough difficulty estimate cheaply, without running the full 19-dimension extractor. If the rough estimate lands within 1.0 of a tier boundary (3.0 / 5.0 / 7.0), the task is "borderline" and Pass 2 (`analyze`, the full extractor with ambiguity/precision marker scanning) runs. Otherwise the router builds an approximate `TaskAnalysis` straight from the Pass 1 signals (`approximate_task_analysis`) and skips the expensive pass. This keeps the router's own overhead proportional to how much the decision actually matters — see `router.py::_is_borderline`.

## How model selection works

Every registered model is a candidate (`candidate_generator.py`). `capability_matcher.py` then applies, per (task, model) pair:

**Hard constraints (elimination, not penalty):**
- task needs vision/multimodal and the model has none
- estimated required context tokens exceed the model's context window
- task needs tool/function-calling and the model doesn't support it
- explicit request constraints (`max_cost`, `max_latency_ms`, `minimum_quality`) not met (soft — falls back to hard-capability survivors if everything gets eliminated)

**Soft signals for survivors:**
- `quality_estimate` — derived from the gap between required and available reasoning/coding/math/instruction-following capability, discounted by the model's reliability score
- `underpowered_risk` — driven by capability gaps *and* by a continuous "ideal tier" signal (difficulty 0-10 maps to an implied tier 1-4; a model below that tier accrues risk)
- `overkill_risk` — the same ideal-tier signal in reverse, plus a capability-margin check, so a frontier model applied to a trivial task is flagged even though nothing is technically "wrong" with the answer it would produce

The router then ranks survivors by a utility function:

```
utility = quality_estimate
        - 0.25 * normalized_cost
        - 0.15 * normalized_latency
        - 0.35 * overkill_risk
        - 0.45 * underpowered_risk
```

Cost and latency are normalized against the current candidate set (not a global constant), so the penalty is always relative to what's actually on the table for this task. The weights are a judgment call, not a physical law — they're isolated in `router.UTILITY_WEIGHTS` for easy tuning.

## How effort selection works

Effort (`low` / `medium` / `high` / `xhigh` / `max`) is chosen independently of the model, from a blend of overall difficulty, number of reasoning steps, output/planning complexity, reliability requirement, and ambiguity (`effort_selector.py`) — matching the real Claude API's `output_config.effort` scale rather than an invented one. It's then clamped to whatever the selected model actually supports (`ModelProfile.max_reasoning_effort`) — Claude Haiku 4.5 never gets asked for effort above `low` (the real API doesn't accept an `effort` parameter for Haiku 4.5 at all, so its ceiling reflects that it has no dial to turn). Effort defaults low; nothing gets `max` unless the difficulty blend actually earns it.

## How cost optimization works

`cost_latency.py` estimates input/output tokens from prompt size and task output-complexity, then applies the model's per-token cost and an effort multiplier (higher effort implies more output/latency). This estimate feeds the utility ranking above, and the benchmark script compares router-selected cost against an "always use the strongest model" baseline (see Results below).

## Model registry

`config/models.yaml` currently registers the four current Claude models, tier 1 (cheapest/fastest) through tier 4 (most capable):

| Model | Tier | Context | $/1K in | $/1K out | Max effort |
|---|---|---|---|---|---|
| `claude-haiku-4-5` | 1 | 200K | $0.001 | $0.005 | `low` (real API has no `effort` param for Haiku 4.5) |
| `claude-sonnet-5` | 2 | 1M | $0.003 | $0.015 | `max` |
| `claude-opus-5` | 3 | 1M | $0.005 | $0.025 | `max` |
| `claude-fable-5` | 4 | 1M | $0.01 | $0.05 | `max` |

Pricing, context windows, and effort ceilings are taken from the real Claude API. `reasoning_score`/`coding_score`/`math_score`/`instruction_following_score`/`creativity_score`/`latency_score`/`reliability_score` are my own relative 0-10 estimates, not claimed benchmark numbers — no benchmark data is faked.

## Adding a new model

Add an entry to `config/models.yaml` — no code changes:

```yaml
- name: claude-example-model
  provider: anthropic
  tier: 3
  reasoning_score: 8.2
  coding_score: 8.0
  math_score: 7.0
  instruction_following_score: 8.5
  creativity_score: 7.5
  vision: true
  audio: false
  tool_use: true
  structured_output: true
  context_window: 1000000
  latency_score: 6.0
  cost_per_input_token: 0.01
  cost_per_output_token: 0.03
  reliability_score: 9.0
  max_reasoning_effort: max
```

To actually call the model, implement `ModelProvider` (`app/providers/base.py`) for its API instead of using `MockProvider` — see the `claude-api` reference for the real Anthropic SDK call shapes.

## Running it

```bash
python -m venv .venv
.venv/Scripts/activate   # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

# API + dashboard
PYTHONPATH=. uvicorn app.main:app --reload
# -> http://localhost:8000/docs      (OpenAPI)
# -> http://localhost:8000/dashboard (dashboard)

# Demo across 20 diverse prompts
PYTHONPATH=. python scripts/demo.py

# Full evaluation benchmark
PYTHONPATH=. python scripts/run_evaluation.py

# Tests
PYTHONPATH=. pytest
```

Or with Docker: `docker compose up --build`, then open `http://localhost:8000/dashboard`.

## API

```
POST /route
{
  "prompt": "...",
  "context": "",
  "attachments": [],
  "constraints": {"max_cost": 0.05, "max_latency_ms": 5000, "minimum_quality": 0.8}
}
```

returns model, effort, confidence, difficulty, dimension scores, cost/latency estimates, overkill/underpowered risk, ranked alternatives, and a text explanation (matches `app/schemas.py::RouteResponse`).

```
POST /feedback           record actual quality/cost/latency/success against a routing decision (telemetry substrate for adaptive routing)
GET  /models              current model registry
GET  /evaluate/benchmark  run the labeled evaluation dataset, return accuracy/underpowered/overkill/cost metrics
POST /evaluate/tournament run one prompt against every registered model (mock execution) for side-by-side comparison
```

## Running evaluations

`app/evaluation/dataset.py` has 24 labeled prompts spanning easy/medium/hard/very-hard plus six adversarial cases specifically designed to defeat a naive keyword/length router (short-but-hard, long-but-trivial, keyword-heavy-but-easy, simple-looking-but-needs-vision, ambiguous, massive-context-trivial-extraction).

Latest benchmark run (`scripts/run_evaluation.py`):

```
Routing accuracy:     70.8%
Underpowered rate:    29.2%
Overkill rate:        0.0%
Average cost:         $0.002629
Cost vs. always-strongest-model: 88.2% savings
```

Every remaining "underpowered" miss is a very-hard/adversarial case (autonomous agentic system design, cascading distributed failure, "Prove P != NP", a multimodal dataset analysis) where the router landed one tier below the label. That's the honest signature of a **heuristic** Pass 1/2 analyzer: it reasons from vocabulary density and structural signals, not real semantic understanding, so it under-escalates on terse expert-level prompts it can't fully parse. The `overkill_rate` of 0% across all 24 cases (including the adversarial keyword-heavy and long-but-trivial traps) is the more important number for this project's stated goal — the router never reaches for a stronger model than a task needs.

`app/analysis/task_analyzer.py::LLMBackedAnalyzer` is a stub for exactly this gap: swap the heuristic `HeuristicTaskAnalyzer` for one backed by a cheap real model call, and the rest of the pipeline (`TaskAnalysis` -> `ComplexityScore` -> routing) needs zero changes.

## Model tournament mode

`POST /evaluate/tournament` runs the same prompt against every registered model via `MockProvider`, ranked by quality estimate then cost. This is meant to be run against real provider outputs (with an LLM-as-judge or human eval swapped in for `quality_estimate`) to validate/improve the router's own quality model over time — it's the seed of an actual evaluation platform, not just a picker.

## Evolving toward a learned routing policy

Every `/route` call writes a `RoutingRecord` (`app/models/telemetry.py`): the prompt, extracted features, estimated difficulty, selection, confidence, and cost/latency estimate. `POST /feedback` fills in actual outcome (quality, cost, latency, success, free-text feedback) against that record afterward. Nothing about the current pipeline requires this data to run — but it's the exact shape a learned policy needs: `(features, estimated_difficulty, selection) -> (actual outcome)` pairs, ready to train a classifier/regressor that could eventually replace or re-weight `effort_selector.py` and `capability_matcher.py`'s heuristics without changing the pipeline shape (`route()` would just call the learned model instead of the rule-based scorer at the same seam).

## Known limitations

- The task analyzer is rule-based (vocabulary + structural signals), not an actual LLM call — no API keys/network access in this environment. It is deliberately isolated behind `TaskAnalyzer` for exactly that reason.
- `quality_estimate` is a capability-gap heuristic, not measured against real model outputs. The tournament mode against a real provider is the path to calibrating it.
- The eval dataset's tier labels are my own judgment calls, documented inline where genuinely debatable (`app/evaluation/dataset.py` notes).
