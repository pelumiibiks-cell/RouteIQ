# Notes to self — defending this project

Personal notes for interviews/reviews, not a polished doc. If someone asks "walk me through this," here's the honest version.

## What it is, in one breath

A router that sits in front of multiple Claude models. It looks at a prompt, estimates how hard the task actually is, and picks the cheapest model likely to handle it reliably — escalating to a stronger (and more expensive) model only when the task genuinely needs it. The whole point is the philosophy in the README: *use the weakest model that'll work, not the strongest one by default.*

## Why I built it

Every app that calls an LLM API faces this decision and mostly gets it wrong two ways: hardcode everything to the strongest model (burns money and latency on trivial requests), or hardcode to the cheapest (falls over on anything hard). I wanted to build the thing that makes that call automatically and can explain *why* — not a toy classifier, an actual pipeline with hard constraints, risk scoring, cost optimization, and an evaluation harness to prove it works.

## How to walk through the architecture out loud

Ten stages, each its own module:

1. **Normalizer** — cleans the prompt, detects code blocks/data blocks/attachments.
2. **Task analyzer** — extracts 20 categories and 19 requirement dimensions (reasoning depth, ambiguity, precision, multimodal need, etc.) from weighted signals — vocabulary density, structural cues, discourse markers. Deliberately *not* a keyword lookup: every signal is a sub-linear vote, several have to agree before a dimension scores high.
3. **Two-pass routing** — a cheap Pass 1 estimates rough difficulty; only if it's borderline between tiers does the expensive Pass 2 (full 19-dim extraction) run. This is the answer to "doesn't the router itself cost money to run?" — yes, so its own cost scales with how much the decision matters.
4. **Complexity scorer** — turns 19 dimensions into one 0-10 score. The one non-obvious design choice: it's a weighted blend of the *top 3* elevated dimensions, not an average of all ten. A pure-math task has `coding_complexity == 0`; averaging would drag its difficulty down for no reason.
5. **Candidate generator / capability matcher** — every registered model is a candidate; hard constraints (needs vision but model has none, context too long, needs tool-use but model can't) eliminate candidates outright. Survivors get soft `quality_estimate`, `overkill_risk`, `underpowered_risk` scores.
6. **Effort selector** — a *separate* decision from which model, mapped onto the real Claude API's `output_config.effort` scale (`low/medium/high/xhigh/max`), then clamped to what the chosen model actually supports (Haiku 4.5 caps at `low` because the real API doesn't take an `effort` param for it at all).
7. **Cost/latency estimator → utility ranking** — `utility = quality - cost_penalty - latency_penalty - overkill_penalty - underpowered_penalty`, all normalized against the current candidate set, not a global constant.
8. **Explanation** — every decision comes with why it won and why the alternatives lost, because a router nobody can audit isn't trustworthy.

## Design decisions I should be ready to defend

**"Why heuristic, not an actual LLM call for task analysis?"** No API access in the dev environment I built this in — I was explicit about that limitation rather than faking it. The whole analyzer sits behind a `TaskAnalyzer` interface with an `LLMBackedAnalyzer` stub already scaffolded; swapping in a real model call for Pass 1/2 requires zero changes downstream (`TaskAnalysis → ComplexityScore → routing` doesn't care where `TaskAnalysis` came from). That's the answer if someone asks "so this doesn't actually understand language" — correct, and I designed the seam so that's a config change, not a rewrite.

**"Why top-3 blend instead of a weighted average for difficulty?"** Averaging over 10 dimensions punishes tasks that are legitimately hard in one axis and irrelevant in the others (an agentic-planning task has `math_complexity = 0`; that shouldn't make it read as "easy"). I tested this — the naive average version was under-escalating pure-reasoning and pure-planning tasks specifically.

**"How do you know it works?"** A 24-case labeled evaluation dataset, six of which are adversarial by design: short-but-hard ("Prove P != NP"), long-but-trivial (150-line changelog, extract version numbers), keyword-heavy-but-easy (mentions "software architect" but the task is a one-line rename), and a vision-required prompt that looks like plain text. Current numbers: 70.8% routing accuracy, **0% overkill rate**, 88.2% cost savings vs. always using the strongest model. I lead with the 0% overkill number because that's the actual thesis of the project — never reach for a stronger model than the task needs — and it held even against the traps designed to fool a naive router.

## The honest weak points — say these before you're asked

- **Routing accuracy is 70.8%, not higher.** Every miss is "underpowered," never "overkill" — the router is systematically conservative on cost, which is the intended bias, but it means some genuinely hard prompts (an autonomous agentic system design, "Prove P != NP") land one tier below my own label. That's the heuristic analyzer's ceiling: it reasons from vocabulary and structure, not real understanding, so terse expert-level prompts it can't fully parse get under-scored. I documented this in the README rather than hiding it.
- **`quality_estimate` is a capability-gap heuristic**, not measured against real model outputs. It's derived from the numeric gap between what a task needs and what a model's registry scores claim it has. I built a "model tournament" mode specifically to close this gap later — run the same prompt against every model for real and use an LLM-as-judge or human eval to calibrate `quality_estimate` against actual outcomes.
- **The capability/cost/reliability scores in `config/models.yaml` are my own relative 0-10 estimates**, not benchmark numbers. I say this explicitly in the README and in code comments — I did not fabricate benchmark citations to make the project look more rigorous than it is.
- **The eval dataset's expected labels are my own judgment calls.** A couple of them I deliberately relabeled mid-build when the router's answer was actually more defensible than my first guess (e.g. "explain this one SQL query" doesn't need a mid-tier model — I was wrong to label it that high originally, and I say so in the dataset file itself).

## Likely interview questions

**"Walk me through what happens when I send a prompt."** → Use the 8-stage walkthrough above, but lead with a concrete example: "Convert 25 USD to EUR" lands on Haiku 4.5 at `low` effort in under a millisecond of router overhead; "find the concurrency bugs in this distributed system" escalates to Opus 5 at `medium` effort because reasoning_depth and coding_complexity both spike.

**"What would you do differently / next?"** → Swap the heuristic analyzer for the `LLMBackedAnalyzer` stub and re-run the eval to see how much accuracy that buys; run the tournament mode against real API calls to calibrate `quality_estimate`; add the learned-routing-policy layer the telemetry table (`RoutingRecord` / `/feedback`) is already shaped for, so the router improves from real outcomes instead of static heuristics.

**"Why does this matter / who would use it?"** → Anyone building on the Claude API paying per-token for every request. This is the shape of an internal model gateway a mid-size AI product team would actually build once they have more than one model in production and a cost problem.

**"What's the trickiest bug you hit building this?"** → Recalibrating the difficulty scorer. Early versions either scored *everything* as easy (a plain weighted average diluted by irrelevant-zero dimensions) or overreacted to synonym-stuffed short prompts (four near-synonymous mentions of "bug" in one sentence inflating a trivial fix's score past a genuinely complex concurrency bug). Fixed with sub-linear hit scaling (`hits^0.7`) plus the top-3 blend — worth having the before/after numbers ready if asked.

## Numbers to have memorized

- 24 labeled eval cases, 6 adversarial
- 70.8% routing accuracy / 0% overkill rate / 29.2% underpowered rate
- 88.2% cost savings vs. always-strongest-model baseline
- 34 automated tests (unit, routing, effort, adversarial, API)
- 4 real Claude models registered: Haiku 4.5, Sonnet 5, Opus 5, Fable 5
- 19 requirement dimensions, 20 task categories, 2-pass analysis
