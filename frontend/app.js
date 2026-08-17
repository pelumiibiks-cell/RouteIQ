const API_BASE = ""; // same origin (dashboard is mounted by the FastAPI app)

const els = {
  prompt: document.getElementById("prompt"),
  maxCost: document.getElementById("maxCost"),
  maxLatency: document.getElementById("maxLatency"),
  minQuality: document.getElementById("minQuality"),
  routeBtn: document.getElementById("routeBtn"),
  status: document.getElementById("status"),
  resultPanel: document.getElementById("resultPanel"),
  difficultyValue: document.getElementById("difficultyValue"),
  twoPassBadge: document.getElementById("twoPassBadge"),
  bars: document.getElementById("bars"),
  modelValue: document.getElementById("modelValue"),
  effortValue: document.getElementById("effortValue"),
  confidenceValue: document.getElementById("confidenceValue"),
  costValue: document.getElementById("costValue"),
  latencyValue: document.getElementById("latencyValue"),
  why: document.getElementById("why"),
  candidatesBody: document.querySelector("#candidatesTable tbody"),
  modelsBody: document.querySelector("#modelsTable tbody"),
};

async function loadModels() {
  const res = await fetch(`${API_BASE}/models`);
  const models = await res.json();
  els.modelsBody.innerHTML = models
    .sort((a, b) => a.tier - b.tier)
    .map(
      (m) => `<tr>
        <td>${m.name}</td><td>${m.tier}</td><td>${m.reasoning_score}</td>
        <td>${m.coding_score}</td><td>${m.math_score}</td><td>${m.vision ? "yes" : "no"}</td>
        <td>${m.context_window.toLocaleString()}</td>
        <td>$${m.cost_per_input_token}</td><td>$${m.cost_per_output_token}</td>
      </tr>`
    )
    .join("");
}

function renderBars(dims) {
  const labels = {
    reasoning: "Reasoning",
    context: "Context",
    coding_complexity: "Coding",
    math_complexity: "Math",
    planning_complexity: "Planning",
    tool_agent_complexity: "Tool/Agent",
    multimodal_complexity: "Multimodal",
    precision_requirement: "Precision",
    ambiguity: "Ambiguity",
    reliability_requirement: "Reliability",
    domain_specialization: "Domain",
    research_requirement: "Research",
  };
  els.bars.innerHTML = Object.entries(dims)
    .filter(([k]) => labels[k])
    .map(([k, v]) => {
      const pct = Math.min(100, (v / 10) * 100);
      return `<div class="bar-row">
        <span>${labels[k]}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
        <span>${v.toFixed(1)}</span>
      </div>`;
    })
    .join("");
}

function renderWhy(data) {
  const positives = data.positive_reasons.map((r) => `<li class="positive">✓ ${r}</li>`).join("");
  const negatives = data.negative_reasons.map((r) => `<li class="negative">✗ ${r}</li>`).join("");
  const rejected = data.rejected_alternatives.map((r) => `<li class="rejected">${r}</li>`).join("");
  els.why.innerHTML = `
    <p>${data.explanation}</p>
    <ul>${positives}${negatives}</ul>
    ${rejected ? `<h2>Rejected Alternatives</h2><ul>${rejected}</ul>` : ""}
  `;
}

function renderCandidates(data) {
  const rows = [
    {
      model: data.model,
      effort: data.effort,
      utility: null,
      quality_estimate: data.quality_estimate,
      overkill_risk: data.overkill_risk,
      underpowered_risk: data.underpowered_risk,
      estimated_cost: data.estimated_cost,
      estimated_latency_ms: data.estimated_latency_ms,
      selected: true,
    },
    ...data.alternatives.map((a) => ({ ...a, selected: false })),
  ];
  els.candidatesBody.innerHTML = rows
    .map(
      (r) => `<tr class="${r.selected ? "selected-row" : ""}">
        <td>${r.model}${r.selected ? " (selected)" : ""}</td>
        <td>${r.effort}</td>
        <td>${r.utility !== null && r.utility !== undefined ? r.utility.toFixed(3) : "-"}</td>
        <td>${(r.quality_estimate * 100).toFixed(0)}%</td>
        <td>${(r.overkill_risk * 100).toFixed(0)}%</td>
        <td>${(r.underpowered_risk * 100).toFixed(0)}%</td>
        <td>$${r.estimated_cost}</td>
        <td>${r.estimated_latency_ms}</td>
      </tr>`
    )
    .join("");
}

async function routePrompt() {
  const prompt = els.prompt.value.trim();
  if (!prompt) return;

  const constraints = {};
  if (els.maxCost.value) constraints.max_cost = parseFloat(els.maxCost.value);
  if (els.maxLatency.value) constraints.max_latency_ms = parseFloat(els.maxLatency.value);
  if (els.minQuality.value) constraints.minimum_quality = parseFloat(els.minQuality.value);

  els.status.textContent = "Routing...";
  els.routeBtn.disabled = true;
  try {
    const res = await fetch(`${API_BASE}/route`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        constraints: Object.keys(constraints).length ? constraints : null,
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    els.resultPanel.hidden = false;
    els.difficultyValue.textContent = data.difficulty.toFixed(1);
    els.twoPassBadge.hidden = !data.two_pass_used;
    renderBars(data.dimension_scores);

    els.modelValue.textContent = data.model;
    els.effortValue.textContent = data.effort;
    els.confidenceValue.textContent = `${Math.round(data.confidence * 100)}%`;
    els.costValue.textContent = `$${data.estimated_cost}`;
    els.latencyValue.textContent = `${data.estimated_latency_ms} ms`;

    renderWhy(data);
    renderCandidates(data);

    els.status.textContent = "";
  } catch (err) {
    els.status.textContent = `Error: ${err.message}`;
  } finally {
    els.routeBtn.disabled = false;
  }
}

els.routeBtn.addEventListener("click", routePrompt);
loadModels();
