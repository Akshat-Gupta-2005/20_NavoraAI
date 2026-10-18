# Navora AI — Project Notes

Extended technical notes for Navora AI: implementation status, engineering case studies, interview preparation, and concept references.

For user-facing documentation — setup, architecture, API reference — see the [README](../README.md).

> **Audience:** this document is written for the project author. It contains candid self-assessment, debugging post-mortems, and interview preparation material. It is intentionally more granular and more self-critical than the README.

---

## Table of Contents

1. [Implementation Status](#1-implementation-status)
2. [Production Readiness Analysis](#2-production-readiness-analysis)
3. [Engineering Case Studies](#3-engineering-case-studies)
4. [Interview Preparation](#4-interview-preparation)
5. [Resume Content](#5-resume-content)
6. [Concept Reference](#6-concept-reference)

---

## 1. Implementation Status

An honest build-versus-planned matrix. Keep this current — it is the ground truth for what you can claim.

| Area | Status | Notes |
|---|---|---|
| Agentic reasoning loop (ReAct) | ✅ Built | LangGraph cyclic state machine |
| Tool calling (10 tools) | ✅ Built | Weather, places, budget, FX |
| Real-time data retrieval | ✅ Built | 4 external APIs |
| Fallback / graceful degradation | ✅ Built | Google Places → Tavily, both failure modes |
| LLM gateway / provider routing | ✅ Built | Groq ⇄ OpenAI via YAML |
| REST API backend | ✅ Built | FastAPI + Pydantic validation |
| Web client | ✅ Built | Streamlit chat UI |
| Config management | ✅ Built | YAML + `.env`, path-independent |
| Version control | ✅ Built | Git initialised; `.env` correctly gitignored |
| Markdown export | ⚠️ Partial | `save_document()` written but not wired to the endpoint |
| Automated tests | ❌ Planned | No pytest suite yet |
| Database / persistence | ❌ Planned | Stateless; no conversation memory |
| Containerisation (Docker) | ❌ Planned | Dockerfile designed, not written |
| CI/CD pipeline | ❌ Planned | No GitHub Actions workflow yet |
| Observability / monitoring | ❌ Planned | `print()` only; `logger/` module empty |
| LLM evaluation harness | ❌ Planned | No golden dataset or judge |
| Auth / rate limiting | ❌ Planned | CORS currently permissive |
| Caching | ❌ Planned | Every request re-hits metered APIs |

**Legend:** ✅ Built and verified · ⚠️ Partially built · ❌ Designed but not implemented

### What Was Actually Verified

| Check | Result |
|---|---|
| All third-party imports resolve on LangChain 1.x | Pass — 8/8 |
| Graph compiles with all 10 tools bound | Pass — `['__start__','agent','tools','__end__']` |
| Config loads with CWD set outside the repo | Pass |
| `multiply("100", 3) == 300.0` (not `"100100100"`) | Pass |
| Missing `GROQ_API_KEY` raises a named `ValueError` | Pass |
| 3 requests → 0 graph rebuilds | Pass |

---

## 2. Production Readiness Analysis

### 2.1 Database / Persistence

**Current:** none. Fully stateless. No checkpointer is passed to `.compile()`, so conversation memory does not survive a request — every query starts cold.

| Need | Solution |
|---|---|
| Conversation memory across turns | `PostgresSaver` passed to `.compile(checkpointer=...)`, keyed by `thread_id` |
| Saved itineraries | PostgreSQL — `users`, `trips`, `itineraries` tables |
| API response cache | Redis with per-source TTL |
| Semantic cache on similar questions | pgvector / Qdrant over question embeddings |

```python
# One-line change to gain durable multi-turn memory:
self.graph = graph_builder.compile(checkpointer=PostgresSaver(conn))
```

### 2.2 Caching

**Current:** none. Every request re-hits every metered API. Two users asking about the same destination on the same day pay twice for identical data.

| Layer | TTL | Rationale |
|---|---|---|
| Weather | 1 hour | Changes slowly |
| Places | 7 days | Attractions rarely change |
| FX rates | 1 hour | Sufficient for trip budgeting |
| Semantic LLM cache | — | Embed the question; serve cached answers above a similarity threshold |

### 2.3 CI/CD

**Current:** git is initialised with an initial commit. No CI workflow yet.

| Stage | Tool |
|---|---|
| Lint | `ruff` |
| Format check | `black --check` |
| Type check | `mypy` |
| Tests | `pytest --cov` (fail under 80%) |
| Secret scanning | `gitleaks` |
| Dependency audit | `pip-audit` |
| Build image | `docker build` |
| Deploy | On `main` merge → staging → manual gate → production |

### 2.4 Observability

**Current:** `logger/logging.py` and `exception/exceptiohandling.py` are empty files. The application uses bare `print()`.

| Pillar | Tool | Signal |
|---|---|---|
| Logging | `structlog` → JSON | Structured logs with a `request_id` correlating every tool call in one agent run |
| Tracing | LangSmith or OpenTelemetry | Full agent trace: every LLM call, prompt, tool invocation, token count, latency |
| Metrics | Prometheus + Grafana | p50/p95/p99 latency, tool error rate, tokens/request, cost/request, fallback rate |
| Alerting | Alertmanager | Error rate > 5%, p95 > 120s, daily spend over budget |

**LLM-specific metrics that generic APM misses:**

- Tokens per request (input/output split) and cost per request
- Tool-call success rate, per tool
- **Fallback rate** — how often Google Places fails into Tavily; a rising rate signals quota or billing problems
- **Loop depth** — agent↔tools cycles per query; a spike indicates the model is thrashing
- Time-to-first-token versus total latency

**Exception hierarchy to fill the empty module:**

```python
class NavoraException(Exception): ...
class ToolExecutionError(NavoraException): ...
class LLMProviderError(NavoraException): ...
class ConfigurationError(NavoraException): ...
```

### 2.5 LLM Evaluation

**Current:** none. There is no way to tell whether a prompt or model change improved or degraded output.

**Why it matters:** LLM behaviour is non-deterministic. Without evals, every prompt tweak is a blind change and regressions ship silently.

| Component | Detail |
|---|---|
| Golden dataset | 50–100 representative queries across destinations, durations, budgets |
| Deterministic assertions | All 9 required sections present? Budget arithmetic internally consistent? Valid markdown? |
| LLM-as-judge | A stronger model scores helpfulness, specificity, faithfulness 1–5 against a rubric |
| Faithfulness check | Does every factual claim trace to a tool observation? |
| Trajectory eval | Did the agent call the right tools in a sensible order? Cheapest high-signal agent eval |
| Regression gate | Block the PR if the aggregate score drops more than X% versus `main` |
| Tooling | LangSmith Evaluations, RAGAS, or `promptfoo` |

### 2.6 Security

| Issue | Risk | Fix |
|---|---|---|
| `allow_origins=["*"]` | Any site can call the API | Explicit origin allowlist per environment |
| No authentication | Anyone can consume your API quota | API keys or OAuth2/JWT |
| No rate limiting | Cost-based denial of service | `slowapi` per-IP and per-key limits |
| Raw exceptions returned to client | Leaks internals | Log full trace server-side; return a generic message plus `request_id` |
| No input validation beyond type | Prompt injection surface | Length caps, content filtering |
| Secrets in `.env` | Fine locally, wrong in production | AWS Secrets Manager / Vault |

### 2.7 Scalability

| Concern | Current | Production approach |
|---|---|---|
| Long requests (20–90s) block | Synchronous | Job queue (Celery/Redis) plus polling or WebSocket streaming |
| Single process | One Uvicorn worker | Multiple workers behind a load balancer; Kubernetes HPA |
| No streaming | User waits on a spinner | `graph.astream()` → Server-Sent Events |
| Unbounded agent loop | No cycle cap | `recursion_limit` on invoke |

---

## 3. Engineering Case Studies

A code audit found and fixed 10 defects. These are the "tell me about a bug you debugged" stories.

| # | Defect | Root cause | Fix |
|---|---|---|---|
| 1 | Currency API returned 404 | `base_url` ended in `/`, then `f"{base_url}/{cur}"` produced a double slash | Removed trailing slash |
| 2 | Hotel cost string-repeated | Tool typed `price_per_night: str` but called `multiply(int,int)` — `"100"*3 == "100100100"` | Retyped to `float` plus explicit coercion |
| 3 | Tavily never authenticated | `.env` template had `TAVILAY_API_KEY` (typo) | Corrected to `TAVILY_API_KEY` |
| 4 | Kelvin reported as °C | Current-weather call omitted `units=metric` while the forecast call had it | Added `units` |
| 5 | Graph rebuilt every request | `GraphBuilder()` inside the route handler | Moved to FastAPI `lifespan` |
| 6 | `raise` on an f-string | `raise f"..."` is a `TypeError`, masking the real error | Replaced with `st.error()` |
| 7 | Config `model_name` ignored | `ChatOpenAI(model_name="o4-mini")` hardcoded after reading config | Use config value; added `else: raise` for unknown providers |
| 8 | **Fallback never fired** | Empty-but-successful Google response returned `None` without raising, so `except` never ran | `raise ValueError` inside `try` on empty result |
| 9 | Config only loaded from repo root | Relative path `"config/config.yaml"` | Resolve from `__file__` |
| 10 | YAML missing from installed package | `find_packages()` ships no data files | `package_data` plus `include_package_data` |

### The Best Story: Defect #8

Place search was designed to fall back to Tavily when Google Places failed. It worked correctly for exceptions — but when Google returned a **successful but empty** response, the `if result:` guard evaluated false, the function fell off the end returning `None`, and the `except` block never executed.

No exception. No log line. No error. The agent simply received nothing and quietly produced a thinner itinerary.

The fix routes both failure modes through the same handler by raising explicitly inside the `try` on an empty result.

**Why this story lands in an interview:** it demonstrates that "no errors in the logs" is not evidence of correctness, and it directly motivates why the empty `logger/` module is a genuine risk rather than a cosmetic gap. Silent degradation is the hardest class of failure to detect, and observability is the only defence.

---

## 4. Interview Preparation

### 4.1 The 30-Second Pitch

> "Navora AI is an agentic travel planner. Rather than asking an LLM to recall travel information — which it hallucinates — I gave the model a toolbox of ten functions wired to live APIs: weather, Google Places, currency rates, and budget arithmetic. It's built on LangGraph as a cyclic state machine implementing the ReAct pattern, so the agent autonomously decides which tools to call and in what order, loops until it has enough information, then synthesises a complete costed itinerary. Serving is FastAPI, the client is Streamlit, and the LLM provider is abstracted behind a gateway so Groq and OpenAI are interchangeable via config. The design decision I'm most confident in is delegating all arithmetic to Python tools — that makes the budget deterministic and auditable instead of hallucinated."

### 4.2 The 2-Minute Version

Add to the above:

- **The layering** — `tools/` are thin LangChain adapters; `utils/` is pure Python with zero framework imports, so the business logic is framework-swappable and fast to unit test.
- **Graceful degradation** — Google Places falls back to Tavily on both exception *and* empty result. I found the empty-result path was silently broken and fixed it.
- **Docstrings as schema** — `bind_tools()` turns docstrings into the JSON Schema the model reads, so docstrings are functionally load-bearing, not comments.
- **Current scope** — the agent layer is complete and verified. Tests, containerisation, CI, and observability are specified in the roadmap and are the next priorities.

### 4.3 Questions & Answers

<details>
<summary><b>Q: What is an "AI agent"? How is it different from a chatbot?</b></summary>

A chatbot maps input to output in one pass. An agent runs a **loop**: it reasons about what it needs, takes an **action** (calls a tool), **observes** the result, and repeats until the task is done. The defining property is **autonomous multi-step decision-making** — I don't hardcode the sequence. For "plan a Goa trip", the agent itself decides to check weather first, then attractions, then compute the budget.
</details>

<details>
<summary><b>Q: Why LangGraph instead of LangChain's AgentExecutor or a plain loop?</b></summary>

Agent loops are **cyclic** — `agent → tools → agent`. Most workflow tools model DAGs, which cannot express a cycle. LangGraph is built for cyclic stateful graphs. Over a hand-rolled `while` loop it provides checkpointing, streaming, visualisation, parallel tool fan-out, and interrupt/resume points as first-class features. Over `AgentExecutor` it provides explicit, inspectable control flow rather than a black box.
</details>

<details>
<summary><b>Q: How do you stop the LLM hallucinating prices and hotels?</b></summary>

Three layers. **Grounding** — every factual claim comes from a live API observation, not model memory. **Deterministic computation** — all arithmetic runs in Python tools, so budgets are calculated, never predicted. **Prompt constraints** — the system prompt mandates tool use and specifies the required output structure. The layer I haven't built yet is an automated faithfulness check that verifies every claim traces to a tool observation; that's in the evaluation roadmap.
</details>

<details>
<summary><b>Q: Why make arithmetic a tool? The model can do maths.</b></summary>

Not reliably. LLMs are next-token predictors; multi-step arithmetic is a known weakness, and a travel budget is exactly that — multiply nightly rate by nights, sum categories, divide by days. One wrong digit destroys user trust. Routing it to Python makes it deterministic, auditable, and unit-testable. The cost is extra round-trips, which is a trade I'd make every time.
</details>

<details>
<summary><b>Q: Walk me through what happens on a request.</b></summary>

Streamlit POSTs to `/query`. Pydantic validates the body. The question is wrapped as `{"messages":[q]}` and passed to the graph — pre-compiled at startup, not per request. The `agent` node prepends the system prompt and calls the LLM with tool schemas bound. `tools_condition` inspects the response: if it contains tool calls, route to `ToolNode`, which executes them and appends `ToolMessage`s to state; then the edge loops back to `agent`. That cycle repeats until the LLM returns a message with no tool calls, which routes to `END`. The last message's content is returned as `{"answer": ...}` and rendered as markdown.
</details>

<details>
<summary><b>Q: What happens when an external API goes down?</b></summary>

It depends on the tool. Place search has a real fallback — Google Places failures, whether an exception or an empty result, transparently route to Tavily AI search, so the agent still gets data. Weather returns `{}` on non-200 and the tool returns a clear "could not fetch" string, so the agent continues without it. Currency raises. What's missing is retry with exponential backoff and a circuit breaker — a repeatedly-failing API is currently retried on every request with no cooldown.
</details>

<details>
<summary><b>Q: How would you test a non-deterministic system?</b></summary>

Separate the deterministic parts from the generative ones. Everything in `utils/` is pure Python and unit-tests normally — that's exactly why I kept LangChain out of it. For agent logic I mock the LLM and assert on **control flow**: given a mocked response containing a tool call, does it route to the tools node? For the generative output I assert **structure, not prose** — required sections present, budget totals internally consistent. For quality regression you need **LLM-as-judge** against a golden dataset with a score threshold gating the PR. And **trajectory evaluation** — did it call the right tools in a sensible order — is the highest-signal, cheapest agent eval.
</details>

<details>
<summary><b>Q: This has no tests, no CI, no Docker. Why should I hire you?</b></summary>

Because I can tell you precisely what's missing, why it matters, and what I'd build. This project was scoped to learn agent orchestration, and that part is complete and verified. Productionising it is well-understood work: pytest with mocked network boundaries, a multi-stage Dockerfile with a non-root user and healthcheck, GitHub Actions running ruff/mypy/pytest with a coverage gate, and LangSmith tracing. I'd sequence tests and observability first, because without them every other change is unverifiable.
</details>

<details>
<summary><b>Q: How do you control cost?</b></summary>

Currently there's no caching, so every request re-hits every metered API, and no token budgeting. The fixes in priority order: Redis caching with per-source TTLs (weather 1h, places 7d, FX 1h), semantic caching on question embeddings, per-request token and cost tracking through the LLM gateway, a `recursion_limit` to cap runaway agent loops, and routing simple queries to a cheaper model. The gateway in `model_loader.py` is the right seam for all of it.
</details>

<details>
<summary><b>Q: How would you scale this to 10,000 users?</b></summary>

The blocker is that requests take 20–90 seconds and are synchronous. First, make it async: accept the request, enqueue a job (Celery + Redis), return a job ID, and stream progress over WebSocket or SSE. Second, cache aggressively — popular destinations repeat heavily, so even a 50% cache hit rate halves both cost and latency. Third, horizontal scaling: containerise, run multiple Uvicorn workers behind a load balancer, autoscale on queue depth. Fourth, move conversation state into Postgres via a LangGraph checkpointer so any worker can serve any request. Fifth, rate limit per user to bound cost exposure.
</details>

<details>
<summary><b>Q: What's the hardest bug you hit here?</b></summary>

A silent fallback failure. Place search was supposed to fall back to Tavily when Google Places failed. It worked for exceptions — but when Google returned a **successful but empty** response, the `if result:` guard was false, the function fell off the end returning `None`, and the `except` block never ran. No exception, no log line, no error. The agent just received nothing and quietly produced a thinner itinerary. I fixed it by raising explicitly inside the `try` on an empty result so both paths converge on the fallback. The lesson is that "no errors in the logs" is not evidence of correctness — which is exactly why I treat the empty logging module as a real risk rather than a cosmetic one.
</details>

<details>
<summary><b>Q: Why Groq?</b></summary>

Latency. Groq runs on LPU hardware with much faster token generation than typical GPU inference. That matters disproportionately for agents because an agent makes **several sequential LLM calls per request** — latency multiplies by loop depth. A 2× faster model is roughly a 2× faster product here. I chose `deepseek-r1-distill-llama-70b` specifically because it's reasoning-distilled, which suits multi-step tool planning. And because the provider sits behind a gateway, switching is a config change if the trade-off shifts.
</details>

<details>
<summary><b>Q: What would you do differently if you started over?</b></summary>

Two things. Write tests alongside the code instead of after — retrofitting tests onto working code is harder than it sounds. And set up structured logging and tracing on day one, because debugging agents without a trace of every LLM call and tool invocation is guesswork.
</details>

### 4.4 Points to Raise Proactively

Bringing these up before being asked reads as engineering maturity.

| Likely challenge | Response |
|---|---|
| "No tests." | "Correct, and it's my top priority. Here's the exact suite structure, and `utils/` was designed specifically to make it easy." |
| "Not deployed." | "It runs locally. I've specified the Dockerfile and CI pipeline; I haven't executed the deployment." |
| "Single agent is simple." | "Deliberately. Multi-agent adds coordination overhead this scope doesn't justify. I can explain when I *would* split it." |
| "CORS is permissive." | "A prototype default and a real vulnerability. Production needs an explicit allowlist, auth, and rate limiting." |
| "Empty logger module." | "The most consequential gap in the repo — defect #8 is exactly the kind of silent failure it would have caught." |

---

## 5. Resume Content

### Entry (matches the GoRide format)

**Navora AI – Agentic AI Travel Planner** · `GitHub`
**Overview:** Built an autonomous AI agent using LangGraph and FastAPI that researches destinations across live APIs and generates complete, costed travel itineraries from a single prompt.
**Features:** ReAct reasoning loop, 10 custom LLM tools, live weather & place search, automatic search-provider fallback, deterministic budget computation, currency conversion, swappable LLM provider, and a Streamlit chat interface.
**Tech Stack:** Python · LangGraph · LangChain · FastAPI · Streamlit · Groq · Pydantic · Google Places API · Tavily · OpenWeatherMap · REST

### Alternate — reliability emphasis

**Navora AI – Agentic AI Travel Planner** · `GitHub`
**Overview:** Built a tool-calling AI agent on LangGraph's cyclic state machine that autonomously orchestrates 4 external APIs to produce grounded, fully-costed travel plans, eliminating LLM hallucination of prices and locations.
**Features:** Autonomous multi-step tool selection, two-tier retrieval with graceful degradation, deterministic Python-based budgeting, YAML-driven LLM gateway (Groq/OpenAI), startup-time graph compilation, and REST API with auto-generated OpenAPI docs.
**Tech Stack:** Python · LangGraph · LangChain · FastAPI · Uvicorn · Streamlit · Groq · Pydantic · Google Places API · Tavily · OpenWeatherMap

### Alternate — plain language, for non-technical screeners

**Navora AI – AI Travel Planning Assistant** · `GitHub`
**Overview:** Built an AI assistant that replaces hours of manual trip research across a dozen websites, returning a complete day-by-day itinerary with verified live data and a full budget breakdown from one request.
**Features:** Real-time weather and place data, dual mainstream/off-beat itineraries, automatic backup data source on provider failure, accurate cost breakdowns and per-day budgets, live currency conversion, and a web chat interface.
**Tech Stack:** Python · LangGraph · LangChain · FastAPI · Streamlit · Groq · Google Places API · Tavily · OpenWeatherMap

### Bullet-Point Variants

**Technical**
- Built an autonomous AI agent using **LangGraph's cyclic state machine** implementing the **ReAct pattern**, orchestrating **10 custom tools** across 4 external APIs to generate complete, costed travel itineraries from a single prompt.
- Designed a **layered tool architecture** separating framework-agnostic service clients from LangChain adapters, making business logic independently unit-testable and the agent framework swappable.
- Implemented **two-tier retrieval with graceful degradation** — Google Places with automatic fallback to Tavily AI search — eliminating a single point of failure in the data pipeline.
- Eliminated LLM arithmetic hallucination by **delegating all budget computation to deterministic Python tools**, making cost breakdowns auditable rather than predicted.

**Impact-focused**
- Architected a production-pattern **LLM gateway** abstracting Groq and OpenAI behind a YAML-driven model registry, enabling provider switching via configuration with zero code changes.
- Reduced per-request overhead by **moving agent graph compilation from request-time to a FastAPI `lifespan` startup hook**, verified as zero rebuilds across repeated requests.
- Conducted a full code audit identifying and resolving **10 defects**, including a silent fallback failure where successful-but-empty API responses bypassed error handling and degraded agent output with no observable error.

**Business framing**
- Built an AI research assistant that replaces hours of manual trip planning across a dozen websites with a single request, returning a complete itinerary with verified live data and a full budget.
- Ensured reliability by designing the system to **automatically switch to a backup data source** when its primary provider is unavailable, so users receive an answer rather than an error.
- Prioritised **trustworthy output** by grounding every factual claim in a live data source and computing all costs deterministically, directly addressing AI's tendency to invent plausible-sounding details.

---

## 6. Concept Reference

### Concepts to Explain Without Notes

**Agentic:** agent vs. chatbot · ReAct · tool calling / function calling · agent loop · state machine · cyclic graph · reducer · tool schema · system prompt · grounding · hallucination · prompt engineering

**Engineering:** REST · ASGI vs. WSGI · Pydantic validation · dependency injection · adapter pattern · separation of concerns · graceful degradation · fail-fast · circuit breaker · idempotency · statelessness · lifespan/startup hooks · CORS

**Production ML:** LLM gateway · semantic caching · token budgeting · observability (logs/metrics/traces) · LLM-as-judge · golden dataset · trajectory evaluation · regression gating · drift

### Business & Non-Technical Vocabulary

| Term | Meaning in this project |
|---|---|
| **Agentic AI** | AI that *does things* rather than only answering |
| **Grounded AI** | AI whose answers are backed by real, checkable sources |
| **Single point of failure** | One component whose outage breaks everything — removed here by the search fallback |
| **Vendor lock-in** | Dependence on one supplier — avoided via the LLM gateway |
| **Fail-fast** | Surface problems immediately rather than degrading silently |
| **Technical debt** | Deferred work, tracked openly in §2 |
| **MVP** | Minimum Viable Product |
| **Cost per request** | Unit economics of an AI product; the main scaling constraint |
| **Time to first token** | How long before the user sees anything — a key UX metric |
| **User trust** | Why deterministic budgets matter more than fast ones |
| **Prototype vs. production** | Working software versus software that survives real traffic |

---

*Maintained alongside the [README](../README.md). Update §1 whenever implementation status changes.*
