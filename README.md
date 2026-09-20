<div align="center">

# 🌍 Navora AI

### An agentic AI travel planner that researches a destination in real time and returns a complete, costed itinerary.

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2-FF6F00)](https://langchain-ai.github.io/langgraph/)
[![LangChain](https://img.shields.io/badge/LangChain-1.4-1C3C3C?logo=langchain&logoColor=white)](https://python.langchain.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.63-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/Groq-LPU_Inference-F55036)](https://groq.com/)

</div>

---

## 📑 Table of Contents

| Section | What it covers |
|---|---|
| [1. What Is Navora AI?](#1-what-is-navora-ai) | Problem, solution, who it's for |
| [2. Implementation Status](#2-implementation-status) | Honest build vs. roadmap matrix |
| [3. Features](#3-features) | Capability list |
| [4. System Architecture](#4-system-architecture) | Diagrams + request lifecycle |
| [5. The Agent: Deep Dive](#5-the-agent-deep-dive) | ReAct loop, LangGraph state machine |
| [6. Tech Stack](#6-tech-stack) | Every library and why |
| [7. Tool Catalog](#7-tool-catalog) | All 10 agent tools |
| [8. Project Structure](#8-project-structure) | Folder-by-folder |
| [9. Data Acquisition Pipeline](#9-data-acquisition-pipeline) | Real-time retrieval + fallback |
| [10. LLM Gateway & Model Routing](#10-llm-gateway--model-routing) | Provider abstraction |
| [11. Getting Started](#11-getting-started) | Install + run |
| [12. Configuration](#12-configuration) | Env vars + YAML |
| [13. API Reference](#13-api-reference) | Endpoint contract |
| [14. Testing](#14-testing) | What exists, what to add |
| [15. Production Readiness](#15-production-readiness-gaps--how-id-close-them) | DB, CI, Docker, monitoring, eval |
| [16. Roadmap](#16-roadmap) | Prioritized next steps |
| [17. Troubleshooting](#17-troubleshooting) | Common failures |
| [18. Engineering Decisions](#18-engineering-decisions--trade-offs) | Why things are built this way |
| [19. Bugs Found & Fixed](#19-bugs-found--fixed) | Debugging case studies |
| [20. Interview Preparation](#20-interview-preparation) | **Q&A, talking points, resume bullets** |
| [21. Glossary](#21-glossary) | Technical + non-technical terms |

---

## 1. What Is Navora AI?

### The Problem

Planning a trip is a **multi-source research task**. A traveller opens a dozen tabs — Google Maps for attractions, TripAdvisor for restaurants, a weather site, a currency converter, a spreadsheet for budgeting — then manually stitches it together. The information is scattered, quickly stale, and the arithmetic is tedious and error-prone.

A plain chatbot cannot solve this. A large language model alone:
- **Hallucinates specifics** — invents hotels, prices, and opening hours.
- **Has a knowledge cutoff** — it cannot know today's weather or today's exchange rate.
- **Is bad at arithmetic** — it predicts tokens, it does not calculate. Multi-step budget maths is unreliable.

### The Solution

Navora AI is an **agentic application**. Instead of asking a model to *recall* an answer, it gives the model a **toolbox** and lets it *decide* which real-world APIs to call, in what order, to research the question — then synthesises the verified results into one document.

> **Plain-English version (for non-technical interviewers):**
> Think of it as hiring a research assistant rather than asking a know-it-all friend. The know-it-all answers instantly from memory and sometimes makes things up. The research assistant says "let me look that up," phones the weather service, checks the map, runs the numbers on a calculator, and comes back with a sourced answer. Navora AI is the research assistant.

Ask it *"Plan a trip to Goa for 5 days"* and it returns:
- Two complete itineraries — one **mainstream tourist** route, one **off-beat** route
- Day-by-day schedule
- Hotel recommendations with per-night costs
- Attractions, restaurants, and activities with details and prices
- Local transport options
- Live weather and forecast
- A full cost breakdown with a per-day budget
- Currency conversion to the traveller's home currency

### Who It's For

| Audience | Value |
|---|---|
| **Travellers** | One prompt replaces an afternoon of tab-juggling |
| **Engineers** | A compact, readable reference implementation of the ReAct agent pattern on LangGraph |
| **Interview candidates** | A project demonstrating agent orchestration, tool design, API integration, graceful degradation, and honest engineering trade-offs |

### What Makes It Non-Trivial

This is not a wrapper around a chat endpoint. The engineering substance is in:

1. **Tool-calling orchestration** — a cyclic state graph that loops between reasoning and acting until the task is complete.
2. **Grounding** — every factual claim (weather, places, exchange rates) comes from a live API, not from model memory.
3. **Deterministic arithmetic** — costs are computed in Python, not predicted by the LLM. This is the single highest-leverage design choice in the project.
4. **Graceful degradation** — when the Google Places API fails, the agent automatically falls back to AI web search rather than returning an error.
5. **Provider abstraction** — the LLM vendor is a one-line config change, not a refactor.

---

## 2. Implementation Status

> **Read this before an interview.** This project is a **working agentic prototype**, not a production-deployed system. The table below is deliberately honest — being able to articulate what is missing and how you would build it is a stronger interview signal than overclaiming. Section 15 covers each gap in depth.

| Area | Status | Notes |
|---|---|---|
| Agentic reasoning loop (ReAct) | ✅ **Built** | LangGraph cyclic state machine |
| Tool calling (10 tools) | ✅ **Built** | Weather, places, budget, FX |
| Real-time data retrieval | ✅ **Built** | 4 external APIs |
| Fallback / graceful degradation | ✅ **Built** | Google Places → Tavily |
| LLM gateway / provider routing | ✅ **Built** | Groq ⇄ OpenAI via YAML |
| REST API backend | ✅ **Built** | FastAPI + Pydantic validation |
| Web client | ✅ **Built** | Streamlit chat UI |
| Config management | ✅ **Built** | YAML + `.env`, path-independent |
| Markdown export | ⚠️ **Partial** | `save_document()` written but not wired to the endpoint |
| Automated tests | ❌ **Not built** | No pytest suite — §14 |
| Database / persistence | ❌ **Not built** | Stateless; no conversation memory — §15.1 |
| Containerisation (Docker) | ❌ **Not built** | §15.3 |
| CI/CD pipeline | ❌ **Not built** | Not yet a git repo — §15.4 |
| Observability / monitoring | ❌ **Not built** | `print()` only; empty logger module — §15.5 |
| LLM evaluation harness | ❌ **Not built** | §15.6 |
| Auth / rate limiting | ❌ **Not built** | CORS is wide open — §15.7 |
| Caching | ❌ **Not built** | Every request re-hits paid APIs — §15.2 |

**Legend:** ✅ Built and verified · ⚠️ Partially built · ❌ Designed but not implemented

---

## 3. Features

- 🧠 **Autonomous multi-step reasoning** — the agent plans its own research sequence; no hardcoded tool order.
- 🔄 **Cyclic ReAct loop** — reason → act → observe → repeat, until it has enough information.
- 🌦️ **Live weather** — current conditions and a 10-interval forecast via OpenWeatherMap.
- 📍 **Real place data** — attractions, restaurants, activities, and transport via Google Places.
- 🛡️ **Automatic fallback** — Tavily AI web search takes over transparently when Google Places fails.
- 🧮 **Deterministic budgeting** — hotel totals, trip totals, and per-day budgets computed in Python.
- 💱 **Live currency conversion** — real-time FX rates.
- 🎭 **Dual itineraries** — mainstream *and* off-beat plans in a single response.
- 🔌 **Swappable LLM provider** — Groq or OpenAI via one YAML line.
- ⚡ **Fast inference** — Groq LPU hardware for low-latency generation.
- 🚀 **Single-build startup** — the agent graph compiles once at boot, not per request.
- 📊 **Self-documenting graph** — renders its own architecture diagram to `my_graph.png` on startup.

---

## 4. System Architecture

### 4.1 High-Level

```
┌───────────────────────────────────────────────────────────────────────┐
│                          PRESENTATION LAYER                           │
│   streamlit_app.py  ·  chat UI, form handling, markdown rendering     │
└─────────────────────────────────┬─────────────────────────────────────┘
                                  │  HTTP POST /query  {"question": "..."}
                                  ▼
┌───────────────────────────────────────────────────────────────────────┐
│                           SERVING LAYER                               │
│   main.py  ·  FastAPI · Pydantic validation · CORS · lifespan boot    │
└─────────────────────────────────┬─────────────────────────────────────┘
                                  │  react_app.invoke({"messages":[...]})
                                  ▼
┌───────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION LAYER                            │
│   agent/agentic_workflow.py  ·  GraphBuilder · LangGraph StateGraph   │
│                                                                       │
│        START ──▶ agent ──(tools_condition)──▶ tools ──┐               │
│                    ▲                                   │              │
│                    └───────────────────────────────────┘              │
│                    └──▶ END   (when no tool call is emitted)          │
└──────────────┬───────────────────────────────────┬────────────────────┘
               │                                   │
               ▼                                   ▼
┌──────────────────────────┐      ┌────────────────────────────────────┐
│    LLM GATEWAY LAYER     │      │          TOOL LAYER                │
│  utils/model_loader.py   │      │  tools/*.py  ·  @tool wrappers     │
│  Groq ⇄ OpenAI routing   │      │  10 callable functions             │
│  config/config.yaml      │      └──────────────┬─────────────────────┘
└──────────────────────────┘                     │
                                                 ▼
                              ┌────────────────────────────────────────┐
                              │           SERVICE LAYER                │
                              │  utils/*.py  ·  raw API clients        │
                              │  (framework-agnostic, pure Python)     │
                              └──────────────┬─────────────────────────┘
                                             ▼
                              ┌────────────────────────────────────────┐
                              │        EXTERNAL DATA SOURCES           │
                              │  OpenWeatherMap · Google Places        │
                              │  Tavily Search  · exchangerate-api     │
                              └────────────────────────────────────────┘
```

### 4.2 Request Lifecycle

| # | Step | Component |
|---|---|---|
| 1 | User submits a question in the browser | `streamlit_app.py` |
| 2 | `POST /query` with a JSON body | HTTP |
| 3 | Pydantic validates the payload against `QueryRequest` | `main.py` |
| 4 | Question wrapped as `{"messages": [question]}` | `main.py` |
| 5 | Pre-compiled graph invoked (built once at startup) | `main.py` |
| 6 | System prompt prepended, LLM called with tool schemas bound | `agent_function` |
| 7 | `tools_condition` routes → `tools` if the LLM requested calls, else → `END` | LangGraph |
| 8 | `ToolNode` executes the requested tools, possibly in parallel | LangGraph |
| 9 | Tool results appended to state as `ToolMessage`s; loop back to step 6 | LangGraph |
| 10 | Loop repeats until the LLM emits a final answer with no tool calls | LangGraph |
| 11 | Last message's `.content` extracted | `main.py` |
| 12 | `{"answer": "<markdown>"}` returned | HTTP |
| 13 | Rendered as markdown with a metadata header | `streamlit_app.py` |

### 4.3 Layered Design Principle

The codebase enforces a strict separation that is worth calling out in an interview:

```
tools/  ──depends on──▶  utils/  ──depends on──▶  external HTTP APIs
  │                        │
  │                        └── Pure Python. Zero LangChain imports.
  │                            Unit-testable in isolation. Reusable
  │                            outside this project entirely.
  │
  └── Thin @tool adapters. Docstrings become the LLM-facing schema.
      Swap agent frameworks and only this layer changes.
```

**Why this matters:** the business logic (`utils/`) has no knowledge of the agent framework. If LangGraph were replaced with CrewAI, AutoGen, or a hand-rolled loop, `utils/` would not change a single line. This is the **Adapter pattern**, and it is the most defensible architectural decision in the repo.

---

## 5. The Agent: Deep Dive

### 5.1 The ReAct Pattern

**ReAct = Reasoning + Acting.** Instead of answering in one shot, the model alternates between thinking and using tools.

```
Thought:     "I need Goa's weather before I can recommend outdoor activities."
Action:      get_weather_forecast(city="Goa")
Observation: "2026-09-20: 28°C, light rain ..."
Thought:     "Rain expected. I should weight indoor options. Now I need attractions."
Action:      search_attractions(place="Goa")
Observation: "Basilica of Bom Jesus, Fort Aguada, ..."
Thought:     "I have enough. Now compute the budget."
Action:      estimate_total_hotel_cost(price_per_night=3500, total_days=5)
Observation: 17500.0
Thought:     "I can now write the final plan."
Final Answer: # Goa 5-Day Itinerary ...
```

The critical property: **the sequence is not hardcoded.** The model chooses which tools to call and when, based on what it has learned so far.

### 5.2 The LangGraph State Machine

Defined in [`agent/agentic_workflow.py`](agent/agentic_workflow.py):

```python
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("agent", self.agent_function)
graph_builder.add_node("tools", ToolNode(tools=self.tools))
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")      # ← the cycle
graph_builder.add_edge("agent", END)
self.graph = graph_builder.compile()
```

| Element | Role |
|---|---|
| `StateGraph` | A graph whose nodes mutate a shared state object |
| `MessagesState` | Built-in state: a list of messages with an **append** reducer |
| `agent` node | Calls the LLM with the system prompt + full history |
| `tools` node | `ToolNode` — executes every tool call in the last AI message |
| `tools_condition` | Prebuilt router: → `"tools"` if tool calls exist, else → `END` |
| `tools → agent` edge | **The cycle.** Makes this an agent rather than a pipeline |
| `.compile()` | Validates and freezes the graph into a runnable |

**Why a graph instead of a `while` loop?** Graphs give you free streaming, checkpointing hooks, visualisation, parallel tool fan-out, and interrupt/resume points. A hand-rolled loop gives you none of those without significant extra work.

### 5.3 State Management

`MessagesState` is a `TypedDict` with one key, `messages`, annotated with an **`add_messages` reducer**. When a node returns `{"messages": [response]}`, the reducer **appends** rather than overwrites. This is what lets conversational context accumulate across loop iterations.

```python
def agent_function(self, state: MessagesState):
    user_question = state["messages"]
    input_question = [self.system_prompt] + user_question   # prepend, never persist
    response = self.llm_with_tools.invoke(input_question)
    return {"messages": [response]}                          # appended by the reducer
```

Note the system prompt is prepended **at call time** and never written into state — so it appears exactly once per LLM call rather than accumulating on every cycle.

### 5.4 Tool Binding

```python
self.llm_with_tools = self.llm.bind_tools(tools=self.tools)
```

`bind_tools` converts each Python function into a **JSON Schema** the model can read. The function name becomes the tool name; the **docstring becomes the description**; type hints become the parameter schema.

> **This is why docstrings are load-bearing in this codebase.** They are not developer comments — they are the prompt the model reads to decide which tool to use. A vague docstring is a functional bug.

### 5.5 Prompt Engineering

[`prompt_library/prompt.py`](prompt_library/prompt.py) holds a single `SystemMessage`. Techniques used:

| Technique | Implementation |
|---|---|
| **Role assignment** | *"You are a helpful AI Travel Agent and Expense Planner"* |
| **Explicit output contract** | A 9-item checklist of required sections |
| **Tool-use nudge** | *"Use the available tools to gather information"* |
| **Anti-clarification instruction** | *"Give full information immediately"* — stops the agent stalling on follow-up questions |
| **Format constraint** | *"Provide everything in one comprehensive response formatted in clean Markdown"* |
| **Creative differentiation** | *"two plans... generic... and off-beat"* |

---

## 6. Tech Stack

### 6.1 Core

| Layer | Technology | Version | Why this choice |
|---|---|---|---|
| Language | Python | 3.10.20 | Ecosystem standard for AI; pinned via `.python-version` |
| Agent orchestration | **LangGraph** | 1.2.11 | Cyclic graphs — needed for agent loops, which DAG tools can't express |
| LLM framework | **LangChain** | 1.4.0 | Tool abstraction, message types, provider adapters |
| Backend API | **FastAPI** | 0.141.1 | Async, auto OpenAPI docs, Pydantic-native |
| ASGI server | **Uvicorn** | 0.52.4 | High-performance async server |
| Frontend | **Streamlit** | 1.63.0 | Pure-Python UI — no JS toolchain for a prototype |
| Validation | **Pydantic** | 2.13.5 | Runtime type enforcement on API + config |
| Package manager | **uv** | 0.11.19 | 10–100× faster than pip; deterministic resolution |
| Config | PyYAML + python-dotenv | 6.0.3 / 1.2.3 | Non-secret config in YAML, secrets in `.env` |

### 6.2 LLM Providers

| Provider | Model | Role |
|---|---|---|
| **Groq** | `deepseek-r1-distill-llama-70b` | Default. LPU hardware → very low latency. A *reasoning-distilled* model, well-suited to multi-step tool planning |
| **OpenAI** | `o4-mini` | Alternative. Reasoning-optimised, cost-efficient |

### 6.3 External Data APIs

| API | Purpose | Failure behaviour |
|---|---|---|
| **OpenWeatherMap** | Current weather + forecast | Returns `{}` on non-200; tool reports a friendly message |
| **Google Places** | Attractions, restaurants, activities, transport | **Falls back to Tavily** |
| **Tavily** | AI-optimised web search | Terminal fallback |
| **exchangerate-api v6** | Live FX rates | Raises on non-200 |

---

## 7. Tool Catalog

All 10 tools bound to the LLM, verified at runtime:

### 🌦️ Weather — [`tools/weather_info_tool.py`](tools/weather_info_tool.py)

| Tool | Signature | Returns |
|---|---|---|
| `get_current_weather` | `(city: str)` | Temperature (°C) + description |
| `get_weather_forecast` | `(city: str)` | 10 forecast intervals, date + temp + description |

### 📍 Place Search — [`tools/place_search_tool.py`](tools/place_search_tool.py)

| Tool | Signature | Primary → Fallback |
|---|---|---|
| `search_attractions` | `(place: str)` | Google Places → Tavily |
| `search_restaurants` | `(place: str)` | Google Places → Tavily |
| `search_activities` | `(place: str)` | Google Places → Tavily |
| `search_transportation` | `(place: str)` | Google Places → Tavily |

### 🧮 Budget — [`tools/expense_calculator_tool.py`](tools/expense_calculator_tool.py)

| Tool | Signature | Returns |
|---|---|---|
| `estimate_total_hotel_cost` | `(price_per_night: float, total_days: float)` | Product |
| `calculate_total_expense` | `(*costs: float)` | Sum |
| `calculate_daily_expense_budget` | `(total_cost: float, days: int)` | Quotient (÷0-safe) |

### 💱 Currency — [`tools/currency_conversion_tool.py`](tools/currency_conversion_tool.py)

| Tool | Signature | Returns |
|---|---|---|
| `convert_currency` | `(amount: float, from_currency: str, to_currency: str)` | Converted amount at live rate |

> **Design note — why arithmetic is a tool.** LLMs are next-token predictors and are unreliable at multi-step arithmetic. Delegating every calculation to Python makes the budget **deterministic and auditable**. This is the highest-value correctness decision in the project and a strong talking point.

---

## 8. Project Structure

```
20_NavoraAI/
│
├── main.py                        # FastAPI app — serving layer, lifespan boot
├── streamlit_app.py               # Streamlit chat client
│
├── agent/
│   └── agentic_workflow.py        # GraphBuilder — LangGraph state machine
│
├── tools/                         # LangChain @tool adapters (LLM-facing)
│   ├── weather_info_tool.py
│   ├── place_search_tool.py
│   ├── expense_calculator_tool.py
│   ├── currency_conversion_tool.py
│   └── arthamatic_op_tool.py      # legacy/unused (AlphaVantage)
│
├── utils/                         # Pure-Python services (framework-agnostic)
│   ├── model_loader.py            # LLM gateway — provider routing
│   ├── config_loader.py           # YAML loader, CWD-independent
│   ├── weather_info.py            # OpenWeatherMap client
│   ├── place_info_search.py       # Google Places + Tavily clients
│   ├── expense_calculator.py      # Arithmetic
│   ├── currency_converter.py      # exchangerate-api client
│   └── save_to_document.py        # Markdown export (not wired up)
│
├── prompt_library/
│   └── prompt.py                  # SYSTEM_PROMPT
│
├── config/
│   └── config.yaml                # Model registry (non-secret)
│
├── logger/logging.py              # EMPTY — see §15.5
├── exception/exceptiohandling.py  # EMPTY (note: filename typo)
│
├── notebook/experiments.ipynb     # Scratch prototyping
│
├── .env                           # secrets (gitignored)
├── .env.name                      # Template for .env
├── requirements.txt               # Dependencies
├── setup.py                       # Package metadata
├── pyproject.toml                 # uv project file
├── .python-version                # Pins Python 3.10
└── my_graph.png                   # Auto-generated graph diagram
```

---

## 9. Data Acquisition Pipeline

Navora AI's "scraping / generation pipeline" is a **real-time, agent-driven retrieval pipeline**. Unlike a batch scraper that crawls on a schedule into a warehouse, retrieval here is **on-demand and model-directed** — the agent decides what to fetch based on the user's question.

```
User question
     │
     ▼
LLM decides which data it needs        ← model-directed, not hardcoded
     │
     ├──▶ Structured API  ── Google Places ──┐
     │                                       │ on exception OR empty result
     │                                       ▼
     ├──▶ AI Web Search  ── Tavily (include_answer="advanced")
     │         │
     │         └── Tavily crawls + LLM-summarises pages into a direct answer
     │
     ├──▶ Structured API  ── OpenWeatherMap (units=metric)
     │
     └──▶ Structured API  ── exchangerate-api v6
                │
                ▼
     Results appended to graph state as ToolMessages
                │
                ▼
     LLM synthesises everything into one Markdown document
```

### Two-Tier Retrieval Strategy

| Tier | Source | Characteristics |
|---|---|---|
| **Tier 1 — Structured** | Google Places | High precision, schema-stable, quota-limited, costs money |
| **Tier 2 — Unstructured** | Tavily AI search | Broad coverage, always available, LLM-summarised, less structured |

The fallback triggers on **both** an exception *and* an empty-but-successful response:

```python
try:
    result = self.google_places_search.google_search_attractions(place)
    if result:
        return f"...as suggested by google: {result}"
    raise ValueError("Google Places returned no results")   # ← forces fallback
except Exception as e:
    tavily_result = self.tavily_search.tavily_search_attractions(place)
    return f"Google cannot find the details due to {e}. ...{tavily_result}"
```

> **Interview talking point:** the `raise` inside the `try` is deliberate. Without it, a successful-but-empty API response returns `None` and the fallback never fires — the agent silently receives nothing. This was a real bug found and fixed (§19).

### Why This Is Not RAG

A common interview follow-up. **RAG** (Retrieval-Augmented Generation) retrieves from a *pre-indexed vector store* you own. Navora AI retrieves from *live third-party APIs* at query time. Both ground the model in external facts, but:

| | RAG | Navora AI |
|---|---|---|
| Source | Your embedded document corpus | Live external APIs |
| Freshness | As fresh as your last index run | Real-time |
| Retrieval trigger | Similarity search on the query | LLM decides via tool calls |
| Infra | Vector DB + embedding pipeline | HTTP clients |

---

## 10. LLM Gateway & Model Routing

[`utils/model_loader.py`](utils/model_loader.py) is a minimal **LLM gateway** — a single abstraction point between the application and any model vendor.

```python
class ModelLoader(BaseModel):
    model_provider: Literal["groq", "openai"] = "groq"

    def load_llm(self):
        if self.model_provider == "groq":
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")
            model_name = self.config["llm"]["groq"]["model_name"]
            llm = ChatGroq(model=model_name, api_key=groq_api_key)
        elif self.model_provider == "openai":
            ...
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")
        return llm
```

**Model registry** — [`config/config.yaml`](config/config.yaml):

```yaml
llm:
  openai:
    provider: "openai"
    model_name: "o4-mini"
  groq:
    provider: "groq"
    model_name: "deepseek-r1-distill-llama-70b"
```

### Properties This Buys

| Property | How |
|---|---|
| **No vendor lock-in** | Swap providers with one argument |
| **Config-driven upgrades** | New model version = YAML edit, no code change |
| **Fail-fast** | Missing keys raise a named error at boot, not a cryptic 401 mid-request |
| **Type-safe** | `Literal["groq","openai"]` rejects typos at construction |

### Production Gateway Upgrade Path

A production gateway (LiteLLM, Portkey, or in-house) adds: automatic **failover** between providers, **cost tracking** per request, **semantic caching**, **rate limiting**, **retry with backoff**, and **A/B routing**. Navora AI's gateway is the correct *seam* for all of these — they slot in behind `load_llm()` without touching the agent.

---

## 11. Getting Started

### Prerequisites

- **Python 3.10+** (3.10.20 is pinned)
- **uv** — install with `pip install uv`
- API keys (see §12)

### Installation

```bash
# 1. Enter the project
cd "path/to/20_NavoraAI"

# 2. Create the virtual environment (uv auto-downloads Python 3.10)
uv venv .venv --python 3.10

# 3. Activate it
.\.venv\Scripts\activate        # Windows PowerShell
source .venv/bin/activate       # macOS / Linux

# 4. Install dependencies (117 packages)
uv pip install -r requirements.txt

# 5. Create your secrets file
cp .env.name .env               # then fill in your keys
```

### Running

Navora AI is **two processes**. Start the backend first.

**Terminal 1 — Backend**
```bash
uvicorn main:app --reload --port 8000
```
Serves the API at `http://localhost:8000` · interactive docs at `http://localhost:8000/docs`

**Terminal 2 — Frontend**
```bash
streamlit run streamlit_app.py
```
Opens at `http://localhost:8501`

> The Streamlit client targets `http://localhost:8000` (hardcoded at [`streamlit_app.py:8`](streamlit_app.py#L8)), so the backend **must** run on port 8000.

**Without activating the venv:**
```bash
.\.venv\Scripts\uvicorn.exe main:app --reload --port 8000
.\.venv\Scripts\streamlit.exe run streamlit_app.py
```

**Backend only (curl):**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Plan a trip to Goa for 5 days"}'
```

### First Query

Try: `Plan a 5-day trip to Goa with a mid-range budget in INR`

Expect **20–90 seconds** — the agent makes multiple sequential LLM calls and several external API round-trips.

---

## 12. Configuration

### Environment Variables — `.env`

| Variable | Required | Purpose | Source |
|---|:---:|---|---|
| `GROQ_API_KEY` | Yes | Default LLM provider | [console.groq.com/keys](https://console.groq.com/keys) |
| `GPLACES_API_KEY` | Yes | Google Places | [console.cloud.google.com](https://console.cloud.google.com) (enable *Places API*) |
| `TAVILY_API_KEY` | Yes | Search fallback | [app.tavily.com](https://app.tavily.com) |
| `OPENWEATHERMAP_API_KEY` | Yes | Weather tools | [openweathermap.org](https://home.openweathermap.org/api_keys) |
| `EXCHANGE_RATE_API_KEY` | Yes | Currency tool | [exchangerate-api.com](https://www.exchangerate-api.com) |
| `OPENAI_API_KEY` | Optional | Only for `model_provider="openai"` | [platform.openai.com](https://platform.openai.com) |

> ⚠️ **`GPLACES_API_KEY` must begin with `AIza`.** The `googlemaps` client validates the key *format* at construction time and raises `ValueError("Invalid API key provided.")` — this kills the app **at startup**, before any request. For local development without a real key, use a placeholder like `AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`: startup succeeds and all place lookups transparently fall through to Tavily.

### Model Config — `config/config.yaml`

Non-secret configuration lives in YAML. To change model:

```yaml
llm:
  groq:
    provider: "groq"
    model_name: "llama-3.3-70b-versatile"   # ← edit here only
```

To switch provider, change the argument in [`main.py`](main.py):
```python
graph = GraphBuilder(model_provider="openai")
```

### Configuration Philosophy

| Kind | Location | Rationale |
|---|---|---|
| **Secrets** | `.env` (gitignored) | Must never enter version control |
| **Tunables** | `config/config.yaml` | Reviewable in PRs, changeable without code edits |
| **Prompts** | `prompt_library/prompt.py` | Versioned with code; prompts *are* logic |

---

## 13. API Reference

### `POST /query`

**Request**
```json
{ "question": "Plan a 5-day trip to Goa" }
```

**Response — 200**
```json
{ "answer": "# 🌍 Goa 5-Day Itinerary\n\n## Day 1 ..." }
```

**Response — 500**
```json
{ "error": "Error code: 401 - Invalid API Key" }
```

**Schema** — enforced by Pydantic:
```python
class QueryRequest(BaseModel):
    question: str
```
A malformed body returns **422 Unprocessable Entity** automatically.

### Interactive Docs

FastAPI generates OpenAPI docs for free:
- **Swagger UI** → `http://localhost:8000/docs`
- **ReDoc** → `http://localhost:8000/redoc`

### Startup Behaviour

The agent graph is built **once** in a FastAPI `lifespan` context manager, not per request:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global react_app
    graph = GraphBuilder(model_provider="groq")
    react_app = graph()
    _dump_graph_png(react_app)
    yield
```

**Trade-off to be aware of:** this surfaces bad API keys at **boot** rather than on the first request. The server refuses to start on a bad key. That is intentional fail-fast behaviour, but it means a misconfigured deployment crashes visibly instead of degrading silently. The PNG render is wrapped in `try/except` so a network failure there can never block startup.

---

## 14. Testing

> **Current status: no automated test suite.** Verification to date has been manual smoke-testing plus ad-hoc verification scripts. This is the single most valuable gap to close, and an interviewer will very likely ask about it.

### What Was Verified Manually

| Check | Result |
|---|---|
| All third-party imports resolve on LangChain 1.x | Pass — 8/8 |
| Graph compiles with all 10 tools bound | Pass — nodes `['__start__','agent','tools','__end__']` |
| Config loads with CWD set outside the repo | Pass |
| `multiply("100", 3) == 300.0` (not `"100100100"`) | Pass |
| Missing `GROQ_API_KEY` raises a named `ValueError` | Pass |
| 3 requests → 0 graph rebuilds | Pass |

### The Test Suite I Would Write

```
tests/
├── unit/
│   ├── test_expense_calculator.py   # pure functions — trivial, high value
│   ├── test_config_loader.py        # CWD independence, missing-file error
│   └── test_currency_converter.py   # URL construction, error paths (mocked)
├── integration/
│   ├── test_tools.py                # each @tool with mocked HTTP (responses/respx)
│   ├── test_fallback.py             # assert Tavily fires on Google failure AND empty result
│   └── test_graph.py                # graph compiles; agent→tools→agent cycle runs
└── e2e/
    └── test_api.py                  # FastAPI TestClient, LLM mocked
```

**Testing pyramid rationale:** the majority of tests should be fast unit tests over `utils/` — which is exactly why `utils/` was kept free of LangChain imports. Mock every network boundary; never hit paid APIs in CI.

**Testing an LLM is different.** You cannot assert exact string equality on a generative output. Strategies:
- **Mock the LLM** for deterministic logic tests (routing, state, tool wiring).
- **Assert structure, not prose** — "response contains a day-by-day section", "budget total equals sum of parts".
- **Assert tool-call traces** — given a weather question, did the agent actually call `get_weather_forecast`?
- **LLM-as-judge** for quality regression — see §15.6.

---

## 15. Production Readiness: Gaps & How I'd Close Them

This section is deliberately explicit about what is **not** built. Each item includes the reasoning an interviewer is looking for.

### 15.1 Database / Persistence

**Current:** None. The app is fully stateless. No checkpointer is passed to `.compile()`, so **conversation memory does not survive a request** — every query starts cold.

**What I'd add:**

| Need | Solution |
|---|---|
| Conversation memory across turns | `langgraph.checkpoint.postgres.PostgresSaver` passed to `.compile(checkpointer=...)`, keyed by `thread_id` |
| Saved itineraries | PostgreSQL — `users`, `trips`, `itineraries` tables |
| API response cache | Redis with per-source TTL |
| Semantic cache on similar questions | pgvector / Qdrant over question embeddings |

```python
# One-line change to gain durable multi-turn memory:
self.graph = graph_builder.compile(checkpointer=PostgresSaver(conn))
```

### 15.2 Caching

**Current:** None. Every request re-hits every paid API. Two users asking about Goa on the same day pay twice for identical data.

**What I'd add:**

| Layer | TTL | Rationale |
|---|---|---|
| Weather | 1 hour | Changes slowly |
| Places | 7 days | Attractions rarely change |
| FX rates | 1 hour | Sufficient for trip budgeting |
| Semantic LLM cache | — | Embed the question; serve cached answers above a similarity threshold |

Expected impact: large cost reduction and a major latency win on repeat destinations.

### 15.3 Containerisation (Docker)

**Current:** No `Dockerfile`.

**What I'd add** — multi-stage build, non-root user, healthcheck:

```dockerfile
FROM python:3.10-slim AS builder
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY requirements.txt setup.py ./
RUN uv pip install --system --no-cache -r requirements.txt

FROM python:3.10-slim
WORKDIR /app
RUN useradd -m -u 1000 appuser
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
```

Plus `docker-compose.yml` running `api`, `ui`, `postgres`, and `redis` on one network.

> **Note:** a `/health` endpoint must be added first — the app currently has none, and `/docs` is a poor healthcheck target because it does not exercise the agent.

### 15.4 CI/CD

**Current:** Not built. The project is **not yet a git repository**.

**What I'd add** — GitHub Actions on every PR:

| Stage | Tool |
|---|---|
| Lint | `ruff` |
| Format check | `black --check` |
| Type check | `mypy` |
| Unit + integration tests | `pytest --cov` (fail under 80%) |
| Secret scanning | `gitleaks` |
| Dependency audit | `pip-audit` |
| Build image | `docker build` |
| Deploy | On `main` merge → staging → manual gate → production |

**First step before any of this:** `git init`, commit, and push. CI cannot exist without version control.

### 15.5 Observability & Monitoring

**Current:** `logger/logging.py` and `exception/exceptiohandling.py` are **empty files**. The app uses bare `print()`.

**What I'd add:**

| Pillar | Tool | Signal |
|---|---|---|
| **Logging** | `structlog` → JSON | Structured logs with a `request_id` correlating every tool call in one agent run |
| **Tracing** | **LangSmith** or OpenTelemetry | Full agent trace: every LLM call, prompt, tool invocation, token count, latency |
| **Metrics** | Prometheus + Grafana | p50/p95/p99 latency, tool error rate, tokens/request, cost/request, fallback rate |
| **Alerting** | Alertmanager | Page on error rate > 5%, p95 > 120s, or daily spend over budget |

**LLM-specific metrics that generic APM misses:**
- Tokens per request (input/output split) and **cost per request**
- **Tool-call success rate** per tool
- **Fallback rate** — how often Google Places fails into Tavily (a rising rate signals quota/billing problems)
- **Loop depth** — how many agent↔tools cycles per query; a spike indicates the model is thrashing
- Time-to-first-token vs. total latency

**A concrete custom exception hierarchy** to fill the empty module:
```python
class NavoraException(Exception): ...
class ToolExecutionError(NavoraException): ...
class LLMProviderError(NavoraException): ...
class ConfigurationError(NavoraException): ...
```

### 15.6 LLM Evaluation

**Current:** None. There is no way to tell whether a prompt or model change made output better or worse.

**Why it matters:** LLM behaviour is non-deterministic. Without evals, every prompt tweak is a blind change and regressions ship silently.

**What I'd add:**

| Component | Detail |
|---|---|
| **Golden dataset** | 50–100 representative queries across destinations, durations, budgets |
| **Deterministic assertions** | All 9 required sections present? Budget arithmetic internally consistent? Valid markdown? |
| **LLM-as-judge** | A stronger model scores helpfulness, specificity, and faithfulness 1–5 against a rubric |
| **Faithfulness / hallucination check** | Does every factual claim trace to a tool observation? |
| **Trajectory eval** | Did the agent call the *right* tools in a sensible order? Cheapest high-signal eval for agents |
| **Regression gate** | Block the PR if the aggregate score drops more than X% vs. `main` |
| **Tooling** | LangSmith Evaluations, RAGAS, or `promptfoo` |

### 15.7 Security

**Current gaps:**

| Issue | Risk | Fix |
|---|---|---|
| `allow_origins=["*"]` | Any site can call the API | Explicit origin allowlist per environment |
| No authentication | Anyone can burn your API quota | API keys or OAuth2/JWT |
| No rate limiting | Cost-based DoS | `slowapi` per-IP + per-key limits |
| Raw exceptions returned to client | Leaks internals | Log full trace server-side; return a generic message + `request_id` |
| No input validation beyond type | Prompt injection surface | Length caps, content filtering |
| Secrets in `.env` | Fine locally, wrong in prod | AWS Secrets Manager / Vault |

### 15.8 Scalability

| Concern | Current | Production approach |
|---|---|---|
| Long requests (20–90s) block | Synchronous | Job queue (Celery/Redis) + polling or WebSocket streaming |
| Single process | One Uvicorn worker | Multiple workers behind a load balancer; K8s HPA |
| No streaming | User stares at a spinner | `graph.astream()` → Server-Sent Events, tokens as they generate |
| Unbounded agent loop | No cycle cap | `recursion_limit` on invoke to prevent runaway cost |

---

## 16. Roadmap

**Phase 1 — Engineering foundations**
- [ ] `git init` + push to GitHub
- [ ] `pytest` suite (unit → integration → e2e)
- [ ] Fill `logger/` with `structlog`; fill `exception/` with the exception hierarchy
- [ ] Add `/health` endpoint
- [ ] Remove dead code (`arthamatic_op_tool.py`); wire up `save_document()`

**Phase 2 — Productionisation**
- [ ] Dockerfile + docker-compose
- [ ] GitHub Actions CI
- [ ] Redis caching layer
- [ ] Rate limiting + API-key auth
- [ ] Lock down CORS

**Phase 3 — Quality**
- [ ] LangSmith tracing
- [ ] Golden eval dataset + LLM-as-judge
- [ ] Prometheus/Grafana dashboard

**Phase 4 — Product**
- [ ] PostgreSQL checkpointer for multi-turn conversations
- [ ] Response streaming (SSE)
- [ ] PDF export
- [ ] Flight/hotel booking APIs (Amadeus, Skyscanner)
- [ ] User accounts + saved trips
- [ ] Multi-city and multi-currency trips

---

## 17. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ValidationError: Invalid API key provided` **at startup** | `GPLACES_API_KEY` missing or not starting with `AIza` | Use a real key or the `AIzaSy...` placeholder |
| `ValueError: GROQ_API_KEY is not set` | Empty key in `.env` | Add a real Groq key |
| `401 Invalid API Key` in the response | Key present but wrong | Regenerate at the provider console |
| Streamlit: `Connection refused` | Backend not running | Start `uvicorn` first, on port 8000 |
| `FileNotFoundError: config.yaml` | Config file deleted | Restore `config/config.yaml` |
| Server won't start, mentions Mermaid | Graph PNG render failed | Non-fatal by design — check network; startup continues |
| Very slow first response | Expected | Multiple LLM + API round-trips; 20–90s is normal |
| Wrong temperature units | Fixed — `units=metric` | See §19 |
| `google.api_core` Python 3.10 EOL warning | Harmless | Ignore, or move to 3.11 |

---

## 18. Engineering Decisions & Trade-offs

| Decision | Alternative | Why this way | Cost |
|---|---|---|---|
| **LangGraph** for orchestration | Hand-rolled `while` loop | Free cycles, checkpointing, streaming, visualisation | Framework dependency + learning curve |
| **Arithmetic as tools** | Let the LLM compute | Deterministic, auditable budgets | Extra round-trips |
| **`utils/` free of LangChain** | Tools call APIs directly | Framework-swappable, unit-testable | One extra layer of indirection |
| **Single agent, many tools** | Multi-agent (planner/researcher/writer) | Far simpler; adequate for this scope | Less specialisation |
| **Graph built at startup** | Per request | Removes rebuild latency from every request | Bad keys crash boot instead of one request |
| **Groq default** | OpenAI default | Much lower latency; agent loops multiply latency | Smaller model catalogue |
| **Streamlit frontend** | React/Next.js | Zero JS toolchain, fast to build | Limited UX control |
| **Two-tier search fallback** | Google Places only | Resilient to quota exhaustion | Two API keys to manage |
| **YAML + `.env` split** | All env vars | Tunables reviewable in PRs | Two config sources |
| **Fail-fast on missing keys** | Lazy failure | Clear errors at boot | Server won't start when misconfigured |

---

## 19. Bugs Found & Fixed

A code audit found and fixed 10 defects. These make excellent "tell me about a bug you debugged" stories.

| # | Bug | Root cause | Fix |
|---|---|---|---|
| 1 | Currency API 404s | `base_url` ended in `/`, then `f"{base_url}/{cur}"` produced a double slash | Removed trailing slash |
| 2 | Hotel cost string-repeated | Tool typed `price_per_night: str` but called `multiply(int,int)` — `"100"*3 == "100100100"` | Retyped to `float` + explicit coercion |
| 3 | Tavily never authenticated | `.env` template had `TAVILAY_API_KEY` (typo) | Corrected to `TAVILY_API_KEY` |
| 4 | Kelvin reported as °C | Current-weather call omitted `units=metric` while the forecast call had it | Added `units` |
| 5 | Graph rebuilt every request | `GraphBuilder()` inside the route handler | Moved to FastAPI `lifespan` |
| 6 | `raise` on an f-string | `raise f"..."` is a `TypeError`, masking the real error | Replaced with `st.error()` |
| 7 | Config model_name ignored | `ChatOpenAI(model_name="o4-mini")` hardcoded after reading config | Use config value; added `else: raise` for unknown providers |
| 8 | **Fallback never fired** | Empty-but-successful Google response returned `None` without raising, so `except` never ran | `raise ValueError` inside `try` on empty result |
| 9 | Config only loaded from repo root | Relative path `"config/config.yaml"` | Resolve from `__file__` |
| 10 | YAML missing from installed package | `find_packages()` ships no data files | `package_data` + `include_package_data` |

> **Bug #8 is the best story.** It is a *silent* failure — no exception, no error log, just an agent quietly reasoning with missing data and producing a worse plan. It illustrates why "no errors in the logs" does not mean "working", and why the empty `logger/` module (§15.5) is a genuine risk rather than a cosmetic gap.

---

## 20. Interview Preparation

### 20.1 The 30-Second Pitch

> "Navora AI is an agentic travel planner. Rather than asking an LLM to recall travel information — which it hallucinates — I gave the model a toolbox of ten functions wired to live APIs: weather, Google Places, currency rates, and budget arithmetic. It's built on LangGraph as a cyclic state machine implementing the ReAct pattern, so the agent autonomously decides which tools to call and in what order, loops until it has enough information, then synthesises a complete costed itinerary. Serving is FastAPI, the client is Streamlit, and the LLM provider is abstracted behind a gateway so Groq and OpenAI are interchangeable via config. The design decision I'm most confident in is delegating all arithmetic to Python tools — that makes the budget deterministic and auditable instead of hallucinated."

### 20.2 The 2-Minute Deep Version

Add to the above:
- **The layering** — `tools/` are thin LangChain adapters; `utils/` is pure Python with zero framework imports, so the business logic is framework-swappable and fast to unit test.
- **Graceful degradation** — Google Places falls back to Tavily AI search on both exception *and* empty result. I found the empty-result path was silently broken and fixed it.
- **Docstrings as schema** — `bind_tools()` turns docstrings into the JSON Schema the model reads, so docstrings are functionally load-bearing, not comments.
- **Honest gaps** — no tests, no Docker, no CI, no database, no eval harness. I know exactly what each would look like and why each matters, and I'd prioritise the test suite and observability first.

### 20.3 Likely Questions & Strong Answers

<details>
<summary><b>Q: What is an "AI agent"? How is it different from a chatbot?</b></summary>

A chatbot maps input to output in one pass. An agent runs a **loop**: it reasons about what it needs, takes an **action** (calls a tool), **observes** the result, and repeats until the task is done. The defining property is **autonomous multi-step decision-making** — I don't hardcode the sequence. For "plan a Goa trip", the agent itself decides to check weather first, then attractions, then compute the budget.
</details>

<details>
<summary><b>Q: Why LangGraph instead of LangChain's AgentExecutor or a plain loop?</b></summary>

Agent loops are **cyclic** — `agent → tools → agent`. Most workflow tools model DAGs, which cannot express a cycle. LangGraph is built for cyclic stateful graphs. Over a hand-rolled `while` loop it gives me checkpointing, streaming, visualisation, parallel tool fan-out, and interrupt/resume points for free. Over `AgentExecutor` it gives explicit, inspectable control flow instead of a black box.
</details>

<details>
<summary><b>Q: How do you stop the LLM hallucinating prices and hotels?</b></summary>

Three layers. **Grounding** — every factual claim comes from a live API observation, not model memory. **Deterministic computation** — all arithmetic runs in Python tools, so budgets are calculated, never predicted. **Prompt constraints** — the system prompt mandates tool use and specifies the required output structure. What I have *not* built is an automated faithfulness check that verifies every claim traces to a tool observation — that's in the eval roadmap.
</details>

<details>
<summary><b>Q: Why make arithmetic a tool? The model can do maths.</b></summary>

It can't, reliably. LLMs are next-token predictors; multi-step arithmetic is a known weakness, and a travel budget is exactly that — multiply nightly rate by nights, sum categories, divide by days. One wrong digit destroys user trust. Routing it to Python makes it deterministic, auditable, and unit-testable. The cost is extra round-trips, which is a trade I'd make every time.
</details>

<details>
<summary><b>Q: Walk me through what happens on a request.</b></summary>

Streamlit POSTs to `/query`. Pydantic validates the body. The question is wrapped as `{"messages":[q]}` and passed to the graph — pre-compiled at startup, not per request. The `agent` node prepends the system prompt and calls the LLM with tool schemas bound. `tools_condition` inspects the response: if it contains tool calls, route to `ToolNode`, which executes them and appends `ToolMessage`s to state; then the edge loops back to `agent`. That cycle repeats until the LLM returns a message with no tool calls, which routes to `END`. The last message's content is returned as `{"answer": ...}` and rendered as markdown.
</details>

<details>
<summary><b>Q: What happens when an external API goes down?</b></summary>

Depends on the tool. Place search has a real fallback — Google Places failures (exception *or* empty result) transparently route to Tavily AI search, so the agent still gets data. Weather returns `{}` on non-200 and the tool returns a friendly "could not fetch" string, so the agent continues without it. Currency raises. What's missing is retry with exponential backoff and a circuit breaker — a repeatedly-failing API is currently retried on every request with no cooldown.
</details>

<details>
<summary><b>Q: How would you test a non-deterministic system?</b></summary>

Separate the deterministic parts from the generative ones. Everything in `utils/` is pure Python and unit-tests normally — that's exactly why I kept LangChain out of it. For agent logic I mock the LLM and assert on **control flow**: given a mocked response containing a tool call, does it route to the tools node? For the generative output I assert **structure not prose** — required sections present, budget totals internally consistent. For quality regression you need **LLM-as-judge** against a golden dataset with a score threshold gating the PR. And **trajectory evaluation** — did it call the right tools in a sensible order — is the highest-signal, cheapest agent eval.
</details>

<details>
<summary><b>Q: This has no tests, no CI, no Docker. Why should I hire you?</b></summary>

Because I can tell you precisely what's missing, why it matters, and what I'd build — which is in the README rather than hidden. This was built to learn agent orchestration, and that part is solid and verified. Productionising it is well-understood work: pytest with mocked network boundaries, a multi-stage Dockerfile with a non-root user and healthcheck, GitHub Actions running ruff/mypy/pytest with a coverage gate, and LangSmith tracing. I'd sequence tests and observability first, because without them every other change is unverifiable. I'd rather show you an honest status table than claim a deployment I don't have.
</details>

<details>
<summary><b>Q: How do you control cost?</b></summary>

Today, poorly — and I know it. There's no caching, so every request re-hits every paid API, and no token budgeting. The fixes in priority order: Redis caching with per-source TTLs (weather 1h, places 7d, FX 1h), semantic caching on question embeddings, per-request token and cost tracking through the LLM gateway, a `recursion_limit` to cap runaway agent loops, and routing simple queries to a cheaper model. The gateway in `model_loader.py` is the right seam for all of it.
</details>

<details>
<summary><b>Q: How would you scale this to 10,000 users?</b></summary>

The blocker is that requests take 20–90 seconds and are synchronous. First, make it async: accept the request, enqueue a job (Celery + Redis), return a job ID, and stream progress over WebSocket or SSE. Second, cache aggressively — popular destinations repeat heavily, so a cache hit rate of even 50% halves both cost and latency. Third, horizontal scaling: containerise, run multiple Uvicorn workers behind a load balancer, autoscale on queue depth. Fourth, move conversation state into Postgres via a LangGraph checkpointer so any worker can serve any request. Fifth, rate limit per user to bound cost exposure.
</details>

<details>
<summary><b>Q: What's the hardest bug you hit here?</b></summary>

A silent fallback failure. Place search was supposed to fall back to Tavily when Google Places failed. It worked for exceptions — but when Google returned a **successful but empty** response, the `if result:` guard was false, the function fell off the end returning `None`, and the `except` block never ran. No exception, no log line, no error. The agent just received nothing and quietly produced a thinner itinerary. I fixed it by raising explicitly inside the `try` on an empty result so both paths converge on the fallback. The lesson is that "no errors in the logs" is not evidence of correctness — which is exactly why I flag the empty logging module as a real risk.
</details>

<details>
<summary><b>Q: Why Groq?</b></summary>

Latency. Groq runs on LPU hardware with much faster token generation than typical GPU inference. That matters disproportionately for agents because an agent makes **several sequential LLM calls per request** — latency multiplies by loop depth. A 2× faster model is roughly a 2× faster product here. I chose `deepseek-r1-distill-llama-70b` specifically because it's reasoning-distilled, which suits multi-step tool planning. And because the provider sits behind a gateway, switching is a config change if the trade-off shifts.
</details>

<details>
<summary><b>Q: What would you do differently if you started over?</b></summary>

Three things. Write tests alongside the code instead of after — retrofitting tests onto working code is much harder than it sounds. Set up structured logging and tracing on day one, because debugging agents without a trace of every LLM call and tool invocation is guesswork. And initialise git immediately — this project isn't under version control yet, which is the root reason there's no CI.
</details>

### 20.4 Resume Bullets

Pick 3–4. Every claim below is verifiable in the code.

**Concise**
> **Navora AI — Agentic Travel Planner** · *Python, LangGraph, LangChain, FastAPI, Streamlit, Groq*
> - Built an autonomous AI agent using **LangGraph's cyclic state machine** implementing the **ReAct pattern**, orchestrating **10 custom tools** across 4 external APIs to generate complete, costed travel itineraries from a single prompt.
> - Designed a **layered tool architecture** separating framework-agnostic service clients from LangChain adapters, making business logic independently unit-testable and the agent framework swappable.
> - Implemented **two-tier retrieval with graceful degradation** — Google Places with automatic fallback to Tavily AI search — eliminating a single point of failure in the data pipeline.
> - Eliminated LLM arithmetic hallucination by **delegating all budget computation to deterministic Python tools**, making cost breakdowns auditable rather than predicted.

**Impact-focused**
> - Architected a production-pattern **LLM gateway** abstracting Groq and OpenAI behind a YAML-driven model registry, enabling provider switching via configuration with zero code changes.
> - Reduced per-request overhead by **moving agent graph compilation from request-time to a FastAPI `lifespan` startup hook**, verified as zero rebuilds across repeated requests.
> - Conducted a full code audit identifying and fixing **10 defects**, including a silent fallback failure where successful-but-empty API responses bypassed error handling and degraded agent output with no observable error.

**Non-technical / business framing** (for recruiters and non-engineering panels)
> - Built an AI research assistant that replaces hours of manual trip planning across a dozen websites with a single request, returning a complete itinerary with verified live data and a full budget.
> - Ensured reliability by designing the system to **automatically switch to a backup data source** when its primary provider is unavailable, so users get an answer rather than an error.
> - Prioritised **trustworthy output** by grounding every factual claim in a live data source and computing all costs deterministically, directly addressing AI's well-known tendency to invent plausible-sounding details.

### 20.5 Red Flags to Pre-empt

Bring these up **before** you're asked — it reads as maturity, not weakness.

| Likely challenge | Your answer |
|---|---|
| "No tests." | "Correct, and it's my top priority. Here's the exact suite structure and why `utils/` was designed to make it easy." |
| "Not deployed." | "It runs locally. I've specified the Dockerfile and CI pipeline; I haven't executed the deployment." |
| "Single agent is simple." | "Deliberately. Multi-agent adds coordination overhead that this scope doesn't justify. I can explain when I *would* split it." |
| "CORS is wide open." | "A prototype default and a real vulnerability. Production needs an explicit allowlist, auth, and rate limiting." |
| "Empty logger module." | "The most honest flaw in the repo — and bug #8 is exactly the kind of silent failure it would have caught." |

### 20.6 Concepts to Be Able to Explain Cold

**Agentic:** agent vs. chatbot · ReAct · tool calling / function calling · agent loop · state machine · cyclic graph · reducer · tool schema · system prompt · grounding · hallucination · prompt engineering

**Engineering:** REST · ASGI vs. WSGI · Pydantic validation · dependency injection · adapter pattern · separation of concerns · graceful degradation · fail-fast · circuit breaker · idempotency · statelessness · lifespan/startup hooks · CORS

**Production ML:** LLM gateway · semantic caching · token budgeting · observability (logs/metrics/traces) · LLM-as-judge · golden dataset · trajectory evaluation · regression gating · drift

---

## 21. Glossary

### Technical

| Term | Meaning |
|---|---|
| **Agent** | An LLM that can take actions via tools in a loop, deciding its own steps |
| **ReAct** | *Reasoning + Acting* — alternating thought and tool use |
| **Tool / Function calling** | Exposing a function's schema so the model can request it be run |
| **LangGraph** | Framework for cyclic, stateful agent graphs |
| **StateGraph** | A graph whose nodes read and mutate shared state |
| **Reducer** | Function defining how a state update merges (here: append) |
| **ToolNode** | Prebuilt node that executes tool calls from the last AI message |
| **`tools_condition`** | Prebuilt router: tools if calls exist, else END |
| **Checkpointer** | Persists graph state so conversations survive restarts |
| **Grounding** | Anchoring generated text in retrieved real data |
| **Hallucination** | Fluent, confident, false model output |
| **RAG** | Retrieval-Augmented Generation — retrieval from an owned index |
| **LLM Gateway** | Abstraction layer over model vendors |
| **LPU** | Groq's Language Processing Unit — inference-optimised hardware |
| **Token** | Sub-word unit; the billing and context unit for LLMs |
| **Context window** | Max tokens a model can consider at once |
| **System prompt** | Instruction defining the model's role and constraints |
| **ASGI** | Async Server Gateway Interface — async successor to WSGI |
| **Pydantic** | Runtime validation from Python type hints |
| **Lifespan** | FastAPI startup/shutdown hook |
| **CORS** | Browser policy controlling cross-origin requests |
| **Graceful degradation** | Continuing with reduced capability instead of failing |
| **Circuit breaker** | Stops calling a failing dependency for a cooldown period |
| **LLM-as-judge** | Using a model to score another model's output |
| **Golden dataset** | Curated test inputs with expected quality criteria |
| **Trajectory evaluation** | Scoring the agent's *sequence of tool calls*, not just the answer |
| **Observability** | Logs + metrics + traces |
| **p95 latency** | 95% of requests are faster than this |
| **Multi-stage build** | Docker pattern separating build tools from the runtime image |

### Non-Technical / Business

| Term | Meaning in this project |
|---|---|
| **Agentic AI** | AI that *does things* rather than only answering |
| **Grounded AI** | AI whose answers are backed by real, checkable sources |
| **Single point of failure** | One component whose outage breaks everything — removed here by the search fallback |
| **Vendor lock-in** | Dependence on one supplier — avoided via the LLM gateway |
| **Fail-fast** | Surface problems immediately rather than degrading silently |
| **Technical debt** | Deferred work (tests, Docker, CI) tracked openly in §15 |
| **MVP** | Minimum Viable Product — what this is |
| **Cost per request** | Unit economics of an AI product; the main scaling constraint |
| **Time to first token** | How long before the user sees *anything* — a key UX metric |
| **User trust** | Why deterministic budgets matter more than fast ones |
| **Prototype vs. production** | Working software vs. software that survives real traffic |

---

## 🙏 Acknowledgements

Built with [LangChain](https://python.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/). Inference by [Groq](https://groq.com/). Data from [OpenWeatherMap](https://openweathermap.org/), [Google Places](https://developers.google.com/maps/documentation/places/web-service), [Tavily](https://tavily.com/), and [exchangerate-api](https://www.exchangerate-api.com/).

## 📄 License

No license file is currently present. Add one (MIT is the common default) before publishing.

---

<div align="center">

**Navora AI** · *Agentic travel planning, grounded in real data.*

Built by **Akshat Gupta**

</div>
