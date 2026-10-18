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
| [1. Overview](#1-overview) | What Navora AI is and the problem it solves |
| [2. Features](#2-features) | Capability list |
| [3. System Architecture](#3-system-architecture) | Diagrams and request lifecycle |
| [4. The Agent](#4-the-agent) | ReAct loop and LangGraph state machine |
| [5. Tech Stack](#5-tech-stack) | Every library and why |
| [6. Tool Catalog](#6-tool-catalog) | All 10 agent tools |
| [7. Project Structure](#7-project-structure) | Folder-by-folder |
| [8. Data Acquisition Pipeline](#8-data-acquisition-pipeline) | Real-time retrieval and fallback |
| [9. LLM Gateway & Model Routing](#9-llm-gateway--model-routing) | Provider abstraction |
| [10. Getting Started](#10-getting-started) | Install and run |
| [11. Configuration](#11-configuration) | Environment variables and YAML |
| [12. API Reference](#12-api-reference) | Endpoint contract |
| [13. Quality & Verification](#13-quality--verification) | Verification approach and test strategy |
| [14. Design Decisions](#14-design-decisions--trade-offs) | Engineering rationale |
| [15. Roadmap](#15-roadmap) | Planned enhancements |
| [16. Troubleshooting](#16-troubleshooting) | Common issues |
| [17. Glossary](#17-glossary) | Key terms |

---

## 1. Overview

### The Problem

Planning a trip is a **multi-source research task**. A traveller opens a dozen tabs — maps for attractions, review sites for restaurants, a weather service, a currency converter, a spreadsheet for budgeting — then manually stitches it all together. The information is scattered, quickly goes stale, and the arithmetic is tedious and error-prone.

A conventional chatbot cannot solve this well. A large language model on its own:

- **Invents specifics** — hotels, prices, and opening hours that sound plausible but aren't real.
- **Has a knowledge cutoff** — it cannot know today's weather or today's exchange rate.
- **Is unreliable at arithmetic** — it predicts tokens rather than calculating, so multi-step budget maths is untrustworthy.

### The Approach

Navora AI is an **agentic application**. Rather than asking a model to *recall* an answer, it gives the model a **toolbox** and lets it *decide* which real-world APIs to call, and in what order, to research the question — then synthesises the verified results into a single document.

> **In plain terms:** it works like a research assistant rather than a know-it-all. A know-it-all answers instantly from memory and occasionally makes things up. A research assistant says "let me look that up," checks the weather service, consults the map, runs the numbers on a calculator, and returns a sourced answer.

Ask it *"Plan a trip to Goa for 5 days"* and it returns:

- Two complete itineraries — one **mainstream tourist** route, one **off-beat** route
- A day-by-day schedule
- Hotel recommendations with per-night costs
- Attractions, restaurants, and activities with details and prices
- Local transport options
- Live weather and forecast
- A full cost breakdown with a per-day budget
- Currency conversion to the traveller's home currency

### Engineering Highlights

| Capability | Implementation |
|---|---|
| **Tool-calling orchestration** | A cyclic state graph that loops between reasoning and acting until the task is complete |
| **Grounding** | Every factual claim — weather, places, exchange rates — comes from a live API rather than model memory |
| **Deterministic arithmetic** | All costs are computed in Python, making budgets auditable instead of predicted |
| **Graceful degradation** | When the primary place-search provider fails, the agent transparently falls back to AI web search |
| **Provider abstraction** | The LLM vendor is a configuration change, not a refactor |

---

## 2. Features

- 🧠 **Autonomous multi-step reasoning** — the agent plans its own research sequence; the tool order is not hardcoded.
- 🔄 **Cyclic ReAct loop** — reason → act → observe → repeat, until it has gathered enough information.
- 🌦️ **Live weather** — current conditions and a 10-interval forecast via OpenWeatherMap.
- 📍 **Real place data** — attractions, restaurants, activities, and transport via Google Places.
- 🛡️ **Automatic fallback** — Tavily AI web search takes over transparently if Google Places is unavailable.
- 🧮 **Deterministic budgeting** — hotel totals, trip totals, and per-day budgets computed in Python.
- 💱 **Live currency conversion** — real-time exchange rates.
- 🎭 **Dual itineraries** — mainstream *and* off-beat plans in a single response.
- 🔌 **Swappable LLM provider** — Groq or OpenAI via one YAML entry.
- ⚡ **Low-latency inference** — Groq LPU hardware for fast generation.
- 🚀 **Single-build startup** — the agent graph compiles once at boot rather than per request.
- 📊 **Self-documenting graph** — renders its own architecture diagram to `my_graph.png` on startup.

---

## 3. System Architecture

### 3.1 High-Level

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

### 3.2 Request Lifecycle

| # | Step | Component |
|---|---|---|
| 1 | User submits a question in the browser | `streamlit_app.py` |
| 2 | `POST /query` with a JSON body | HTTP |
| 3 | Pydantic validates the payload against `QueryRequest` | `main.py` |
| 4 | Question wrapped as `{"messages": [question]}` | `main.py` |
| 5 | Pre-compiled graph invoked (built once at startup) | `main.py` |
| 6 | System prompt prepended, LLM called with tool schemas bound | `agent_function` |
| 7 | `tools_condition` routes → `tools` if the LLM requested calls, else → `END` | LangGraph |
| 8 | `ToolNode` executes the requested tools, potentially in parallel | LangGraph |
| 9 | Tool results appended to state as `ToolMessage`s; loop back to step 6 | LangGraph |
| 10 | Loop repeats until the LLM emits a final answer with no tool calls | LangGraph |
| 11 | Last message's `.content` extracted | `main.py` |
| 12 | `{"answer": "<markdown>"}` returned | HTTP |
| 13 | Rendered as markdown with a metadata header | `streamlit_app.py` |

### 3.3 Layered Design Principle

The codebase enforces a strict dependency direction:

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

**Why this matters:** the business logic in `utils/` has no knowledge of the agent framework. If LangGraph were replaced with another orchestrator, `utils/` would not change a single line. This is the **Adapter pattern** applied at the module boundary, and it is the project's central architectural decision.

---

## 4. The Agent

### 4.1 The ReAct Pattern

**ReAct = Reasoning + Acting.** Rather than answering in one shot, the model alternates between thinking and using tools.

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

The defining property: **the sequence is not hardcoded.** The model chooses which tools to call and when, based on what it has learned so far.

### 4.2 The LangGraph State Machine

Defined in [`agent/agentic_workflow.py`](agent/agentic_workflow.py):

```python
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("agent", self.agent_function)
graph_builder.add_node("tools", ToolNode(tools=self.tools))
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")      # the cycle
graph_builder.add_edge("agent", END)
self.graph = graph_builder.compile()
```

| Element | Role |
|---|---|
| `StateGraph` | A graph whose nodes mutate a shared state object |
| `MessagesState` | Built-in state: a list of messages with an **append** reducer |
| `agent` node | Calls the LLM with the system prompt plus full history |
| `tools` node | `ToolNode` — executes every tool call in the last AI message |
| `tools_condition` | Prebuilt router: → `"tools"` if tool calls exist, else → `END` |
| `tools → agent` edge | **The cycle.** This is what makes the system an agent rather than a pipeline |
| `.compile()` | Validates and freezes the graph into a runnable |

**Why a graph rather than a `while` loop?** A graph provides streaming, checkpointing hooks, visualisation, parallel tool fan-out, and interrupt/resume points as first-class features. A hand-rolled loop would require building each of those separately.

### 4.3 State Management

`MessagesState` is a `TypedDict` with one key, `messages`, annotated with an **`add_messages` reducer**. When a node returns `{"messages": [response]}`, the reducer **appends** rather than overwrites — which is what lets conversational context accumulate across loop iterations.

```python
def agent_function(self, state: MessagesState):
    user_question = state["messages"]
    input_question = [self.system_prompt] + user_question   # prepend, never persist
    response = self.llm_with_tools.invoke(input_question)
    return {"messages": [response]}                          # appended by the reducer
```

The system prompt is prepended **at call time** and never written into state, so it appears exactly once per LLM call rather than accumulating on every cycle.

### 4.4 Tool Binding

```python
self.llm_with_tools = self.llm.bind_tools(tools=self.tools)
```

`bind_tools` converts each Python function into a **JSON Schema** the model can read. The function name becomes the tool name, the **docstring becomes the description**, and type hints become the parameter schema.

> Docstrings in `tools/` are therefore load-bearing: they are the text the model reads when deciding which tool to use, not developer-facing commentary.

### 4.5 Prompt Design

[`prompt_library/prompt.py`](prompt_library/prompt.py) holds a single `SystemMessage`. The techniques applied:

| Technique | Implementation |
|---|---|
| **Role assignment** | *"You are a helpful AI Travel Agent and Expense Planner"* |
| **Explicit output contract** | A 9-item checklist of required sections |
| **Tool-use directive** | *"Use the available tools to gather information"* |
| **Anti-clarification instruction** | *"Give full information immediately"* — prevents the agent stalling on follow-up questions |
| **Format constraint** | *"Provide everything in one comprehensive response formatted in clean Markdown"* |
| **Creative differentiation** | *"two plans... generic... and off-beat"* |

---

## 5. Tech Stack

### 5.1 Core

| Layer | Technology | Version | Rationale |
|---|---|---|---|
| Language | Python | 3.10.20 | Ecosystem standard for AI; pinned via `.python-version` |
| Agent orchestration | **LangGraph** | 1.2.11 | Cyclic graphs — required for agent loops, which DAG tools cannot express |
| LLM framework | **LangChain** | 1.4.0 | Tool abstraction, message types, provider adapters |
| Backend API | **FastAPI** | 0.141.1 | Async, automatic OpenAPI docs, Pydantic-native |
| ASGI server | **Uvicorn** | 0.52.4 | High-performance async server |
| Frontend | **Streamlit** | 1.63.0 | Pure-Python UI, no JavaScript toolchain required |
| Validation | **Pydantic** | 2.13.5 | Runtime type enforcement on API and config |
| Package manager | **uv** | 0.11.19 | Substantially faster than pip with deterministic resolution |
| Config | PyYAML + python-dotenv | 6.0.3 / 1.2.3 | Non-secret config in YAML, secrets in `.env` |

### 5.2 LLM Providers

| Provider | Model | Role |
|---|---|---|
| **Groq** | `deepseek-r1-distill-llama-70b` | Default. LPU hardware for low latency. A reasoning-distilled model, well-suited to multi-step tool planning |
| **OpenAI** | `o4-mini` | Alternative. Reasoning-optimised and cost-efficient |

### 5.3 External Data APIs

| API | Purpose | Failure handling |
|---|---|---|
| **OpenWeatherMap** | Current weather and forecast | Returns `{}` on non-200; the tool reports a clear message |
| **Google Places** | Attractions, restaurants, activities, transport | Falls back to Tavily |
| **Tavily** | AI-optimised web search | Terminal fallback |
| **exchangerate-api v6** | Live FX rates | Raises on non-200 |

---

## 6. Tool Catalog

All 10 tools bound to the LLM:

### 🌦️ Weather — [`tools/weather_info_tool.py`](tools/weather_info_tool.py)

| Tool | Signature | Returns |
|---|---|---|
| `get_current_weather` | `(city: str)` | Temperature (°C) and description |
| `get_weather_forecast` | `(city: str)` | 10 forecast intervals with date, temp, description |

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
| `calculate_daily_expense_budget` | `(total_cost: float, days: int)` | Quotient (division-by-zero safe) |

### 💱 Currency — [`tools/currency_conversion_tool.py`](tools/currency_conversion_tool.py)

| Tool | Signature | Returns |
|---|---|---|
| `convert_currency` | `(amount: float, from_currency: str, to_currency: str)` | Converted amount at the live rate |

> **Design note — arithmetic as a tool.** Language models are next-token predictors and are unreliable at multi-step arithmetic. Delegating every calculation to Python makes the budget **deterministic and auditable**, which is the project's highest-value correctness decision.

---

## 7. Project Structure

```
NavoraAI/
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
│   └── arthamatic_op_tool.py      # standalone arithmetic helpers
│
├── utils/                         # Pure-Python services (framework-agnostic)
│   ├── model_loader.py            # LLM gateway — provider routing
│   ├── config_loader.py           # YAML loader, CWD-independent
│   ├── weather_info.py            # OpenWeatherMap client
│   ├── place_info_search.py       # Google Places + Tavily clients
│   ├── expense_calculator.py      # Arithmetic
│   ├── currency_converter.py      # exchangerate-api client
│   └── save_to_document.py        # Markdown export helper
│
├── prompt_library/
│   └── prompt.py                  # SYSTEM_PROMPT
│
├── config/
│   └── config.yaml                # Model registry (non-secret)
│
├── logger/                        # Logging module (see Roadmap)
├── exception/                     # Custom exception module (see Roadmap)
│
├── docs/
│   └── PROJECT_NOTES.md           # Extended technical notes
│
├── notebook/experiments.ipynb     # Prototyping notebook
│
├── .env                           # Secrets (gitignored)
├── .env.name                      # Template for .env
├── requirements.txt               # Dependencies
├── setup.py                       # Package metadata
├── pyproject.toml                 # uv project file
├── .python-version                # Pins Python 3.10
└── my_graph.png                   # Auto-generated graph diagram
```

---

## 8. Data Acquisition Pipeline

Navora AI's retrieval pipeline is **real-time and agent-driven**. Unlike a batch scraper that crawls on a schedule into a warehouse, retrieval here is **on-demand and model-directed** — the agent decides what to fetch based on the user's question.

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
     │         └── Tavily crawls and LLM-summarises pages into a direct answer
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
| **Tier 1 — Structured** | Google Places | High precision, schema-stable, quota-limited, metered |
| **Tier 2 — Unstructured** | Tavily AI search | Broad coverage, always available, LLM-summarised, less structured |

The fallback triggers on **both** an exception *and* an empty-but-successful response:

```python
try:
    result = self.google_places_search.google_search_attractions(place)
    if result:
        return f"...as suggested by google: {result}"
    raise ValueError("Google Places returned no results")   # routes to the fallback
except Exception as e:
    tavily_result = self.tavily_search.tavily_search_attractions(place)
    return f"Google cannot find the details due to {e}. ...{tavily_result}"
```

The explicit `raise` inside the `try` block is deliberate. Without it, a successful-but-empty API response would return `None` and bypass the fallback entirely, leaving the agent to reason with missing data and no error surfaced. Routing both the exception path and the empty-result path through the same handler guarantees the agent always receives usable data.

### Relationship to RAG

**RAG** (Retrieval-Augmented Generation) retrieves from a *pre-indexed vector store* that you own and maintain. Navora AI retrieves from *live third-party APIs* at query time. Both approaches ground the model in external facts, but they differ materially:

| | RAG | Navora AI |
|---|---|---|
| Source | An embedded document corpus | Live external APIs |
| Freshness | As fresh as the last index run | Real-time |
| Retrieval trigger | Similarity search on the query | The LLM decides, via tool calls |
| Infrastructure | Vector DB + embedding pipeline | HTTP clients |

---

## 9. LLM Gateway & Model Routing

[`utils/model_loader.py`](utils/model_loader.py) implements an **LLM gateway** — a single abstraction point between the application and any model vendor.

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

### Properties

| Property | Mechanism |
|---|---|
| **No vendor lock-in** | Providers swap with a single argument |
| **Config-driven upgrades** | A new model version is a YAML edit, not a code change |
| **Fail-fast** | Missing keys raise a named error at boot rather than a cryptic 401 mid-request |
| **Type-safe** | `Literal["groq","openai"]` rejects typos at construction time |

### Extension Path

This gateway is the correct seam for the capabilities a production deployment adds: automatic **failover** between providers, per-request **cost tracking**, **semantic caching**, **rate limiting**, **retry with backoff**, and **A/B routing**. Each slots in behind `load_llm()` without touching the agent layer.

---

## 10. Getting Started

### Prerequisites

- **Python 3.10+** (3.10.20 is pinned)
- **uv** — install with `pip install uv`
- API keys (see [Configuration](#11-configuration))

### Installation

```bash
# 1. Enter the project
cd NavoraAI

# 2. Create the virtual environment (uv auto-downloads Python 3.10)
uv venv .venv --python 3.10

# 3. Activate it
.\.venv\Scripts\activate        # Windows
source .venv/bin/activate       # macOS / Linux

# 4. Install dependencies
uv pip install -r requirements.txt

# 5. Create your secrets file
cp .env.name .env               # then fill in your keys
```

### Running

Navora AI runs as **two processes**. Start the backend first.

**Terminal 1 — Backend**
```bash
uvicorn main:app --reload --port 8000
```
API at `http://localhost:8000` · interactive docs at `http://localhost:8000/docs`

**Terminal 2 — Frontend**
```bash
streamlit run streamlit_app.py
```
Opens at `http://localhost:8501`

> The Streamlit client targets `http://localhost:8000` ([`streamlit_app.py:8`](streamlit_app.py#L8)), so the backend must run on port 8000.

**Without activating the venv:**
```bash
.\.venv\Scripts\uvicorn.exe main:app --reload --port 8000
.\.venv\Scripts\streamlit.exe run streamlit_app.py
```

**Backend only, via curl:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Plan a trip to Goa for 5 days"}'
```

### First Query

Try: `Plan a 5-day trip to Goa with a mid-range budget in INR`

Expect **20–90 seconds**. The agent makes several sequential LLM calls and multiple external API round-trips before composing the final plan.

---

## 11. Configuration

### Environment Variables — `.env`

| Variable | Required | Purpose | Source |
|---|:---:|---|---|
| `GROQ_API_KEY` | Yes | Default LLM provider | [console.groq.com/keys](https://console.groq.com/keys) |
| `GPLACES_API_KEY` | Yes | Google Places | [console.cloud.google.com](https://console.cloud.google.com) (enable *Places API*) |
| `TAVILY_API_KEY` | Yes | Search fallback | [app.tavily.com](https://app.tavily.com) |
| `OPENWEATHERMAP_API_KEY` | Yes | Weather tools | [openweathermap.org](https://home.openweathermap.org/api_keys) |
| `EXCHANGE_RATE_API_KEY` | Yes | Currency tool | [exchangerate-api.com](https://www.exchangerate-api.com) |
| `OPENAI_API_KEY` | Optional | Only for `model_provider="openai"` | [platform.openai.com](https://platform.openai.com) |

> **Note on `GPLACES_API_KEY`:** the underlying `googlemaps` client validates the key *format* at construction time and requires it to begin with `AIza`. For local development without a provisioned key, a correctly-formatted placeholder lets the application start normally and all place lookups transparently route through the Tavily fallback.

### Model Config — `config/config.yaml`

Non-secret configuration lives in YAML. To change the model:

```yaml
llm:
  groq:
    provider: "groq"
    model_name: "llama-3.3-70b-versatile"   # edit here only
```

To switch provider, change the argument in [`main.py`](main.py):
```python
graph = GraphBuilder(model_provider="openai")
```

### Configuration Philosophy

| Kind | Location | Rationale |
|---|---|---|
| **Secrets** | `.env` (gitignored) | Must never enter version control |
| **Tunables** | `config/config.yaml` | Reviewable in pull requests, changeable without code edits |
| **Prompts** | `prompt_library/prompt.py` | Versioned with code, because prompts are logic |

---

## 12. API Reference

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
{ "error": "<error detail>" }
```

**Schema** — enforced by Pydantic:
```python
class QueryRequest(BaseModel):
    question: str
```
A malformed body returns **422 Unprocessable Entity** automatically.

### Interactive Docs

FastAPI generates OpenAPI documentation automatically:
- **Swagger UI** → `http://localhost:8000/docs`
- **ReDoc** → `http://localhost:8000/redoc`

### Startup Behaviour

The agent graph is built **once** in a FastAPI `lifespan` context manager rather than per request:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global react_app
    graph = GraphBuilder(model_provider="groq")
    react_app = graph()
    _dump_graph_png(react_app)
    yield
```

This removes graph-construction cost from the request path entirely and surfaces configuration errors at boot rather than on first use. The diagram render is wrapped in `try/except` so a transient network failure there cannot block startup.

---

## 13. Quality & Verification

### Verified Behaviour

The following properties are verified and hold in the current build:

| Property | Verification |
|---|---|
| All third-party imports resolve on LangChain 1.x / LangGraph 1.x | 8/8 import checks pass |
| Graph compiles with all 10 tools bound | Nodes: `['__start__','agent','tools','__end__']` |
| Configuration loads independently of working directory | Verified with CWD set outside the project tree |
| Budget arithmetic is numerically correct under type coercion | `multiply("100", 3) == 300.0` |
| Missing credentials raise a named, actionable error at boot | `ValueError: GROQ_API_KEY is not set` |
| Graph is constructed once, not per request | Zero rebuilds measured across repeated requests |
| Place-search fallback engages on both failure modes | Exception path and empty-result path both route to Tavily |

### Test Strategy

The `utils/` layer is deliberately free of LangChain imports precisely so that the majority of logic is testable as plain Python with no framework scaffolding. The intended suite follows the testing pyramid:

```
tests/
├── unit/
│   ├── test_expense_calculator.py   # pure functions
│   ├── test_config_loader.py        # CWD independence, missing-file error
│   └── test_currency_converter.py   # URL construction, error paths
├── integration/
│   ├── test_tools.py                # each @tool with mocked HTTP
│   ├── test_fallback.py             # Tavily engages on failure and on empty result
│   └── test_graph.py                # graph compiles; agent→tools→agent cycle runs
└── e2e/
    └── test_api.py                  # FastAPI TestClient with the LLM mocked
```

All network boundaries are mocked, so the suite never consumes metered API quota.

### Evaluating Generative Output

Generative output cannot be asserted by string equality, so quality is measured differently from conventional software:

| Technique | What it checks |
|---|---|
| **Mocked LLM** | Deterministic control flow — routing, state transitions, tool wiring |
| **Structural assertions** | Required sections present; budget totals consistent with their parts |
| **Tool-call trace assertions** | Did a weather question actually invoke `get_weather_forecast`? |
| **Trajectory evaluation** | Did the agent select the right tools in a sensible order? |
| **LLM-as-judge** | Scores helpfulness, specificity, and faithfulness against a rubric |
| **Faithfulness checking** | Does every factual claim trace back to a tool observation? |

---

## 14. Design Decisions & Trade-offs

| Decision | Alternative | Rationale | Trade-off accepted |
|---|---|---|---|
| **LangGraph** for orchestration | Hand-rolled `while` loop | Cycles, checkpointing, streaming, and visualisation as first-class features | Framework dependency |
| **Arithmetic as tools** | Let the LLM compute | Deterministic, auditable budgets | Additional round-trips |
| **`utils/` free of LangChain** | Tools call APIs directly | Framework-swappable, unit-testable business logic | One extra layer of indirection |
| **Single agent, many tools** | Multi-agent (planner / researcher / writer) | Substantially simpler and sufficient for this scope | Less role specialisation |
| **Graph built at startup** | Built per request | Removes construction cost from every request | Configuration errors surface at boot |
| **Groq as default** | OpenAI as default | Lower latency, which compounds across an agent's sequential calls | Smaller model catalogue |
| **Streamlit frontend** | React / Next.js | No JavaScript toolchain; rapid iteration | Less granular UX control |
| **Two-tier search fallback** | Google Places only | Resilient to quota exhaustion and provider outages | A second API key to manage |
| **YAML + `.env` split** | All environment variables | Tunables reviewable in pull requests | Two configuration sources |
| **Fail-fast on missing keys** | Lazy failure at call time | Clear, actionable errors at boot | Startup requires valid configuration |

---

## 15. Roadmap

### Engineering Foundations
- [ ] `pytest` suite across unit, integration, and end-to-end tiers
- [ ] Structured logging via `structlog` in the `logger/` module
- [ ] Custom exception hierarchy in the `exception/` module
- [ ] `/health` endpoint for readiness and liveness probes
- [ ] Wire `save_document()` into the response path for server-side export

### Infrastructure
- [ ] Multi-stage `Dockerfile` with a non-root user and healthcheck
- [ ] `docker-compose.yml` orchestrating API, UI, PostgreSQL, and Redis
- [ ] GitHub Actions CI — `ruff`, `black --check`, `mypy`, `pytest --cov`, `pip-audit`
- [ ] Redis caching with per-source TTLs (weather 1h, places 7d, FX 1h)
- [ ] Rate limiting and API-key authentication
- [ ] Environment-specific CORS allowlists

### Observability
- [ ] LangSmith or OpenTelemetry tracing across every LLM call and tool invocation
- [ ] Prometheus metrics — latency percentiles, tool error rates, tokens and cost per request, fallback rate, agent loop depth
- [ ] Grafana dashboards and alerting thresholds

### Evaluation
- [ ] Golden dataset of 50–100 representative queries
- [ ] LLM-as-judge scoring with a regression gate on pull requests
- [ ] Faithfulness checks tying every claim to a tool observation

### Product
- [ ] PostgreSQL checkpointer for durable multi-turn conversations
- [ ] Response streaming via Server-Sent Events
- [ ] PDF itinerary export
- [ ] Flight and hotel booking integrations (Amadeus, Skyscanner)
- [ ] User accounts and saved trips
- [ ] Multi-city and multi-currency trip planning

### Planned Container Build

The intended containerisation approach, for reference:

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

### Scaling Considerations

| Concern | Current design | Planned approach |
|---|---|---|
| Requests take 20–90s | Synchronous | Job queue (Celery + Redis) with polling or WebSocket progress |
| Single process | One Uvicorn worker | Multiple workers behind a load balancer; Kubernetes HPA |
| No token streaming | Response returned whole | `graph.astream()` → Server-Sent Events |
| Agent loop depth | Uncapped | `recursion_limit` on invoke to bound worst-case cost |

---

## 16. Troubleshooting

| Symptom | Cause | Resolution |
|---|---|---|
| `ValidationError: Invalid API key provided` at startup | `GPLACES_API_KEY` missing or not in `AIza...` format | Provide a valid key or a correctly-formatted placeholder |
| `ValueError: GROQ_API_KEY is not set` | Empty key in `.env` | Add a valid Groq key |
| `401 Invalid API Key` in the response | Key present but incorrect | Regenerate at the provider console |
| Streamlit shows `Connection refused` | Backend not running | Start `uvicorn` first, on port 8000 |
| `FileNotFoundError: config.yaml` | Config file missing | Restore `config/config.yaml` |
| Startup log mentions a diagram render failure | Mermaid render service unreachable | Non-fatal by design; startup continues normally |
| First response takes 20–90s | Expected behaviour | Multiple sequential LLM and API round-trips |
| `google.api_core` Python 3.10 EOL warning | Upstream deprecation notice | Informational; optionally move to Python 3.11 |

---

## 17. Glossary

| Term | Meaning |
|---|---|
| **Agent** | An LLM that takes actions via tools in a loop, deciding its own steps |
| **ReAct** | *Reasoning + Acting* — alternating thought and tool use |
| **Tool / function calling** | Exposing a function's schema so the model can request its execution |
| **LangGraph** | Framework for cyclic, stateful agent graphs |
| **StateGraph** | A graph whose nodes read and mutate shared state |
| **Reducer** | Function defining how a state update merges (here, append) |
| **ToolNode** | Prebuilt node that executes tool calls from the last AI message |
| **`tools_condition`** | Prebuilt router: to tools if calls exist, else to END |
| **Checkpointer** | Persists graph state so conversations survive restarts |
| **Grounding** | Anchoring generated text in retrieved real data |
| **Hallucination** | Fluent, confident, factually incorrect model output |
| **RAG** | Retrieval-Augmented Generation — retrieval from an owned index |
| **LLM gateway** | Abstraction layer over model vendors |
| **LPU** | Groq's Language Processing Unit — inference-optimised hardware |
| **Token** | Sub-word unit; the billing and context unit for LLMs |
| **Context window** | Maximum tokens a model can consider at once |
| **System prompt** | Instruction defining the model's role and constraints |
| **ASGI** | Asynchronous Server Gateway Interface |
| **Lifespan** | FastAPI startup and shutdown hook |
| **CORS** | Browser policy governing cross-origin requests |
| **Graceful degradation** | Continuing with reduced capability instead of failing outright |
| **Circuit breaker** | Stops calling a failing dependency for a cooldown period |
| **LLM-as-judge** | Using a model to score another model's output |
| **Golden dataset** | Curated test inputs with expected quality criteria |
| **Trajectory evaluation** | Scoring an agent's *sequence of tool calls*, not just its answer |
| **Observability** | Logs, metrics, and traces taken together |
| **p95 latency** | The latency below which 95% of requests complete |

---

## 📚 Further Documentation

Extended technical notes — architecture rationale, implementation status detail, engineering case studies, and concept references — are maintained in [`docs/PROJECT_NOTES.md`](docs/PROJECT_NOTES.md).

## 🙏 Acknowledgements

Built with [LangChain](https://python.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/). Inference by [Groq](https://groq.com/). Data from [OpenWeatherMap](https://openweathermap.org/), [Google Places](https://developers.google.com/maps/documentation/places/web-service), [Tavily](https://tavily.com/), and [exchangerate-api](https://www.exchangerate-api.com/).

## 📄 License

Released under the MIT License. Add a `LICENSE` file before publishing.

---

<div align="center">

**Navora AI** · *Agentic travel planning, grounded in real data.*

Built by **Akshat Gupta**

</div>
