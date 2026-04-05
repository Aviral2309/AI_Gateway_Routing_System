# AI Gateway Routing system

> An intelligent multi-LLM orchestration gateway that routes queries to the optimal model using ML-based classification, a multi-agent execution pipeline, Redis caching, and PostgreSQL-backed analytics — reducing inference cost while maintaining response quality.

---

## Why This Exists

Every LLM application has the same hidden inefficiency: every query, regardless of complexity, gets routed to the same model and charged at the same rate. A question like *"What is the capital of France?"* costs the same as *"Design a distributed rate limiter with Redis"* — even though the first needs 50 tokens and the second needs 2000.

**Nexus solves this.** It classifies every query before touching any LLM, routes it to the cheapest model that can handle it well, caches repeated answers, and for genuinely complex queries — decomposes them into sub-tasks that run in parallel across different models. The result: lower cost, faster responses, and better answers for hard problems.

```
Simple query  → Gemini Flash    → answer in 400ms   
Medium query  → Gemini Flash    → answer in 800ms   
Complex query → Agent Pipeline  → 3 tasks in 1.2s 
Same query x2 → Redis Cache     → answer in 0ms    
```

---

## Architecture

```
                         ┌─────────────────────────────────────────┐
                         │           Nexus LLM Router               │
                         │                                          │
  POST /chat ──────────▶ │  1. ML Classifier (RandomForest)        │
                         │     ↓ tier + intent                      │
                         │  2. Redis Cache Check                    │
                         │     ↓ miss                               │
                         │  3. Routing Engine                       │
                         │     ↓ cost / budget / strategy           │
                         │                                          │
                         │  ┌─ Simple / Medium ─┐                  │
                         │  │  Direct LLM call  │                  │
                         │  └───────────────────┘                  │
                         │                                          │
                         │  ┌─ Complex ─────────────────────────┐  │
                         │  │  Agent Pipeline                   │  │
                         │  │  PlannerAgent  → sub-tasks        │  │
                         │  │  ExecutorAgent → parallel run     │  │
                         │  │  SynthesizerAgent → merge         │  │
                         │  └───────────────────────────────────┘  │
                         │                                          │
                         │  4. LLM-as-Judge Evaluation             │
                         │  5. PostgreSQL Tracking                  │
                         │  6. Response + full metadata             │
                         └─────────────────────────────────────────┘
                                         │
                              ┌──────────┴──────────┐
                              ▼                     ▼
                         Gemini Flash          Ollama (local)
                    Gemini flash lite          codellama / llama3.1:8b
```

---

## Features

### ML-Based Query Classification
- **RandomForest classifier** trained on labeled query examples
- **12 engineered features** per query: token count, code block detection, math detection, clause depth, 4 intent keyword scores — all extracted locally in under 1ms
- **3-tier output**: `simple` / `medium` / `complex`
- **5 intent labels**: `code`, `reasoning`, `factual`, `creative`, `general`
- Retrains automatically on your own logged data via `scripts/retrain_classifier.py`

### Intelligent Routing Engine
- **Routing table** mapping `(tier, intent)` → ordered model list
- **4 routing strategies**: `balanced`, `cost_optimized`, `quality_first`, `local_first`
- **Budget enforcement**: per-session spend limits, auto-fallback to free Ollama when budget exhausted
- **Automatic retry + fallback**: if primary model fails, tries next in the list
- **Health-check aware**: only routes to models that are actually reachable

### Agentic AI Pipeline
- **PlannerAgent**: detects compound queries (implement + explain, A vs B, multi-requirement) and decomposes into 2–4 focused sub-tasks
- **ExecutorAgent**: runs independent sub-tasks in **parallel** with `asyncio.gather`, passes dependency results as context to downstream tasks
- **SynthesizerAgent**: merges partial results into a single coherent answer using an LLM
- Activated automatically for `complex` tier queries; can be forced on/off per request

### Redis Semantic Cache
- SHA-256 normalized cache keys (lowercased + stripped queries)
- Cache hits return in **0ms at $0 cost**
- Configurable TTL (default 1 hour)
- Hit/miss stats exposed at `/metrics`
- Graceful fallback to no-cache if Redis is unavailable

### PostgreSQL Persistence
- Every request logged with full metadata: tier, intent, model, tokens, cost, latency, quality score, cache hit, agent decomposition
- Session-level budget tracking that **survives server restarts**
- Analytics endpoints: cost over time, per-model breakdown, tier/intent distribution
- Async writes via asyncpg — zero impact on response latency

### LLM-as-Judge Evaluation
- Uses Gemini Flash to score every response 0.0–1.0 after it's generated
- Scores feed into per-model quality metrics over time
- Heuristic fallback (rule-based scoring) when API is unavailable
- Simple factual queries skip evaluation to save tokens

### Production Infrastructure
- **Sliding window rate limiter** (Redis-backed, in-memory fallback)
- **Graceful degradation**: works without Redis, without PostgreSQL, without either
- **CORS middleware** for frontend integration
- **Structured logging** via structlog

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| API | FastAPI + Uvicorn | Async-native, auto-docs, fast |
| ML | scikit-learn RandomForest | Fast inference, interpretable, no GPU needed |
| LLMs | Gemini 1.5 Flash/Pro, Ollama | Cloud + local coverage |
| Cache | Redis + aioredis | Sub-millisecond cache hits |
| Database | PostgreSQL + SQLAlchemy async + asyncpg | Reliable persistence, full analytics |
| Agents | Custom pipeline (asyncio) | No framework lock-in, full control |
| HTTP | httpx async | Non-blocking LLM calls |
| Config | pydantic-settings | Type-safe env vars |
| Logging | structlog | Structured JSON logs |

---

## Quick Start

### Prerequisites

- Python 3.11 or 3.12
- Docker (for PostgreSQL + Redis) or install manually
- Gemini API key — free at [aistudio.google.com](https://aistudio.google.com)
- Ollama (optional) — install from [ollama.com](https://ollama.com)

### 1. Clone

```bash
git clone https://github.com/yourusername/nexus-llm-router.git
cd nexus-llm-router
```

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Start infrastructure

```bash
docker-compose up -d
# Starts PostgreSQL on :5432 and Redis on :6379
```

### 4. Configure

```bash
cp .env.example .env
```

Edit `.env` — only one line is required:
```env
GEMINI_API_KEY=your_key_here
```

Everything else has working defaults.

### 5. Initialize database

```bash
python scripts/setup_db.py
```

### 6. Run

```bash
uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs** — full interactive Swagger UI.

---

## Usage

### Standard routing

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
```

```json
{
  "content": "The capital of France is Paris.",
  "model_used": "gemini-2.5-flash",
  "from_cache": false,
  "was_decomposed": false,
  "tier": "simple",
  "intent": "factual",
  "complexity_score": 0.12,
  "routing_strategy": "balanced",
  "latency_ms": 412.3,
  "cost_usd": 0.0000021,
  "quality_score": 0.97,
  "success": true,
  "agent_steps": null
}
```

### Same query again — Redis cache hit

```json
{
  "content": "The capital of France is Paris.",
  "model_used": "gemini-2.5-flash[cached]",
  "from_cache": true,
  "latency_ms": 0.0,
  "cost_usd": 0.0
}
```

### Complex query — agent pipeline kicks in

```bash
curl -X POST http://localhost:8000/chat \
  -d '{"query": "Design a URL shortener with Redis caching, rate limiting, and analytics, then implement the core components"}'
```

```json
{
  "content": "## URL Shortener System Design\n\n### Architecture...",
  "model_used": "ollama-local, gemini-1.5-flash",
  "was_decomposed": true,
  "tier": "complex",
  "agent_steps": [
    {"task_id": "a1b2", "description": "High-level architecture...", "model_used": "ollama-local", "status": "success", "latency_ms": 1823.4},
    {"task_id": "c3d4", "description": "Implementation details...",  "model_used": "ollama-local", "status": "success", "latency_ms": 2104.1}
  ]
}
```

### Override routing strategy

```bash
# Cheapest model always
curl -X POST http://localhost:8000/chat \
  -d '{"query": "Explain microservices", "strategy": "cost_optimized"}'

# Best model always
curl -X POST http://localhost:8000/chat \
  -d '{"query": "Explain microservices", "strategy": "quality_first"}'

# Prefer local Ollama (privacy / offline)
curl -X POST http://localhost:8000/chat \
  -d '{"query": "Explain microservices", "strategy": "local_first"}'

# Force agents on or off
curl -X POST http://localhost:8000/chat \
  -d '{"query": "Explain microservices", "use_agents": false}'
```

### Debug classification

```bash
curl -X POST http://localhost:8000/classify \
  -d '{"query": "Implement a distributed rate limiter with Redis"}'
```

```json
{
  "tier": "complex",
  "intent": "code",
  "complexity_score": 0.71,
  "confidence": 0.88,
  "will_use_agents": true
}
```

### Analytics

```bash
# Real-time memory stats (always fast)
curl http://localhost:8000/metrics

# Full PostgreSQL analytics
curl "http://localhost:8000/metrics?source=db"

# Hourly cost for last 48 hours
curl "http://localhost:8000/analytics/cost?hours=48"

# Recent request log
curl "http://localhost:8000/analytics/requests?limit=20"

# System health
curl http://localhost:8000/health
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Route a query through the full pipeline |
| `POST` | `/classify` | Classify a query without calling any LLM |
| `GET` | `/metrics` | In-memory analytics |
| `GET` | `/metrics?source=db` | Full PostgreSQL analytics |
| `GET` | `/analytics/cost?hours=N` | Hourly cost breakdown |
| `GET` | `/analytics/requests?limit=N` | Recent request log |
| `GET` | `/health` | All component health status |
| `GET` | `/models` | Registered models + live health |
| `DELETE` | `/cache` | Flush Redis cache |

### `/chat` request body

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | required | The user query (1–8000 chars) |
| `history` | array | `[]` | Prior conversation messages |
| `system_prompt` | string | null | Optional system prompt |
| `session_id` | string | auto | For budget tracking across requests |
| `strategy` | string | `balanced` | `balanced` / `cost_optimized` / `quality_first` / `local_first` |
| `temperature` | float | `0.7` | LLM temperature (0.0–1.0) |
| `max_tokens` | int | `1024` | Max output tokens |
| `use_agents` | bool | null | `null`=auto, `true`=force on, `false`=force off |

---

## Configuration

All settings via `.env`:

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_DEFAULT_MODEL` | `llama3` | Model to pull/use |
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `CACHE_TTL_SECONDS` | `3600` | Cache entry lifetime |
| `CACHE_ENABLED` | `true` | Enable Redis cache |
| `DEFAULT_BUDGET_USD` | `5.00` | Per-session budget cap |
| `MAX_RETRIES` | `2` | Fallback attempts per request |
| `ENABLE_EVALUATION` | `true` | LLM-as-judge scoring |
| `ENABLE_AGENTS` | `true` | Agentic pipeline for complex queries |
| `AGENT_MAX_STEPS` | `5` | Max sub-tasks per query |
| `RATE_LIMIT_REQUESTS` | `100` | Max requests per window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate limit window |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` |

---

## Project Structure

```
nexus-llm-router/
├── app/
│   ├── main.py                        # FastAPI app + all routes
│   │
│   ├── classifier/
│   │   ├── feature_extractor.py       # 12-feature local extractor (<1ms, no API)
│   │   └── query_classifier.py        # RandomForest — simple / medium / complex
│   │
│   ├── router/
│   │   ├── llm_router.py              # Central orchestrator
│   │   └── routing_rules.py           # Routing table, budget logic, strategies
│   │
│   ├── models/
│   │   ├── base_model.py              # Abstract interface all adapters implement
│   │   ├── gemini_model.py            # Google Gemini Flash + Pro
│   │   └── ollama_model.py            # Ollama local models
│   │
│   ├── agents/
│   │   ├── base_agent.py              # SubTask, AgentTrace data structures
│   │   ├── planner_agent.py           # Decomposes complex queries into sub-tasks
│   │   ├── executor_agent.py          # Parallel sub-task execution with dependencies
│   │   ├── synthesizer_agent.py       # Merges partial results into final answer
│   │   └── orchestrator.py            # Coordinates the full agent pipeline
│   │
│   ├── evaluator/
│   │   └── response_evaluator.py      # LLM-as-judge quality scoring (0–1)
│   │
│   ├── cache/
│   │   └── redis_cache.py             # Redis cache with normalized keys
│   │
│   ├── db/
│   │   ├── models.py                  # SQLAlchemy ORM (request_logs, session_budgets)
│   │   ├── session.py                 # Async engine + session factory
│   │   └── repository.py             # All DB queries in one place
│   │
│   ├── middleware/
│   │   └── rate_limiter.py            # Sliding window rate limiter
│   │
│   └── tracking/
│       └── tracker.py                 # Dual-write: PostgreSQL + in-memory fallback
│
├── config/
│   └── settings.py                    # Pydantic settings from env vars
│
├── tests/
│   └── test_nexus.py                  # 35 unit tests — no API keys needed
│
├── scripts/
│   ├── setup_db.py                    # One-time database initialization
│   └── retrain_classifier.py         # Retrain ML model on real logged data
│
├── docker-compose.yml                 # PostgreSQL + Redis
├── requirements.txt
└── .env.example
```

---

## Adding a New LLM

The adapter interface makes adding any new model a single file. Here is a GPT-4o example:

```python
# app/models/openai_model.py
import time
import httpx
from app.models.base_model import BaseLLM, LLMResponse, Message
from config.settings import settings

class OpenAIModel(BaseLLM):
    model_id     = "gpt-4o-mini"
    display_name = "GPT-4o Mini"
    max_context_tokens = 128_000

    async def complete(self, messages, system_prompt=None, temperature=0.7, max_tokens=1024):
        t0 = time.monotonic()
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            )
            r.raise_for_status()
            data = r.json()
        latency_ms = (time.monotonic() - t0) * 1000
        content    = data["choices"][0]["message"]["content"]
        usage      = data["usage"]
        return LLMResponse(
            content=content, model_id=self.model_id,
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
            latency_ms=round(latency_ms, 1),
            cost_usd=self.estimate_cost(usage["prompt_tokens"], usage["completion_tokens"]),
            success=True,
        )

    def estimate_cost(self, inp, out):
        return inp / 1000 * 0.00015 + out / 1000 * 0.0006

    async def health_check(self):
        r = await self.complete([Message(role="user", content="hi")], max_tokens=5)
        return r.success
```

Then register it in `app/router/llm_router.py`:
```python
from app.models.openai_model import OpenAIModel

def _register_models(self):
    self.register_model(GeminiModel("gemini-1.5-flash"))
    self.register_model(OllamaModel())
    self.register_model(OpenAIModel())  # add this
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

35 tests across all layers: classifier, routing rules, tracker, Redis cache, planner agent, and evaluator. All run without a live database, Redis instance, or API key.

---

## Setup Without Docker

### PostgreSQL

**Windows:** Download from [postgresql.org/download/windows](https://www.postgresql.org/download/windows/). After install, open pgAdmin or SQL Shell:
```sql
CREATE DATABASE nexus_router;
```
Update `.env`: `DATABASE_URL=postgresql+asyncpg://postgres:YOUR_PASSWORD@localhost:5432/nexus_router`

**macOS:** `brew install postgresql@16 && brew services start postgresql@16`

**Linux:** `sudo apt install postgresql && sudo systemctl start postgresql`

Then run: `python scripts/setup_db.py`

### Redis

**Windows:** Download from [github.com/tporadowski/redis/releases](https://github.com/tporadowski/redis/releases), run `redis-server.exe`.

**macOS:** `brew install redis && brew services start redis`

**Linux:** `sudo apt install redis-server && sudo systemctl start redis`

Redis is optional — the app automatically falls back to in-memory rate limiting and disables caching if Redis is unavailable.

### Ollama

```bash
# Install: https://ollama.com
ollama pull llama3        # general purpose
ollama pull codellama     # better for code queries
ollama pull mistral       # fast and capable
```

---

## Retraining the Classifier

After collecting real traffic (at least 30 requests logged), retrain:

```bash
python scripts/retrain_classifier.py
```

The script pulls logged queries from PostgreSQL, weights training examples by quality score (better-rated responses = stronger signal), trains a new RandomForest, runs 5-fold cross-validation, and saves to `classifier_model.pkl`. Restart the server to load it. Accuracy improves significantly with 500+ real examples.

---

## Author

Aviral Mittal
