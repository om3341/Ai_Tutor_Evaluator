# AI Teacher Benchmark Platform

AI Teacher Benchmark Platform is a FastAPI and Streamlit application for evaluating educational LLMs as K-12 tutoring systems. It benchmarks one model at a time across a curated teaching dataset, stores responses and judge scores, and produces analytics for pedagogy, multilingual quality, hallucination risk, latency, and overall learning effectiveness.

This project is designed for educational AI evaluation, not chatbot battle ranking. The workflow is single-model benchmarking: load one model, run dataset prompts, collect responses, judge each response, aggregate scores, and review benchmark reports.

## Features

- Single-model benchmark workflow with one active model at a time
- Streamlit dashboard for loading models, stepping through benchmark items, editing prompts, generating responses, and evaluating collected outputs
- FastAPI backend with typed Pydantic schemas
- Gemini judge integration for single-response educational evaluation
- PostgreSQL persistence with async SQLAlchemy and Alembic migrations
- Benchmark run history and per-item score storage
- Multilingual analytics for Hinglish, Hindi, Marathi, code-switching, transliteration, and regional-language quality
- Latency analytics for generation latency, judge latency, averages, p95-style summaries, and trends
- Hallucination tracking for fabricated facts, unsafe explanations, overconfidence, and misleading educational content
- Insights engine for strengths, weaknesses, bottlenecks, and benchmark summaries
- Local model adapters for vLLM/OpenAI-compatible servers and Ollama-compatible models
- Model autostart hooks for remote GPU servers, SSH tunnels, vLLM, and Ollama
- Plotly-powered visual dashboards

## Evaluation Metrics

The judge scores each response across seven core categories:

- `correctness`
- `teaching_quality`
- `adaptation`
- `emotional_intelligence`
- `multilingual_quality`
- `hallucination_risk`
- `conversation_quality`

The platform also computes aggregate benchmark scores, safety classifications, learning-effectiveness labels, latency totals, and dashboard-ready analytics.

## Architecture

```text
backend/
  api/                  FastAPI routers for benchmarks, analytics, generation, leaderboard
  analytics/            Multilingual, latency, hallucination, and insight calculators
  database/             SQLAlchemy models, CRUD helpers, Alembic migrations
  models/               Local/remote model clients
  prompts/              Gemini judge and benchmark report prompts
  schemas/              Pydantic request/response contracts
  services/             Benchmark orchestration, analytics, model process manager

frontend/
  analytics/            Streamlit analytics dashboards
  api_client.py         Backend client helpers
  app.py                Main Streamlit benchmark UI
  components.py         Reusable UI blocks
  visualizations.py     Plotly chart helpers
```

Root `app.py` launches the Streamlit frontend. `backend/main.py` launches the FastAPI backend.

## Benchmark Flow

1. Select a model in the Streamlit dashboard.
2. Click **Load Model**.
3. Load benchmark dataset items.
4. Review or edit each prompt.
5. Generate a model response.
6. Review or edit the response.
7. Save the item and move to the next one.
8. Evaluate collected responses with the Gemini judge.
9. Review aggregate benchmark results, item-level scores, and analytics dashboards.

This staged flow helps avoid unnecessary judge calls and works better with rate-limited judge APIs.

## Setup

Create a virtual environment and install dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create your environment file:

```bash
cp .env.example .env
```

Edit `.env` with your local settings:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
DATABASE_URL=postgresql+asyncpg://llm_compare:llm_compare@127.0.0.1:5432/llm_compare
QWEN_BASE_URL=http://your-vllm-host:8001/v1
GEMMA_BASE_URL=http://127.0.0.1:11435
```

Do not commit `.env`. The repository includes `.env.example` for safe configuration examples.

## Database

Create a PostgreSQL database and user:

```bash
createdb llm_compare
psql -d llm_compare -c "CREATE USER llm_compare WITH PASSWORD 'llm_compare';"
psql -d llm_compare -c "GRANT ALL PRIVILEGES ON DATABASE llm_compare TO llm_compare;"
psql -d llm_compare -c "GRANT ALL ON SCHEMA public TO llm_compare;"
```

Run migrations:

```bash
alembic upgrade head
```

## Running The App

Start the FastAPI backend:

```bash
uvicorn backend.main:app --reload
```

Start the Streamlit frontend in another terminal:

```bash
streamlit run app.py
```

The frontend expects the backend at `http://127.0.0.1:8000` by default.

## Model Servers

Qwen can be served through vLLM:

```bash
vllm serve Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 8001 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --trust-remote-code
```

Ollama models can be exposed locally or through an SSH tunnel:

```bash
ssh -N -L 11435:127.0.0.1:11434 user@your-gpu-host
```

The backend can also autostart configured model commands using:

```bash
MODEL_AUTOSTART_ENABLED=true
QWEN_START_COMMAND=...
GEMMA_START_COMMAND=...
LLAMA_START_COMMAND=...
```

Use SSH keys or another non-interactive authentication method for autostart commands.

## Key API Endpoints

Health:

```text
GET /health
```

Benchmarking:

```text
GET  /benchmarks/health
GET  /benchmarks/dataset
GET  /benchmarks/model
POST /benchmarks/model/load
POST /benchmarks/model/unload
POST /benchmarks/evaluate-collected
GET  /benchmarks/runs
GET  /benchmarks/runs/{run_id}
```

Model generation:

```text
POST /models/qwen/generate
POST /models/gemma/generate
POST /models/llama/generate
```

Analytics:

```text
GET /analytics
GET /analytics/multilingual
GET /analytics/latency
GET /analytics/latency/models
GET /analytics/latency/history
GET /analytics/hallucinations
GET /analytics/insights
```

## Example Generation Request

```bash
curl -X POST http://127.0.0.1:8000/models/qwen/generate \
  -H "Content-Type: application/json" \
  -d '{
    "student_prompt": "Explain photosynthesis in Hinglish for class 6 in 3 short points.",
    "student_level": "Class 6",
    "language": "Hinglish",
    "model_name": "Qwen 3.8B"
  }'
```

## Example Benchmark Evaluation Request

```bash
curl -X POST http://127.0.0.1:8000/benchmarks/evaluate-collected \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "Qwen 3.8B",
    "dataset_name": "k12_teacher_core_v1",
    "items": [
      {
        "item_id": "k12_hinglish_science_001",
        "student_prompt": "Explain photosynthesis in Hinglish for class 6 in 3 short points.",
        "student_level": "Class 6",
        "language": "Hinglish",
        "subject": "Science",
        "rubric": "Reward accuracy, simple language, and natural Hinglish.",
        "response": "Plants sunlight, water, and carbon dioxide use karke apna food banate hain. Is process ko photosynthesis kehte hain. Isme oxygen bhi release hoti hai.",
        "generation_latency_ms": 1200
      }
    ]
  }'
```

## Notes

- Gemini API quota limits can affect benchmark evaluation. The staged workflow lets you generate all model responses first and judge them later.
- Large local models need enough free GPU memory. If model loading fails, check `logs/model_processes/`.
- `.env`, logs, model weights, and virtual environments are ignored by Git.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
