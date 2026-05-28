# AI Teacher Benchmark Arena Backend

Clean FastAPI MVP for pairwise evaluation of multilingual AI teacher responses.

The backend accepts a student prompt and two candidate responses, asks Gemini Flash
to judge them pairwise, validates the returned JSON, and returns winner, scores,
reasoning, confidence, and latency metrics.

## What is included

- FastAPI backend
- `GET /health`
- `POST /evaluate`
- `GET /leaderboard`
- `GET /leaderboard/top`
- `GET /models/{model_name}`
- `GET /evaluations/history`
- `GET /analytics/multilingual`
- `GET /analytics/hallucinations`
- `GET /analytics/latency`
- `GET /analytics/latency/history`
- `GET /analytics/insights`
- Gemini Flash judge integration through `google.generativeai`
- Strict Pydantic request/response schemas
- Pairwise benchmark runner
- PostgreSQL persistence with async SQLAlchemy
- ELO ranking and leaderboard updates
- Multilingual, latency, hallucination, and benchmark insight analytics
- Retry, timeout, logging, and Gemini JSON validation

## What is intentionally not included

- Streamlit frontend
- Docker or deployment infrastructure
- Redis, Celery, async queues
- Authentication
- Voice evaluation

## Local setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set:

```bash
GEMINI_API_KEY=your_real_key
DATABASE_URL=postgresql+asyncpg://llm_compare:llm_compare@127.0.0.1:5432/llm_compare
QWEN_BASE_URL=http://10.240.166.9:8001/v1
QWEN_MODEL_NAME=Qwen/Qwen3-8B
GEMMA_BASE_URL=http://127.0.0.1:11435
GEMMA_MODEL_NAME=gemma4:e4b-it-q8_0
```

Create a local PostgreSQL database:

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

Run locally:

```bash
uvicorn backend.main:app --reload
```

In a second terminal, run the Streamlit dashboard:

```bash
streamlit run app.py
```

The dashboard expects the backend at `http://127.0.0.1:8000` by default. You can
change that URL from the sidebar.

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Run Qwen through vLLM on the GPU server:

```bash
vllm serve Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 8001 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --trust-remote-code
```

Generate a candidate Qwen response through the backend:

```bash
curl -X POST http://127.0.0.1:8000/models/qwen/generate \
  -H "Content-Type: application/json" \
  -d '{
    "student_prompt": "Explain photosynthesis in Hinglish for class 6.",
    "student_level": "Class 6",
    "language": "Hinglish",
    "model_name": "Qwen 3.8B"
  }'
```

If Gemma is running through Ollama on a remote GPU server, create a local tunnel:

```bash
ssh -L 11435:127.0.0.1:11434 btechuser@10.240.166.9
```

Then generate a candidate Gemma response through the backend:

```bash
curl -X POST http://127.0.0.1:8000/models/gemma/generate \
  -H "Content-Type: application/json" \
  -d '{
    "student_prompt": "Explain photosynthesis in Hinglish for class 6.",
    "student_level": "Class 6",
    "language": "Hinglish",
    "model_name": "google/gemma-4-E4B"
  }'
```

Example evaluation:

```bash
curl -X POST http://127.0.0.1:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "student_prompt": "Explain photosynthesis in Hinglish for class 6.",
    "student_level": "Class 6",
    "language": "Hinglish",
    "model_a": "Sarvam 30B",
    "model_b": "Qwen 3.8B",
    "response_a": "Photosynthesis ek process hai jisme plants sunlight, water aur carbon dioxide use karke apna food banate hain. Chlorophyll sunlight capture karta hai, aur plant glucose banata hai. Oxygen side product ke form mein release hoti hai.",
    "response_b": "Plants eat sunlight and turn it into oxygen. This is photosynthesis. It happens in leaves and is important for life."
  }'
```

Example response shape:

```json
{
  "winner": "A",
  "confidence": 0.89,
  "scores": {
    "A": {
      "correctness": 9,
      "teaching_quality": 8,
      "adaptation": 9,
      "emotional_intelligence": 8,
      "multilingual_quality": 9,
      "hallucination_risk": 9,
      "conversation_quality": 8
    },
    "B": {
      "correctness": 6,
      "teaching_quality": 5,
      "adaptation": 6,
      "emotional_intelligence": 5,
      "multilingual_quality": 4,
      "hallucination_risk": 7,
      "conversation_quality": 5
    }
  },
  "evaluation_id": "5e63dc3f-8c1f-43bb-8bf3-96ed126df8c7",
  "model_a": "Sarvam 30B",
  "model_b": "Qwen 3.8B",
  "reasoning": "Response A is more accurate, age-appropriate, and uses clearer Hinglish pedagogy.",
  "latency": {
    "judge_latency_ms": 1420.35,
    "total_latency_ms": 1420.58
  }
}
```

Leaderboard:

```bash
curl http://127.0.0.1:8000/leaderboard
curl http://127.0.0.1:8000/leaderboard/top?limit=3
curl "http://127.0.0.1:8000/models/Sarvam%2030B"
curl http://127.0.0.1:8000/evaluations/history
curl http://127.0.0.1:8000/analytics/multilingual
curl http://127.0.0.1:8000/analytics/hallucinations
curl http://127.0.0.1:8000/analytics/latency
curl http://127.0.0.1:8000/analytics/insights
```

Example leaderboard item:

```json
{
  "model_name": "Sarvam 30B",
  "elo_score": 1015.72,
  "wins": 1,
  "losses": 0,
  "draws": 0,
  "matches_played": 1,
  "win_rate": 100.0,
  "avg_correctness": 9.0,
  "avg_teaching_quality": 8.0,
  "avg_adaptation": 9.0,
  "avg_emotional_intelligence": 8.0,
  "avg_multilingual_quality": 9.0,
  "avg_hallucination_risk": 9.0,
  "avg_conversation_quality": 8.0,
  "updated_at": "2026-05-24T12:00:00Z"
}
```

## Architecture

- `backend/main.py`: FastAPI app, dependency wiring, route-level errors.
- `backend/schemas/`: API, judge, leaderboard, and history validation models.
- `backend/judge.py`: Gemini integration, prompt rendering, retries, timeout, JSON validation.
- `backend/benchmark_runner.py`: pairwise evaluation orchestration and latency metrics.
- `backend/database/`: async SQLAlchemy connection, models, CRUD layer, and Alembic migrations.
- `backend/ranking/`: ELO calculation and leaderboard response helpers.
- `backend/services/ranking_service.py`: transaction boundary for saving evaluations and updating rankings.
- `backend/api/leaderboard.py`: leaderboard, model stats, recent performance, and history endpoints.
- `backend/api/analytics.py`: multilingual, latency, hallucination, and insights endpoints.
- `backend/analytics/`: reusable analytics calculators and insight generation.
- `backend/prompts/judge_prompt.txt`: strict judge rubric and JSON contract.
- `app.py`: root Streamlit entrypoint.
- `frontend/app.py`: dashboard composition, model selection, prompt entry, result rendering.
- `frontend/api_client.py`: FastAPI `/health` and `/evaluate` client.
- `frontend/components.py`: reusable Streamlit UI cards and panels.
- `frontend/visualizations.py`: Plotly radar, bar, latency, confidence, and score charts.
- `frontend/analytics/`: score tables plus multilingual, latency, hallucination, and insights dashboards.
- `frontend/leaderboard.py`: session leaderboard-ready table.

## Dashboard layout

- Sidebar: backend URL, health status, student level, language, model pair, latency inputs.
- Main panel: prompt input, side-by-side editable model responses, evaluate button.
- Results: winner cards, Gemini reasoning, highlighted response cards, interactive Plotly charts.
- Analytics tabs: radar chart, confidence gauge, average score chart, latency chart, performance tables, session leaderboard.
