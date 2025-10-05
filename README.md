
# 🧬 DoubleHelix v0.8.1 — Emergence Enhancements
Adds memory consolidation & forgetting, curiosity/surprise, multi-scale loops, prompt evolution, pattern detection, and multi-agent perspectives.

## Run (local)
### Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Configure (backend/.env)
- Set `DRY_RUN=false` and your `OPENAI_API_KEY` to enable real LLM reflection.
- Optional thresholds: CONSOLIDATION_INTERVAL, META_INTERVAL, MEMORY_MAX, SURPRISE_THRESHOLD, PATTERN_THRESHOLD.

## Useful endpoints
- `GET /health`
- `POST /chat` (body: `{ "prompt":"...", "use_perspectives": true|false }`)
- `POST /tick` — planner + multi-scale operations
- `GET /emergence` — metrics view
- `POST /meta_reflect` — on-demand meta analysis
- `GET /policy`, `GET /goals`
