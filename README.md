# Palm Mind RAG Backend

FastAPI backend with a document ingestion API and a conversational RAG API
(custom pipeline, Redis memory, interview booking).

## Setup
```bash
cp .env.example .env        # add your LLM_API_KEY
docker compose up -d        # Qdrant + Redis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Docs: http://localhost:8000/docs

(More sections coming: architecture, endpoints, curl examples.)
