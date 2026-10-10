# Engagement Tracker

Audience-intelligence tool that collects public feedback about a product, classifies sentiment, and produces per-country readouts to support product-launch decisions.

## Prerequisites

- Python 3.11+
- Node.js 18+

## Backend setup

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

cp .env.example .env       # fill in your API keys
pip install -r requirements.txt
uvicorn main:app --reload
```

The API starts at `http://localhost:8000`.

## Frontend setup

```bash
cd frontend
cp .env.example .env       # default: VITE_API_URL=http://localhost:8000
npm install
npm run dev
```

The dev server starts at `http://localhost:5173`.

## Environment variables

### Backend (`backend/.env`)

| Variable | Purpose |
|---|---|
| `NEWS_API_KEY` | NewsAPI.org key (required for search) |
| `GNEWS_API_KEY` | GNews API key |
| `REDDIT_CLIENT_ID` | Reddit app client ID |
| `REDDIT_CLIENT_SECRET` | Reddit app secret |
| `REDDIT_USER_AGENT` | Reddit API user-agent string |
| `YOUTUBE_API_KEY` | YouTube Data API key |
| `LLM_MODEL` | LLM model identifier |
| `LLM_API_KEY` | LLM provider API key |
| `DATABASE_URL` | Database connection string |

### Frontend (`frontend/.env`)

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_URL` | `http://localhost:8000` | Backend API base URL |
