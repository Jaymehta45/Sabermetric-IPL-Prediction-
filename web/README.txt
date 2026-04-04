IPLpred web UI (FastAPI + Jinja2 + static CSS/JS)

Install (from repo root):
  pip install -r requirements-web.txt

Run (from repo root; restart after pulling UI changes):
  uvicorn web.dashboard:app --reload --host 0.0.0.0 --port 8000

Static CSS/JS use ?v=… on the URL so the browser refetches after updates. If the UI still looks old, hard-refresh (Cmd+Shift+R).

Open:
  http://127.0.0.1:8000

Pages (read-only — no match input):
  /           Dashboard: stats + recent pred vs actual rows from prediction_log.csv
  /history    Full prediction log table, filters, charts

Pre-match predictions: use predict_match_outcomes.py and scripts/log_prediction.py.

API:
  GET /api/prediction-log — JSON rows from data/processed/prediction_log.csv
  GET /health — status

Requires data/processed and models/ as in the main pipeline (run from repo root).
