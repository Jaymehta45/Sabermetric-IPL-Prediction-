IPLpred web UI (FastAPI + Jinja2 + static CSS/JS)

Install (from repo root):
  pip install -r requirements-web.txt

Run (from repo root; restart after pulling UI changes):
  uvicorn web.app:app --reload --host 0.0.0.0 --port 8000

Static CSS/JS use ?v=… on the URL so the browser refetches after updates. If the UI still looks old, hard-refresh (Cmd+Shift+R).

Open:
  http://127.0.0.1:8000

Pages:
  /           Dashboard + recent prediction log
  /predict    Form → POST /api/predict
  /history    Full prediction_log.csv table + filter

Requires data/processed and models/ as in the main pipeline (same working directory as repo root).
