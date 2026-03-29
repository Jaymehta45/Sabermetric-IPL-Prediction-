Ball-by-ball match files go here (CSV or Cricsheet-style JSON), one file per match — e.g. 80 IPL games.

The unified dataset builder (python build_unified_dataset.py) scans this folder recursively, together with data/raw/.

Daily process (pre-match inputs, predictions, post-match ingest, retrain, error review): see ../DAILY_WORKFLOW.txt at repo root.

IPL 2026 fixtures (home/away, dates, cities): ../data/processed/ipl_2026_fixtures.csv — use scripts/show_fixture.py or iplpred.core.fixtures.
