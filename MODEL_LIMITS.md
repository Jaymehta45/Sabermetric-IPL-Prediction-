# What these models do and do not do

- **Pre-match only**: Features are built from history **before** the fixture (`shift(1)` rolling windows in `player_features`). They do not see ball outcomes from the same match.

- **Winner labels**: Prefer official IPL results where available; otherwise a **runs proxy** (higher team total). Proxy rows are **down-weighted** in training. The classifier is a **calibrated random forest** on team aggregates, not a full simulation.

- **Team totals**: Ridge heads on the same feature matrix, optionally trained on **`log1p(runs)`** with a clip, then blended in the simulator with pitch and venue calibration. They are **not** a replacement for the Monte Carlo process.

- **Momentum**: Last-five franchise win rate from `match_training_dataset.csv`, with **canonical franchise names** (e.g. MI → Mumbai Indians). Unknown strings are passed through unchanged.

- **Not modeled**: Weather, toss beyond ordering, exact XI uncertainty, impact-sub causal effect, live state.

Use `scripts/evaluate_match_models.py` for holdout accuracy/MAE; treat metrics as **directional**, not guarantees.
