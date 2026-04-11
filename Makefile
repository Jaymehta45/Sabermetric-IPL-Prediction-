# Full data refresh: unified → match stats (incl. ICC supplement) → features → priors → registry/identity → training tables.
.PHONY: build-all build-data test retrain-after-match train-win-ensemble clean-generated

build-all: build-data
	python3 build_training_dataset.py

build-data:
	python3 build_unified_dataset.py
	python3 build_player_match_stats.py
	python3 build_features.py
	python3 scripts/build_player_registry.py
	python3 scripts/build_player_identity.py
	python3 scripts/fill_player_feature_priors.py
	python3 scripts/build_player_registry.py
	python3 scripts/build_player_identity.py

test:
	python3 -m unittest discover -s tests -p "test_*.py" -v

# Full rebuild + retrain after adding a ball-by-ball file under matches/ (Process 2 — data half).
retrain-after-match:
	bash scripts/retrain_after_match.sh

# Refresh hybrid win-probability bundle after train_match_winner_model.py (safe if <40 matches: skips).
train-win-ensemble:
	python3 -m iplpred.training.train_win_prob_ensemble

# Remove local __pycache__ and .DS_Store (ignored by git; safe before commits).
clean-generated:
	bash scripts/clean_generated.sh

eval:
	python3 scripts/evaluate_match_models.py

# Public URL via Cloudflare (requires cloudflared; app must listen on PORT, default 8000).
.PHONY: tunnel
tunnel:
	bash scripts/cloudflared_tunnel.sh
