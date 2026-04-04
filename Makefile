# Full data refresh: unified → match stats (incl. ICC supplement) → features → priors → registry/identity → training tables.
.PHONY: build-all build-data test

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

# Public URL via Cloudflare (requires cloudflared; app must listen on PORT, default 8000).
.PHONY: tunnel
tunnel:
	bash scripts/cloudflared_tunnel.sh
