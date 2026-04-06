"""Franchise canonicalization and rolling-feature sanity checks."""

from __future__ import annotations

import unittest

import pandas as pd

from iplpred.core.franchise_normalize import canonical_franchise
from iplpred.pipeline.build_features import build_base_columns


class TestFranchiseCanonical(unittest.TestCase):
    def test_abbrev_and_aliases(self) -> None:
        self.assertEqual(canonical_franchise("MI"), "Mumbai Indians")
        self.assertEqual(canonical_franchise("Kings XI Punjab"), "Punjab Kings")
        self.assertEqual(
            canonical_franchise("Royal Challengers Bengaluru"),
            "Royal Challengers Bangalore",
        )

    def test_unknown_passthrough(self) -> None:
        self.assertEqual(canonical_franchise("Some New Team"), "Some New Team")


class TestRollingFormLeakage(unittest.TestCase):
    def test_form_runs_uses_prior_matches_only(self) -> None:
        """Rolling form uses shift(1) — same-match runs must not enter form_runs."""
        pms = pd.DataFrame(
            {
                "player_id": ["p1", "p1", "p1"],
                "batting_team": ["A", "A", "A"],
                "bowling_team": ["B", "B", "B"],
                "balls": [10, 10, 10],
                "balls_bowled": [0, 0, 0],
                "runs": [50.0, 60.0, 70.0],
                "runs_conceded": [0.0, 0.0, 0.0],
                "wickets": [0, 0, 0],
            }
        )
        out = build_base_columns(pms)
        self.assertAlmostEqual(float(out["form_runs"].iloc[0]), 0.0)
        self.assertAlmostEqual(float(out["form_runs"].iloc[1]), 50.0)
        self.assertAlmostEqual(float(out["form_runs"].iloc[2]), 55.0)


if __name__ == "__main__":
    unittest.main()
