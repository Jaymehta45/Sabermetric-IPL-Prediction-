"""Recent innings priors and venue affinity helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from iplpred.core.recent_innings_context import (
    blend_innings_targets_with_recent_scores,
    recent_team_innings_scores,
    venue_same_ground,
)


class TestVenueSameGround(unittest.TestCase):
    def test_chinnaswamy_aliases(self) -> None:
        self.assertTrue(
            venue_same_ground(
                "M Chinnaswamy Stadium, Bengaluru",
                "Bengaluru",
            )
        )
        self.assertTrue(
            venue_same_ground("M.Chinnaswamy Stadium, Bengaluru", "Bengaluru")
        )


class TestRecentTeamScores(unittest.TestCase):
    def test_role_specific_recent(self) -> None:
        # A always listed as team1 and bats first → team1_total_runs is A's first innings.
        hist = pd.DataFrame(
            {
                "team1_name": ["A", "A", "A"],
                "team2_name": ["B", "B", "B"],
                "team1_total_runs": [200.0, 180.0, 170.0],
                "team2_total_runs": [55.0, 60.0, 65.0],
                "venue": ["X", "X", "X"],
                "match_date": pd.to_datetime(["2026-04-01", "2026-04-08", "2026-04-12"]),
                "team1_bats_first_signal": [1.0, 1.0, 1.0],
            }
        )
        p, m = recent_team_innings_scores(
            "A",
            "2026-04-15",
            role="first",
            venue=None,
            last_n=2,
            history=hist,
        )
        self.assertIsNotNone(p)
        assert p is not None
        self.assertAlmostEqual(p, 175.0, places=5)  # mean(180, 170)
        self.assertGreaterEqual(m["n_role"], 1)

    def test_blend_moves_toward_prior(self) -> None:
        t1, t2, meta = blend_innings_targets_with_recent_scores(
            "Royal Challengers Bangalore",
            "Delhi Capitals",
            "M Chinnaswamy Stadium, Bengaluru",
            "2026-04-18",
            200.0,
            180.0,
            last_n=3,
            weight=0.5,
        )
        self.assertIn("team1_recent", meta)
        self.assertGreater(t1, 0.0)
        self.assertGreater(t2, 0.0)


if __name__ == "__main__":
    unittest.main()
