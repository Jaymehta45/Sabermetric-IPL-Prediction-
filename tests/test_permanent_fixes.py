"""Golden / regression tests for identity bridge, win-prob bounds, impact scoring."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from iplpred.core.squad_utils import clear_resolve_cache, resolve_to_squad_player_id
from iplpred.core.match_context import (
    MatchContext,
    PitchMultipliers,
    parse_pitch_report,
    resolve_pitch_multipliers,
)
from iplpred.match.inference_feature_rows import franchise_match, venue_matches
from iplpred.match.match_simulator import (
    _calibrated_innings_targets,
    _impact_base,
    _team_calibration_scales,
)
from iplpred.match.win_prob_ensemble import apply_ensemble_and_calibrate


class TestSquadBridge(unittest.TestCase):
    def test_fin_allen_kkr(self) -> None:
        clear_resolve_cache()
        r = resolve_to_squad_player_id("Kolkata Knight Riders", "Finn Allen")
        self.assertEqual(r, "Finn Allen")


class TestWinProbBounds(unittest.TestCase):
    def test_ensemble_clipped(self) -> None:
        p = apply_ensemble_and_calibrate(0.001, 0.001)
        self.assertIsNotNone(p)
        assert p is not None
        self.assertGreaterEqual(p, 0.02)
        self.assertLessEqual(p, 0.98)


class TestInferenceVenueFranchise(unittest.TestCase):
    def test_venue_token(self) -> None:
        self.assertTrue(
            venue_matches(
                "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
                "Ekana",
            )
        )
        self.assertFalse(venue_matches("", "Mumbai"))

    def test_franchise_loose(self) -> None:
        self.assertTrue(franchise_match("Delhi Capitals", "Delhi Capitals"))
        self.assertTrue(franchise_match("Mumbai Indians", "Mumbai"))


class TestPitchHighPar(unittest.TestCase):
    def test_belter_and_cited_par_lifts_runs_multipliers(self) -> None:
        text = (
            "Eden Gardens belter. Last year 199 was the average score batting first. "
            "Expect more than 200 when chasing. Fours and sixes everywhere."
        )
        pm = parse_pitch_report(text)
        self.assertGreater(pm.first_innings_runs, 1.4)
        self.assertGreater(pm.second_innings_runs, 1.4)

    def test_gt_rr_style_commentary_high_par_not_pulled_down_by_bowlers_keyword(self) -> None:
        text = (
            "Narendra Modi Stadium. Grass and tramlines for fast bowlers to aim at. "
            "Average score here 216, economy 10.2 for fast bowlers. "
            "High scoring run chase. It's a 200-run ground today."
        )
        pm = parse_pitch_report(text)
        self.assertGreater(pm.first_innings_runs, 1.35)

    def test_venue_nudge_ahmedabad(self) -> None:
        ctx = MatchContext(
            pitch_text="Flat deck, good for batting.",
            venue="Narendra Modi Stadium, Ahmedabad",
        )
        pm = resolve_pitch_multipliers(ctx)
        plain = parse_pitch_report(ctx.pitch_text or "")
        self.assertGreater(pm.first_innings_runs, plain.first_innings_runs)


class TestTeamRunCalibration(unittest.TestCase):
    def test_anchor_lifts_low_raw_sums(self) -> None:
        pm = PitchMultipliers()
        t1, t2, m = _calibrated_innings_targets(40.0, 50.0, pm, None)
        self.assertEqual(m, "raw_sum_anchor")
        self.assertGreater(t1, 100.0)
        self.assertGreater(t2, 100.0)
        s1, s2 = _team_calibration_scales(40.0, 50.0, t1, t2)
        self.assertGreater(s1, 1.5)
        self.assertGreater(s2, 1.5)

    def test_high_par_pitch_blends_ml_totals_upward(self) -> None:
        pm = parse_pitch_report(
            "Belter. 199 average batting first. More than 200 when chasing. "
            "Fours and sixes everywhere."
        )
        t1, t2, m = _calibrated_innings_targets(40.0, 50.0, pm, (118.0, 120.0))
        self.assertEqual(m, "ml_team_total")
        self.assertGreater((t1 + t2) / 2.0, 170.0)


class TestImpactBase(unittest.TestCase):
    def test_bowler_not_minimal(self) -> None:
        runs = pd.Series([2.0, 30.0])
        wk = pd.Series([3.0, 0.1])
        role = pd.Series([1.0, 0.0])  # bowler, batter
        ib = _impact_base(runs, wk, role)
        # Bowler (row 0) should have higher impact than raw 2+75=77
        self.assertGreater(float(ib.iloc[0]), 77.0)


if __name__ == "__main__":
    unittest.main()
