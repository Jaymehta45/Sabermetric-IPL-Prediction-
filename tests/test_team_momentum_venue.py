"""Venue key canonicalization and GT–Ahmedabad shrinkage."""

from __future__ import annotations

import unittest

from iplpred.core.team_momentum import (
    IPL_AHMEDABAD_KEY,
    _canonical_venue_key,
    venue_momentum_row_from_history,
)


class TestVenueCanonical(unittest.TestCase):
    def test_ahmedabad_aliases_merge(self) -> None:
        self.assertEqual(_canonical_venue_key("Narendra Modi Stadium, Ahmedabad"), IPL_AHMEDABAD_KEY)
        self.assertEqual(_canonical_venue_key("Ahmedabad"), IPL_AHMEDABAD_KEY)

    def test_kkr_gt_ahmedabad_inference_prior(self) -> None:
        v1, v2 = venue_momentum_row_from_history(
            "Kolkata Knight Riders",
            "Gujarat Titans",
            "Narendra Modi Stadium, Ahmedabad",
            "2026-04-17",
        )
        self.assertEqual(v1, 0.5)
        self.assertGreater(v2, 0.5)


if __name__ == "__main__":
    unittest.main()
