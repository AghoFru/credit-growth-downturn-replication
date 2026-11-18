import unittest

import pandas as pd

from experiments.break_even_sensitivity import episode_losses, flag_episodes


def paths_for(starts: list[int], gap: float) -> pd.DataFrame:
    rows = [
        {
            "iso": "TST",
            "start_year": start,
            "horizon": h,
            "trend": gap,
            "flat_peak": -gap,
        }
        for start in starts
        for h in range(1, 11)
    ]
    return pd.DataFrame(rows)


class BreakEvenSensitivityTest(unittest.TestCase):
    def test_flags_overlap_war_and_non_overlapping_subset(self):
        flags = flag_episodes(
            pd.DataFrame({"iso": "TST", "start_year": [1900, 1904, 1916, 1930]})
        )
        self.assertEqual(
            flags["next_start_within_10y"].tolist(), [True, False, False, False]
        )
        self.assertEqual(flags["war_start"].tolist(), [False, False, True, False])
        self.assertEqual(
            flags["non_overlapping_subset"].tolist(), [True, False, True, True]
        )

    def test_dedup_counts_each_year_once_and_signed_keeps_gains(self):
        starts = [1900, 1904]
        flags = flag_episodes(pd.DataFrame({"iso": "TST", "start_year": starts}))
        losses = episode_losses(paths_for(starts, 1.0), flags, rate=0.0)
        first = losses.set_index("start_year").loc[1900]
        # The first window is cut at the next start: years 1 to 4 only.
        self.assertAlmostEqual(first["trend_floored_dedup"], 4.0)
        self.assertAlmostEqual(first["trend_floored"], 10.0)
        # Negative gaps are output gains: the floor removes them, signed sums keep them.
        self.assertAlmostEqual(first["flat_peak_floored"], 0.0)
        self.assertAlmostEqual(first["flat_peak_signed"], -10.0)


if __name__ == "__main__":
    unittest.main()
