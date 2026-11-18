import unittest

import numpy as np
import pandas as pd

from experiments.run_severity_analysis import make_episodes


class SeverityAnalysisTest(unittest.TestCase):
    def test_peak_to_trough_and_recovery_dates(self):
        years = np.arange(1990, 2016)
        output = 100 * np.exp(0.02 * (years - 1990))
        output[14:17] = [output[13] * 0.96, output[13] * 0.92, output[13] * 0.96]
        output[17] = output[13] * 1.01
        panel = pd.DataFrame(
            {
                "iso": "TST",
                "year": years,
                "rgdpbarro": output,
                "gdp": 1000.0,
                "tloans": 500.0,
                "hpnom": np.exp(0.02 * (years - 1990)),
                "iy": 0.2,
                "cpi": np.exp(0.02 * (years - 1990)),
                "unemp": 5.0,
                "stir": 3.0,
                "crisisJST": 0,
            }
        )
        panel["credit_gdp"] = 50.0
        panel["credit_growth_3y"] = 0.0
        panel["house_price_growth_3y"] = 0.0
        panel["equity_price_growth_3y"] = 0.0
        panel["real_growth_3y"] = 6.0
        panel["inflation_3y"] = 6.0
        episodes = make_episodes(panel)
        event = episodes.iloc[0]
        self.assertEqual(event["start_year"], 2003)
        self.assertEqual(event["trough_year"], 2005)
        self.assertEqual(event["recovery_years"], 4)
        self.assertAlmostEqual(event["peak_to_trough_pct"], 8.0)


if __name__ == "__main__":
    unittest.main()
