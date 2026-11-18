from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/jst-2013-benchmark-01"
ARCHIVE = RUN / "source/archive/WCBB_replication_14aug2015"
PANEL = ARCHIVE / "data/private.dta"
SOURCE = ARCHIVE / "programs/AMTregressions_Oct2012.do"
ARCHIVE_ZIP = RUN / "source/WCBB_replication.zip"
SCRIPT = ROOT / "experiments/reproduce_jst_table8.py"
OUTPUT = RUN / "results/jst_table8_reproduction.csv"
METADATA = RUN / "run_metadata.json"

CONTROLS = ["drprv", "dlrgdp", "dlcpi", "dlriy", "stir", "ltrate", "cay"]
TERMS = [
    "normal_recession",
    "financial_recession",
    "normal_excess_credit",
    "financial_excess_credit",
]
EXPECTED = np.array(
    [
        [-1.271, -2.828, -0.264, -0.401],
        [0.693, -4.135, -0.684, -0.991],
        [3.179, -3.586, -0.771, -0.374],
        [3.838, -2.751, -0.932, -1.298],
        [4.813, -1.365, -0.711, -0.895],
    ]
)
EXPECTED_SE = np.array(
    [
        [0.360, 0.574, 0.166, 0.213],
        [0.644, 1.027, 0.297, 0.381],
        [0.869, 1.386, 0.401, 0.515],
        [1.122, 1.791, 0.518, 0.665],
        [1.197, 1.910, 0.552, 0.709],
    ]
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    panel = pd.read_stata(PANEL).sort_values(["ccode", "year"]).copy()

    for control in CONTROLS:
        panel[control] = (panel[control] - panel[control].mean()) / panel[control].std(
            ddof=1
        )
        panel[f"lag_{control}"] = panel.groupby("ccode")[control].shift(1)

    for horizon in range(1, 6):
        panel[f"outcome_{horizon}"] = 100 * (
            panel.groupby("ccode")["lrgdp"].shift(-horizon) - panel["lrgdp"]
        )

    for country_code in range(1, 14):
        panel[f"country_{country_code}"] = np.where(
            panel["ccode"] == country_code, 1 - 1 / 14, -1 / 14
        )

    regressors = (
        ["pk_norm", "pk_fin", "srx_norm_prv", "srx_fin_prv"]
        + CONTROLS
        + [f"lag_{control}" for control in CONTROLS]
        + [f"country_{country_code}" for country_code in range(1, 14)]
    )
    rows = []
    for horizon in range(1, 6):
        sample = panel.loc[panel["core"].eq(1)].dropna(
            subset=regressors + [f"outcome_{horizon}"]
        )
        design = sample[regressors].to_numpy(dtype=float)
        outcome = sample[f"outcome_{horizon}"].to_numpy(dtype=float)
        fit = sm.OLS(outcome, design).fit()
        assert len(sample) == 121
        assert sample["pk_norm"].sum() == 92
        assert sample["pk_fin"].sum() == 29
        for index, term in enumerate(TERMS):
            rows.append(
                {
                    "horizon": horizon,
                    "term": term,
                    "coefficient": fit.params[index],
                    "standard_error": fit.bse[index],
                    "observations": int(fit.nobs),
                    "normal_recessions": int(sample["pk_norm"].sum()),
                    "financial_recessions": int(sample["pk_fin"].sum()),
                }
            )

    estimates = pd.DataFrame(rows)
    coefficient_matrix = estimates.pivot(
        index="horizon", columns="term", values="coefficient"
    )[TERMS].to_numpy()
    standard_error_matrix = estimates.pivot(
        index="horizon", columns="term", values="standard_error"
    )[TERMS].to_numpy()
    assert np.array_equal(np.round(coefficient_matrix, 3), EXPECTED)
    assert np.array_equal(np.round(standard_error_matrix, 3), EXPECTED_SE)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(OUTPUT, index=False, float_format="%.10f")
    METADATA.write_text(
        json.dumps(
            {
                "analysis": "JST 2013 Table 8 independent Python reproduction",
                "command": "uv run python experiments/reproduce_jst_table8.py",
                "random_seed": None,
                "specification": {
                    "source_table": "table8.tex",
                    "panel": "private.dta, original author archive vintage",
                    "sample": "core == 1 and complete outcome, contemporaneous controls, and one lag",
                    "horizons": [1, 2, 3, 4, 5],
                    "country_effects": "13 demeaned country indicators, no constant",
                    "standard_errors": "classical OLS, matching Stata reg default",
                    "rhs_standardization": "full-panel sample standard deviation, ddof=1",
                    "reproduction_check": "all 20 coefficients and standard errors match published three-decimal values",
                },
                "inputs": {
                    "author_archive": {
                        "path": str(ARCHIVE_ZIP.relative_to(ROOT)),
                        "sha256": sha256(ARCHIVE_ZIP),
                        "source": "https://www.moritzschularick.com/academic/datasets-and-codes/",
                    },
                    "panel": {
                        "path": str(PANEL.relative_to(ROOT)),
                        "sha256": sha256(PANEL),
                    },
                    "regression_source": {
                        "path": str(SOURCE.relative_to(ROOT)),
                        "sha256": sha256(SOURCE),
                    },
                    "published_table": {
                        "path": str((ARCHIVE / "tables/table8.tex").relative_to(ROOT)),
                        "sha256": sha256(ARCHIVE / "tables/table8.tex"),
                    },
                },
                "code": {
                    "path": str(SCRIPT.relative_to(ROOT)),
                    "sha256": sha256(SCRIPT),
                },
                "output": {
                    "path": str(OUTPUT.relative_to(ROOT)),
                    "sha256": sha256(OUTPUT),
                },
                "environment": {
                    "python": platform.python_version(),
                    "pandas": pd.__version__,
                    "numpy": np.__version__,
                    "statsmodels": statsmodels.__version__,
                    "platform": platform.platform(),
                    "code_version": "not recorded, workspace has no project Git metadata",
                    "author_archive_license": "no redistribution license was identified in the downloaded archive",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(estimates.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(
        "Table 8 matches the authors' archived output (table8.tex) at three decimals. "
        f"Results: {OUTPUT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
