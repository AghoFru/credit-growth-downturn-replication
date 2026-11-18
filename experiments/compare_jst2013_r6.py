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
RUN = ROOT / "runs/jst-r6-vintage-compare-01"
ARCHIVE = ROOT / "runs/jst-2013-benchmark-01/source/archive/WCBB_replication_14aug2015"
ORIGINAL_PANEL = ARCHIVE / "data/private.dta"
R6_PANEL = ROOT / "data/raw/JSTdatasetR6.dta"
SCRIPT = ROOT / "experiments/compare_jst2013_r6.py"
OUTPUT = RUN / "results/jst_table8_vintage_comparison.csv"
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
COUNTRY_TERMS = [f"country_{code}" for code in range(1, 14)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def outcomes(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["ccode", "year"]).copy()
    for horizon in range(1, 6):
        future = panel.groupby("ccode")["lrgdp"].shift(-horizon)
        panel[f"y{horizon}"] = 100 * (future - panel["lrgdp"])
    for code in range(1, 14):
        panel[f"country_{code}"] = np.where(panel["ccode"] == code, 1 - 1 / 14, -1 / 14)
    return panel


def prepare_original(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    for control in CONTROLS:
        panel[control] = (panel[control] - panel[control].mean()) / panel[control].std(
            ddof=1
        )
        panel[f"lag_{control}"] = panel.groupby("ccode")[control].shift()
    return outcomes(panel)


def prepare_r6(r6: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    keys = ["iso", "year"]
    chronology = original[keys + ["ccode", "core", "pk_norm", "pk_fin", "dx_lrgdp_bb"]]
    panel = r6.copy()
    panel["year"] = panel["year"].astype(int)
    panel = panel.merge(chronology, on=keys, how="inner", validate="one_to_one")
    panel = panel.sort_values(["ccode", "year"]).copy()
    panel["lrgdp"] = np.log(panel["rgdpbarro"])
    raw_controls = {
        "drprv": np.log(panel["tloans"]) - np.log(panel["cpi"]) - np.log(panel["pop"]),
        "dlrgdp": np.log(panel["rgdpbarro"]),
        "dlcpi": np.log(panel["cpi"]),
        "dlriy": np.log(panel["iy"] * panel["rgdpbarro"]),
    }
    for control, raw in raw_controls.items():
        panel[control] = raw.groupby(panel["ccode"], sort=False).diff() * 100
    panel["cay"] = 100 * panel["ca"] / panel["gdp"]

    credit_ratio = panel["tloans"] / panel["gdp"]
    lookup = {
        (int(row.ccode), int(row.year)): row.credit_ratio
        for row in panel.assign(credit_ratio=credit_ratio).itertuples()
    }
    excess = []
    for row in panel.assign(credit_ratio=credit_ratio).itertuples():
        duration = row.dx_lrgdp_bb
        prior = (
            lookup.get((int(row.ccode), int(row.year - duration)), np.nan)
            if pd.notna(duration) and duration > 0
            else np.nan
        )
        excess.append(
            100 * (row.credit_ratio - prior) / duration
            if pd.notna(prior) and pd.notna(row.credit_ratio)
            else np.nan
        )
    panel["excess_credit"] = excess

    for control in CONTROLS:
        panel[control] = (panel[control] - panel[control].mean()) / panel[control].std(
            ddof=1
        )
        panel[f"lag_{control}"] = panel.groupby("ccode")[control].shift()

    panel["normal_excess_credit"] = 0.0
    panel["financial_excess_credit"] = 0.0
    for flag, term in (
        ("pk_norm", "normal_excess_credit"),
        ("pk_fin", "financial_excess_credit"),
    ):
        in_bin = (
            panel["core"].eq(1) & panel[flag].eq(1) & panel["excess_credit"].notna()
        )
        panel.loc[in_bin, term] = (
            panel.loc[in_bin, "excess_credit"]
            - panel.loc[in_bin, "excess_credit"].mean()
        )
        missing = (
            panel["core"].eq(1) & panel[flag].eq(1) & panel["excess_credit"].isna()
        )
        panel.loc[missing, term] = np.nan

    return outcomes(panel)


def model_rows(
    panel: pd.DataFrame,
    sample_name: str,
    interaction_terms: tuple[str, str],
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    terms = ["pk_norm", "pk_fin", *interaction_terms]
    regressors = [*terms, *CONTROLS, *(f"lag_{c}" for c in CONTROLS), *COUNTRY_TERMS]
    sample = panel.loc[panel["core"].eq(1)].dropna(
        subset=[*regressors, *(f"y{h}" for h in range(1, 6))]
    )
    rows = []
    for horizon in range(1, 6):
        fit = sm.OLS(
            sample[f"y{horizon}"].to_numpy(float), sample[regressors].to_numpy(float)
        ).fit()
        for index, term in enumerate(TERMS):
            rows.append(
                {
                    "sample": sample_name,
                    "horizon": horizon,
                    "term": term,
                    "coefficient": fit.params[index],
                    "standard_error": fit.bse[index],
                    "observations": int(fit.nobs),
                    "normal_recessions": int(sample["pk_norm"].sum()),
                    "financial_recessions": int(sample["pk_fin"].sum()),
                }
            )
    return rows, sample


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    original_raw = pd.read_stata(ORIGINAL_PANEL).sort_values(["ccode", "year"]).copy()
    original_raw["year"] = pd.to_datetime(original_raw["year"]).dt.year
    original = prepare_original(original_raw)
    r6 = prepare_r6(pd.read_stata(R6_PANEL), original_raw)

    original_rows, original_sample = model_rows(
        original, "original full sample", ("srx_norm_prv", "srx_fin_prv")
    )
    r6_rows, r6_sample = model_rows(
        r6,
        "R6 all complete original-date events",
        ("normal_excess_credit", "financial_excess_credit"),
    )
    common_keys = original_sample[["iso", "year"]].merge(
        r6_sample[["iso", "year"]], on=["iso", "year"], validate="one_to_one"
    )
    original_common = original_sample.merge(
        common_keys, on=["iso", "year"], validate="one_to_one"
    )
    r6_common = r6_sample.merge(common_keys, on=["iso", "year"], validate="one_to_one")
    original_common_rows, _ = model_rows(
        original_common, "original common sample", ("srx_norm_prv", "srx_fin_prv")
    )
    r6_common_rows, _ = model_rows(
        r6_common,
        "R6 common sample",
        ("normal_excess_credit", "financial_excess_credit"),
    )
    estimates = pd.DataFrame(
        original_rows + original_common_rows + r6_rows + r6_common_rows
    )

    check = estimates.loc[estimates["sample"].eq("original full sample")]
    coefficient_matrix = check.pivot(
        index="horizon", columns="term", values="coefficient"
    )[TERMS].to_numpy()
    standard_error_matrix = check.pivot(
        index="horizon", columns="term", values="standard_error"
    )[TERMS].to_numpy()
    assert len(original_sample) == 121
    assert len(original_common) == len(r6_common) == 116
    assert len(r6_sample) == 148
    assert int(r6_sample["pk_norm"].sum()) == 116
    assert int(r6_sample["pk_fin"].sum()) == 32
    assert np.array_equal(np.round(coefficient_matrix, 3), EXPECTED)
    assert np.array_equal(np.round(standard_error_matrix, 3), EXPECTED_SE)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(OUTPUT, index=False, float_format="%.10f")
    original_code = [
        ARCHIVE / "programs/data_Oct2012.do",
        ARCHIVE / "programs/AMTregressions_Oct2012.do",
        ARCHIVE / "auxprograms/annual_brybos_loop.do",
        ARCHIVE / "auxprograms/peak_dummies_August2012Alan.do",
        ARCHIVE / "tables/table8.tex",
    ]
    metadata = {
        "analysis": "JST Table 8 original-vintage and R6 matched-event comparison",
        "command": "uv run python experiments/compare_jst2013_r6.py",
        "random_seed": None,
        "specification": {
            "chronology": "original 2013 core Bry-Boschan peak years and normal/financial labels",
            "original_full_sample": "author private.dta, original Table 8 transformations and controls",
            "common_sample": "the 116 original Table 8 events with complete inputs under both vintages",
            "r6_all_sample": "all original-date core events with complete R6 model inputs",
            "r6_excess_credit": "100 times the peak-to-prior-trough tloans/GDP change divided by original expansion duration",
            "r6_controls": "originally defined seven controls, standardized over the matched 14-country 1870-2008 panel, with one annual lag",
            "standard_errors": "classical OLS, matching Stata reg default",
            "interpretation": "common-sample vintage comparison is descriptive, not causal",
        },
        "samples": {
            "original_full": {
                "n": len(original_sample),
                "normal": int(original_sample.pk_norm.sum()),
                "financial": int(original_sample.pk_fin.sum()),
            },
            "original_common": {
                "n": len(original_common),
                "normal": int(original_common.pk_norm.sum()),
                "financial": int(original_common.pk_fin.sum()),
            },
            "r6_common": {
                "n": len(r6_common),
                "normal": int(r6_common.pk_norm.sum()),
                "financial": int(r6_common.pk_fin.sum()),
            },
            "r6_all": {
                "n": len(r6_sample),
                "normal": int(r6_sample.pk_norm.sum()),
                "financial": int(r6_sample.pk_fin.sum()),
            },
        },
        "inputs": {
            "original_panel": {
                "path": str(ORIGINAL_PANEL.relative_to(ROOT)),
                "sha256": sha256(ORIGINAL_PANEL),
            },
            "r6_panel": {
                "path": str(R6_PANEL.relative_to(ROOT)),
                "sha256": sha256(R6_PANEL),
            },
            "original_source": {
                str(path.relative_to(ROOT)): sha256(path) for path in original_code
            },
        },
        "code": {"path": str(SCRIPT.relative_to(ROOT)), "sha256": sha256(SCRIPT)},
        "output": {"path": str(OUTPUT.relative_to(ROOT)), "sha256": sha256(OUTPUT)},
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "statsmodels": statsmodels.__version__,
            "platform": platform.platform(),
            "code_version": "not recorded, project has no Git metadata",
        },
    }
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n")
    print(estimates.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
