from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.formula.api as smf
from compare_jst2013_r6 import (
    ARCHIVE,
    ORIGINAL_PANEL,
    R6_PANEL,
    model_rows,
    prepare_original,
    prepare_r6,
)
from compare_jst2013_r6 import (
    CONTROLS as JST_CONTROLS,
)
from compare_jst_r6_event_rules import original_events, original_peak_outcomes
from run_severity_analysis import load_panel
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/jst-r6-spec-bridge-01"
SCRIPT = ROOT / "experiments/decompose_jst_r6_specification.py"
FIXED_KEYS = RUN / "results/fixed_sample_keys.csv"
SAMPLE_FLOW = RUN / "results/sample_flow.csv"
SAMPLE_SELECTION = RUN / "results/jst_r6_sample_selection.csv"
ESTIMATES = RUN / "results/jst_r6_specification_bridge.csv"
METADATA = RUN / "run_metadata.json"
CURRENT_CONTROLS = [
    "credit_gdp",
    "house_price_growth_3y",
    "investment_share",
    "real_growth_3y",
    "inflation_3y",
    "interest_rate",
]
SEVERITY_OUTCOMES = [
    "peak_to_trough_pct",
    "cumulative_loss_5y_pct_gdp_years",
]
FORWARD_OUTCOMES = [f"y{horizon}" for horizon in range(1, 6)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def standardize_within_group(
    frame: pd.DataFrame,
    source: str,
    group: pd.Series,
    target: str,
) -> None:
    selected = frame.loc[group, source]
    scale = float(selected.std(ddof=1))
    assert scale > 0
    frame[target] = 0.0
    frame.loc[group, target] = (selected - float(selected.mean())) / scale


def sample_selection_estimates(
    frame: pd.DataFrame, sample_name: str
) -> list[dict[str, object]]:
    frame = frame.copy()
    frame["financial_event"] = frame["pk_fin"].astype(float)
    controls = [*JST_CONTROLS, *(f"lag_{control}" for control in JST_CONTROLS)]
    terms = [
        "financial_event",
        "normal_excess_credit",
        "financial_excess_credit",
        *controls,
    ]
    rows = []
    for horizon, outcome in enumerate(FORWARD_OUTCOMES, start=1):
        fit = smf.ols(
            f"{outcome} ~ " + " + ".join([*terms, "C(iso)"]), data=frame
        ).fit()
        clustered = smf.ols(
            f"{outcome} ~ " + " + ".join([*terms, "C(iso)"]), data=frame
        ).fit(cov_type="cluster", cov_kwds={"groups": frame["iso"]})
        for term in ("normal_excess_credit", "financial_excess_credit"):
            rows.append(
                {
                    "sample": sample_name,
                    "horizon": horizon,
                    "term": term,
                    "coefficient_raw_units": float(fit.params[term]),
                    "classical_standard_error": float(fit.bse[term]),
                    "clustered_standard_error": float(clustered.bse[term]),
                    "observations": int(fit.nobs),
                    "countries": frame["iso"].nunique(),
                }
            )
    return rows


def estimate(
    frame: pd.DataFrame,
    model: str,
    outcome: str,
    horizon: int | None,
    terms: list[str],
    focal_terms: list[tuple[str, str]],
    predictor_scale: str,
    controls: list[str],
    decade_effects: bool,
) -> list[dict[str, object]]:
    rhs = [*terms, *controls, "C(iso)"]
    if decade_effects:
        rhs.append("C(decade)")
    fit = smf.ols(f"{outcome} ~ " + " + ".join(rhs), data=frame).fit(
        cov_type="cluster", cov_kwds={"groups": frame["iso"]}
    )
    critical = float(student_t.ppf(0.975, frame["iso"].nunique() - 1))
    outcome_sd = float(frame[outcome].std(ddof=1))
    rows = []
    for term, predictor_label in focal_terms:
        coefficient = float(fit.params[term])
        standard_error = float(fit.bse[term])
        scale_source, scale_group = {
            "normal_excess_credit_z": ("normal_excess_credit", "pk_norm"),
            "financial_excess_credit_z": ("financial_excess_credit", "pk_fin"),
            "credit_growth_normal_z": ("credit_growth_3y", "pk_norm"),
            "credit_growth_financial_z": ("credit_growth_3y", "pk_fin"),
            "credit_growth_normal_interaction": ("credit_growth_3y", None),
            "credit_growth_financial_interaction": ("credit_growth_3y", None),
            "credit_growth_pooled_z": ("credit_growth_3y", None),
        }[term]
        predictor_values = frame[scale_source]
        if scale_group is not None:
            predictor_values = predictor_values.loc[frame[scale_group].eq(1)]
        predictor_sd = float(predictor_values.std(ddof=1))
        raw_coefficient = coefficient / predictor_sd
        raw_standard_error = standard_error / predictor_sd
        rows.append(
            {
                "model": model,
                "outcome": outcome,
                "horizon": horizon,
                "term": term,
                "predictor": predictor_label,
                "predictor_scale": predictor_scale,
                "coefficient_per_scale_unit": coefficient,
                "clustered_standard_error": standard_error,
                "ci_low_95": coefficient - critical * standard_error,
                "ci_high_95": coefficient + critical * standard_error,
                "standardized_beta": coefficient / outcome_sd,
                "predictor_sd_raw_units": predictor_sd,
                "coefficient_raw_per_unit": raw_coefficient,
                "clustered_se_raw_per_unit": raw_standard_error,
                "ci_low_raw_per_unit": raw_coefficient - critical * raw_standard_error,
                "ci_high_raw_per_unit": raw_coefficient + critical * raw_standard_error,
                "outcome_sd": outcome_sd,
                "observations": int(fit.nobs),
                "countries": frame["iso"].nunique(),
                "r_squared": float(fit.rsquared),
            }
        )
    return rows


def run_specification_bridge(frame: pd.DataFrame) -> pd.DataFrame:
    jst_controls = [*JST_CONTROLS, *(f"lag_{control}" for control in JST_CONTROLS)]
    growth_controls = [
        "credit_growth_normal_z",
        "credit_growth_financial_z",
        "financial_event",
    ]
    current_controls = CURRENT_CONTROLS
    common_credit = ["credit_growth_pooled_z"]
    typed_current = [
        "credit_growth_normal_interaction",
        "credit_growth_financial_interaction",
        "financial_event",
    ]
    typed_common = ["credit_growth_pooled_z", "financial_event"]
    model_specs = [
        (
            "M0 R6 JST design",
            [
                "normal_excess_credit_z",
                "financial_excess_credit_z",
                "financial_event",
            ],
            jst_controls,
            False,
            [
                ("normal_excess_credit_z", "normal excess credit"),
                ("financial_excess_credit_z", "financial excess credit"),
            ],
            "one within-type standard deviation",
        ),
        (
            "M1 predictor switch",
            growth_controls,
            jst_controls,
            False,
            [
                ("credit_growth_normal_z", "normal 3-year credit growth"),
                ("credit_growth_financial_z", "financial 3-year credit growth"),
            ],
            "one within-type standard deviation",
        ),
        (
            "M2 current controls",
            typed_current,
            current_controls,
            False,
            [
                ("credit_growth_normal_interaction", "normal 3-year credit growth"),
                (
                    "credit_growth_financial_interaction",
                    "financial 3-year credit growth",
                ),
            ],
            "one pooled standard deviation",
        ),
        (
            "M3 add decade effects",
            typed_current,
            current_controls,
            True,
            [
                ("credit_growth_normal_interaction", "normal 3-year credit growth"),
                (
                    "credit_growth_financial_interaction",
                    "financial 3-year credit growth",
                ),
            ],
            "one pooled standard deviation",
        ),
        (
            "M4 common credit slope",
            typed_common,
            current_controls,
            True,
            [("credit_growth_pooled_z", "3-year credit growth")],
            "one pooled standard deviation",
        ),
        (
            "M5 drop event-type effects",
            common_credit,
            current_controls,
            True,
            [("credit_growth_pooled_z", "3-year credit growth")],
            "one pooled standard deviation",
        ),
    ]
    rows = []
    for horizon, outcome in enumerate(FORWARD_OUTCOMES, start=1):
        for model, terms, controls, decade_effects, focal, scale in model_specs:
            rows.extend(
                estimate(
                    frame,
                    model,
                    outcome,
                    horizon,
                    terms,
                    focal,
                    scale,
                    controls,
                    decade_effects,
                )
            )

    for outcome in SEVERITY_OUTCOMES:
        rows.extend(
            estimate(
                frame,
                "M6 severity outcome switch",
                outcome,
                None,
                common_credit,
                [("credit_growth_pooled_z", "3-year credit growth")],
                "one pooled standard deviation",
                current_controls,
                True,
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    FIXED_KEYS.parent.mkdir(parents=True, exist_ok=True)
    original_raw = pd.read_stata(ORIGINAL_PANEL).sort_values(["ccode", "year"]).copy()
    original_raw["year"] = pd.to_datetime(original_raw["year"]).dt.year
    original = prepare_original(original_raw)
    r6 = prepare_r6(pd.read_stata(R6_PANEL), original_raw)
    _, original_sample = model_rows(
        original, "original full sample", ("srx_norm_prv", "srx_fin_prv")
    )
    _, r6_sample = model_rows(
        r6,
        "R6 all original-date events",
        ("normal_excess_credit", "financial_excess_credit"),
    )
    common_keys = original_sample[["iso", "year"]].merge(
        r6_sample[["iso", "year"]], on=["iso", "year"], validate="one_to_one"
    )
    r6_common = r6_sample.merge(common_keys, on=["iso", "year"], validate="one_to_one")
    assert len(r6_common) == 116
    assert int(r6_common["pk_norm"].sum()) == 88
    assert int(r6_common["pk_fin"].sum()) == 28

    panel = load_panel()
    original_dates = original_events()
    severity = original_dates.merge(
        original_peak_outcomes(panel, original_dates).drop(
            columns=["pk_norm", "pk_fin"]
        ),
        on=["iso", "start_year"],
        validate="one_to_one",
    ).rename(columns={"start_year": "year"})
    bridge = r6_common.merge(
        severity, on=["iso", "year"], suffixes=("", "_severity"), validate="one_to_one"
    )
    assert len(bridge) == 116
    assert (bridge["pk_norm"] == bridge["pk_norm_severity"]).all()
    assert (bridge["pk_fin"] == bridge["pk_fin_severity"]).all()

    selected = bridge.dropna(
        subset=[
            *FORWARD_OUTCOMES,
            *SEVERITY_OUTCOMES,
            "credit_growth_3y",
            *CURRENT_CONTROLS,
        ]
    ).copy()
    selected["financial_event"] = selected["pk_fin"].astype(float)
    assert len(selected) == 94
    assert selected["iso"].nunique() == 14
    assert not selected.duplicated(["iso", "year"]).any()

    normal = selected["pk_norm"].eq(1)
    financial = selected["pk_fin"].eq(1)
    standardize_within_group(
        selected, "normal_excess_credit", normal, "normal_excess_credit_z"
    )
    standardize_within_group(
        selected, "financial_excess_credit", financial, "financial_excess_credit_z"
    )
    standardize_within_group(
        selected, "credit_growth_3y", normal, "credit_growth_normal_z"
    )
    standardize_within_group(
        selected, "credit_growth_3y", financial, "credit_growth_financial_z"
    )
    pooled_sd = float(selected["credit_growth_3y"].std(ddof=1))
    assert pooled_sd > 0
    selected["credit_growth_pooled_z"] = (
        selected["credit_growth_3y"] - selected["credit_growth_3y"].mean()
    ) / pooled_sd
    selected["credit_growth_normal_interaction"] = selected[
        "credit_growth_pooled_z"
    ] * normal.astype(float)
    selected["credit_growth_financial_interaction"] = selected[
        "credit_growth_pooled_z"
    ] * financial.astype(float)
    selected["decade"] = (selected["year"] // 10 * 10).astype(str)

    sample_flow = pd.DataFrame(
        [
            {"stage": "R6 complete original-date Table 8 events", "events": 116},
            {
                "stage": "with current 3-year credit growth",
                "events": int(bridge["credit_growth_3y"].notna().sum()),
            },
            {
                "stage": "with current house-price growth",
                "events": int(bridge["house_price_growth_3y"].notna().sum()),
            },
            {
                "stage": "complete bridge cohort",
                "events": len(selected),
            },
        ]
    )
    assert sample_flow["events"].tolist() == [116, 115, 95, 94]
    sample_flow.to_csv(SAMPLE_FLOW, index=False)
    selected[["iso", "year", "pk_norm", "pk_fin"]].to_csv(FIXED_KEYS, index=False)

    selection = pd.DataFrame(
        sample_selection_estimates(r6_common, "R6 Table 8 common sample, N=116")
        + sample_selection_estimates(selected, "Fixed bridge sample, N=94")
    )
    estimates = run_specification_bridge(selected)
    assert len(estimates) == 52
    assert estimates["observations"].eq(94).all()
    selection.to_csv(SAMPLE_SELECTION, index=False, float_format="%.10f")
    estimates.to_csv(ESTIMATES, index=False, float_format="%.10f")

    metadata = {
        "analysis": "JST R6 predictor, control, and outcome specification bridge",
        "command": "uv run python experiments/decompose_jst_r6_specification.py",
        "random_seed": None,
        "sample": {
            "starting_common_events": 116,
            "starting_normal": 88,
            "starting_financial": 28,
            "bridge_complete_cases": len(selected),
            "bridge_countries": selected["iso"].nunique(),
            "key_sha256": sha256(FIXED_KEYS),
        },
        "model_sequence": {
            "M0": "R6 JST Table 8 predictor and controls on the fixed bridge sample, standardized credit effects reported per within-type standard deviation.",
            "M1": "Swap to three-year credit growth, retaining event-type slopes, JST controls, lags, country effects, and forward-growth outcomes.",
            "M2": "Swap to current severity controls, retaining event-type slopes and country effects.",
            "M3": "Add decade fixed effects to M2.",
            "M4": "Replace event-type-specific credit slopes with one common slope and retain event-type intercepts.",
            "M5": "Drop event-type intercepts to match the current severity regression form.",
            "M6": "Keep M5 fixed and replace forward-growth outcomes with peak-to-trough and five-year cumulative losses.",
        },
        "inference": "Country-clustered standard errors and t intervals with 13 degrees of freedom. Every bridge model uses the same 94 exact country-year keys.",
        "limitations": [
            "The fixed bridge sample is selected from original JST events with complete R6 Table 8 outcomes and predictors.",
            "The 94-event sample is smaller than the 299-event release 6 analysis and does not explain every difference between the studies.",
            "R6 tloans is the closest available match to original loans1, but the definitions are not proven identical.",
            "The analysis is descriptive and does not identify causal effects.",
        ],
        "inputs": {
            "original_panel": {
                "path": str(ORIGINAL_PANEL.relative_to(ROOT)),
                "sha256": sha256(ORIGINAL_PANEL),
            },
            "r6_panel": {
                "path": str(R6_PANEL.relative_to(ROOT)),
                "sha256": sha256(R6_PANEL),
            },
            "original_table8_code": {
                "path": str(
                    (ARCHIVE / "programs/AMTregressions_Oct2012.do").relative_to(ROOT)
                ),
                "sha256": sha256(ARCHIVE / "programs/AMTregressions_Oct2012.do"),
            },
        },
        "outputs": {
            "fixed_keys": {
                "path": str(FIXED_KEYS.relative_to(ROOT)),
                "sha256": sha256(FIXED_KEYS),
            },
            "sample_flow": {
                "path": str(SAMPLE_FLOW.relative_to(ROOT)),
                "sha256": sha256(SAMPLE_FLOW),
            },
            "sample_selection": {
                "path": str(SAMPLE_SELECTION.relative_to(ROOT)),
                "sha256": sha256(SAMPLE_SELECTION),
            },
            "estimates": {
                "path": str(ESTIMATES.relative_to(ROOT)),
                "sha256": sha256(ESTIMATES),
            },
        },
        "code": {"path": str(SCRIPT.relative_to(ROOT)), "sha256": sha256(SCRIPT)},
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "platform": platform.platform(),
            "code_version": "not recorded, workspace has no project Git metadata",
            "original_archive_license": "no redistribution terms were identified in the downloaded archive",
        },
    }
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(sample_flow.to_string(index=False))
    print(estimates.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
