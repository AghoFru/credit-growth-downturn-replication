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
from run_severity_analysis import load_panel, make_episodes
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "runs/jst-2013-benchmark-01/source/archive/WCBB_replication_14aug2015"
ORIGINAL_PANEL = ARCHIVE / "data/private.dta"
R6_PANEL = ROOT / "data/raw/JSTdatasetR6.dta"
RUN = ROOT / "runs/jst-r6-event-rules-01"
SCRIPT = ROOT / "experiments/compare_jst_r6_event_rules.py"
EVENTS_OUT = RUN / "results/event_rule_concordance.csv"
ESTIMATES_OUT = RUN / "results/event_rule_associations.csv"
METADATA = RUN / "run_metadata.json"
MAX_HORIZON = 10
PREDICTORS = [
    "credit_growth_3y",
    "credit_gdp",
    "house_price_growth_3y",
    "investment_share",
    "real_growth_3y",
    "inflation_3y",
    "interest_rate",
]
OUTCOMES = ["peak_to_trough_pct", "cumulative_loss_5y_pct_gdp_years"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def original_events() -> pd.DataFrame:
    original = pd.read_stata(ORIGINAL_PANEL)
    original["year"] = pd.to_datetime(original["year"]).dt.year
    events = original.loc[
        original["core"].eq(1) & (original["pk_norm"].eq(1) | original["pk_fin"].eq(1)),
        ["iso", "year", "pk_norm", "pk_fin"],
    ].rename(columns={"year": "start_year"})
    assert not events.duplicated(["iso", "start_year"]).any()
    return events


def original_peak_outcomes(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    records = []
    for country, group in panel.groupby("iso", sort=True):
        data = group.set_index("year").sort_index()
        output = data["rgdpbarro"]
        for event in events.loc[events["iso"].eq(country)].itertuples():
            start = int(event.start_year)
            peak = output.get(start, np.nan)
            years = np.arange(start + 1, start + MAX_HORIZON + 1)
            future = output.reindex(years)
            contiguous = future.notna().cumprod().astype(bool)
            future = future.loc[contiguous]
            future_years = future.index.to_numpy(dtype=int)
            if pd.isna(peak) or len(future) == 0:
                trough = np.nan
                trough_year = np.nan
                duration = np.nan
            else:
                values = future.to_numpy(dtype=float)
                trough_index = 0
                for index in range(1, len(values)):
                    if values[index] < values[trough_index]:
                        trough_index = index
                    if index > trough_index and values[index] >= peak:
                        break
                trough = float(values[trough_index])
                trough_year = int(future_years[trough_index])
                duration = trough_year - start

            prior_years = np.arange(start - 5, start)
            prior = output.reindex(prior_years)
            if len(prior) == 5 and prior.notna().all() and len(future) >= 5:
                slope, intercept = np.polyfit(prior_years, np.log(prior.to_numpy()), 1)
                loss_years = np.arange(start + 1, start + 6)
                trend = slope * loss_years + intercept
                losses = 100 * (
                    1 - np.exp(np.log(output.reindex(loss_years).to_numpy()) - trend)
                )
                cumulative_loss = float(np.maximum(losses, 0).sum())
            else:
                cumulative_loss = np.nan

            before = data.reindex(range(start - 3, start))
            predictors = {
                "credit_growth_3y": data.reindex([start - 1])["credit_growth_3y"].iloc[
                    0
                ],
                "credit_gdp": before["credit_gdp"].mean(),
                "house_price_growth_3y": data.reindex([start - 1])[
                    "house_price_growth_3y"
                ].iloc[0],
                "investment_share": before["iy"].mean(),
                "real_growth_3y": data.reindex([start - 1])["real_growth_3y"].iloc[0],
                "inflation_3y": data.reindex([start - 1])["inflation_3y"].iloc[0],
                "interest_rate": before["stir"].mean(),
            }
            records.append(
                {
                    "iso": country,
                    "start_year": start,
                    "trough_year": trough_year,
                    "duration_years": duration,
                    "peak_to_trough_pct": (
                        100 * (1 - trough / peak)
                        if pd.notna(peak) and pd.notna(trough)
                        else np.nan
                    ),
                    "cumulative_loss_5y_pct_gdp_years": cumulative_loss,
                    "pk_norm": int(event.pk_norm),
                    "pk_fin": int(event.pk_fin),
                    **predictors,
                }
            )
    result = pd.DataFrame(records)
    assert not result.duplicated(["iso", "start_year"]).any()
    return result


def estimate(events: pd.DataFrame, sample_name: str) -> list[dict[str, object]]:
    rows = []
    for outcome in OUTCOMES:
        complete = events.dropna(subset=[outcome, "iso", "decade", *PREDICTORS]).copy()
        if len(complete) < 40 or complete["iso"].nunique() < 2:
            continue
        formula = f"{outcome} ~ " + " + ".join(PREDICTORS) + " + C(iso) + C(decade)"
        fit = smf.ols(formula, data=complete).fit(
            cov_type="cluster", cov_kwds={"groups": complete["iso"]}
        )
        coefficient = float(fit.params["credit_growth_3y"])
        standard_error = float(fit.bse["credit_growth_3y"])
        critical_value = float(student_t.ppf(0.975, complete["iso"].nunique() - 1))
        rows.append(
            {
                "sample": sample_name,
                "outcome": outcome,
                "predictor": "credit_growth_3y",
                "coefficient": coefficient,
                "cluster_se": standard_error,
                "ci_low_95": coefficient - critical_value * standard_error,
                "ci_high_95": coefficient + critical_value * standard_error,
                "n_events": len(complete),
                "countries": complete["iso"].nunique(),
                "r_squared": float(fit.rsquared),
            }
        )
    return rows


def main() -> None:
    EVENTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    annual = make_episodes(panel).rename(columns={"start_year": "year"})
    annual["event_rule"] = "annual output decline"
    annual["pk_norm"] = np.nan
    annual["pk_fin"] = np.nan
    original = original_events()
    original["event_rule"] = "original JST core peak"
    original_severity = original_peak_outcomes(panel, original)
    original = original.merge(
        original_severity.drop(columns=["pk_norm", "pk_fin"]),
        on=["iso", "start_year"],
        validate="one_to_one",
    )
    original = original.rename(columns={"start_year": "year"})
    annual["decade"] = (annual["year"] // 10 * 10).astype(str)
    original["decade"] = (original["year"] // 10 * 10).astype(str)

    original_country_years = pd.read_stata(ORIGINAL_PANEL, columns=["iso", "year"])
    original_country_years["year"] = pd.to_datetime(
        original_country_years["year"]
    ).dt.year
    original_countries = set(original_country_years["iso"].unique())
    coverage = (annual["iso"].isin(original_countries)) & annual["year"].between(
        1870, 2002
    )
    annual_common_coverage = annual.loc[coverage].copy()
    original_common_coverage = original.loc[original["year"].between(1870, 2002)].copy()

    annual_keys = set(
        zip(annual_common_coverage["iso"], annual_common_coverage["year"])
    )
    original_keys = set(
        zip(original_common_coverage["iso"], original_common_coverage["year"])
    )
    overlap = annual_keys & original_keys
    assert len(annual) == 299
    assert len(original) == 223
    assert len(annual_common_coverage) == 225
    assert len(original_common_coverage) == 223
    assert len(overlap) == 182

    annual_common_coverage["sample"] = "annual, common country-period coverage"
    original_common_coverage["sample"] = "JST, common country-period coverage"
    matched = annual_common_coverage.loc[
        annual_common_coverage[["iso", "year"]].apply(tuple, axis=1).isin(overlap)
    ].copy()
    matched["sample"] = "exact-date overlap"
    original_overlap = original_common_coverage.loc[
        original_common_coverage[["iso", "year"]].apply(tuple, axis=1).isin(overlap)
    ]
    shared = matched.merge(
        original_overlap,
        on=["iso", "year"],
        suffixes=("_annual", "_jst"),
        validate="one_to_one",
    )
    shared_columns = [*OUTCOMES, *PREDICTORS]
    for column in shared_columns:
        assert np.allclose(
            shared[f"{column}_annual"],
            shared[f"{column}_jst"],
            equal_nan=True,
        ), column
    annual_only = annual_common_coverage.loc[
        ~annual_common_coverage[["iso", "year"]].apply(tuple, axis=1).isin(overlap)
    ].copy()
    annual_only["sample"] = "annual-only dates, common coverage"
    jst_only = original_common_coverage.loc[
        ~original_common_coverage[["iso", "year"]].apply(tuple, axis=1).isin(overlap)
    ].copy()
    jst_only["sample"] = "JST-only dates, common coverage"
    assert len(annual_only) == 43
    assert len(jst_only) == 41

    event_rows = []
    cohorts = [
        ("annual, full R6 coverage", annual),
        ("annual, common country-period coverage", annual_common_coverage),
        ("JST, common country-period coverage", original_common_coverage),
        ("exact-date overlap", matched),
        ("annual-only dates, common coverage", annual_only),
        ("JST-only dates, common coverage", jst_only),
    ]
    for sample_name, sample in cohorts:
        event_rows.append(
            {
                "sample": sample_name,
                "events": len(sample),
                "countries": sample["iso"].nunique(),
                "start_year_min": int(sample["year"].min()),
                "start_year_max": int(sample["year"].max()),
                "normal_jst_events": int(sample["pk_norm"].fillna(0).sum()),
                "financial_jst_events": int(sample["pk_fin"].fillna(0).sum()),
            }
        )
    pd.DataFrame(event_rows).to_csv(EVENTS_OUT, index=False)

    estimate_rows = []
    for sample_name, sample in cohorts:
        estimate_rows.extend(estimate(sample, sample_name))
    estimates = pd.DataFrame(estimate_rows)
    estimates.to_csv(ESTIMATES_OUT, index=False, float_format="%.10f")

    metadata = {
        "analysis": "R6 harmonized event-rule comparison",
        "command": "uv run python experiments/compare_jst_r6_event_rules.py",
        "random_seed": None,
        "method": {
            "event_rules": "Current annual output-decline episodes and original JST core Bry-Boschan peaks are dated on R6 country-year data.",
            "common_coverage": "14 original JST countries and event years 1870-2002.",
            "exact_match": "Same ISO country and event year.",
            "outcomes": "R6 peak-to-trough decline and 5-year cumulative losses relative to a 5-year pre-event linear trend.",
            "regression": "Current severity predictors with country and decade fixed effects and country-clustered standard errors.",
            "interpretation": "Descriptive event construction comparison, not causal policy evidence.",
            "limitations": [
                "Original chronology dates come from an older data vintage and are applied to R6 output.",
                "The original JST event count differs from the 121 complete observations in Table 8 because this cohort uses chronology flags before regression complete-case exclusions.",
                "The common-date overlap and unmatched cohorts share event outcomes only when their country and year match exactly.",
            ],
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
        },
        "code": {"path": str(SCRIPT.relative_to(ROOT)), "sha256": sha256(SCRIPT)},
        "outputs": {
            "event_concordance": {
                "path": str(EVENTS_OUT.relative_to(ROOT)),
                "sha256": sha256(EVENTS_OUT),
            },
            "associations": {
                "path": str(ESTIMATES_OUT.relative_to(ROOT)),
                "sha256": sha256(ESTIMATES_OUT),
            },
        },
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
    print(pd.DataFrame(event_rows).to_string(index=False))
    print(estimates.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
