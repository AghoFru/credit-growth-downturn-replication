from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/jst-brybos-frontier-01"
ARCHIVE = ROOT / "runs/jst-2013-benchmark-01/source/archive/WCBB_replication_14aug2015"
PANEL = ARCHIVE / "data/private.dta"
SCRIPT = ROOT / "experiments/build_jst_brybos_frontier.py"
RESULTS = RUN / "results"
CORE_WINDOWS = ((1870, 1908), (1921, 1933), (1948, 2003))
DISCOUNT_RATES = (0, 0.03, 0.07)
COSTS_PCT = (0.05, 0.1, 0.25, 0.5, 1.0)
BOOTSTRAP_DRAWS = 2000
SEED = 17


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def in_core_window(year: int) -> bool:
    return any(start <= year <= end for start, end in CORE_WINDOWS)


def eligible_exposure(panel: pd.DataFrame) -> pd.Series:
    exposure = {}
    for country, group in panel.groupby("iso", sort=True):
        log_gdp = group.set_index("year")["lrgdp"].sort_index()
        exposure[country] = sum(
            in_core_window(int(year))
            and log_gdp.reindex(range(int(year) - 5, int(year))).notna().all()
            for year in log_gdp.index
        )
    return pd.Series(exposure, dtype="int64")


def recession_phase_losses(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for country, group in panel.groupby("iso", sort=True):
        group = group.sort_values("year").set_index("year")
        peaks = group.loc[
            group["core"].eq(1) & (group["pk_norm"].eq(1) | group["pk_fin"].eq(1))
        ]
        for year, event in peaks.iterrows():
            year = int(year)
            after_peak = group.index.to_numpy() > year
            troughs = group.index[after_peak & group["th_bb"].eq(1).to_numpy()]
            next_peaks = group.index[after_peak & group["pk_bb"].eq(1).to_numpy()]
            if not len(troughs) or (len(next_peaks) and troughs[0] >= next_peaks[0]):
                continue

            trough = int(troughs[0])
            pre_years = np.arange(year - 5, year)
            phase_years = np.arange(year + 1, trough + 1)
            pre = group["lrgdp"].reindex(pre_years)
            phase = group["lrgdp"].reindex(phase_years)
            if not pre.notna().all() or not phase.notna().all():
                continue

            slope, intercept = np.polyfit(pre_years, pre.to_numpy(dtype=float), 1)
            actual = phase.to_numpy(dtype=float)
            trend_loss = 100 * (1 - np.exp(actual - (slope * phase_years + intercept)))
            flat_loss = 100 * (1 - np.exp(actual - float(group.loc[year, "lrgdp"])))
            for path, annual_loss in (
                ("pre-episode linear trend", trend_loss),
                ("flat pre-downturn peak", flat_loss),
            ):
                for rate in DISCOUNT_RATES:
                    discount = (1.0 + rate) ** -np.arange(1, len(phase_years) + 1)
                    rows.append(
                        {
                            "iso": country,
                            "peak_year": year,
                            "trough_year": trough,
                            "duration_years": len(phase_years),
                            "financial_recession": int(event["pk_fin"] == 1),
                            "counterfactual": path,
                            "discount_rate_pct": rate * 100,
                            "discounted_loss_pct_gdp_years": float(
                                np.dot(np.maximum(annual_loss, 0), discount)
                            ),
                        }
                    )

    events = pd.DataFrame(rows)
    bounds = events.drop_duplicates(["iso", "peak_year"])
    for _, group in bounds.groupby("iso", sort=True):
        group = group.sort_values("peak_year")
        assert (
            group["peak_year"].to_numpy()[1:] > group["trough_year"].to_numpy()[:-1]
        ).all()
    return events


def build_frontier(events: pd.DataFrame, exposure: pd.Series) -> pd.DataFrame:
    eligible_years = int(exposure.sum())
    rows = []
    for (path, rate), group in events.groupby(["counterfactual", "discount_rate_pct"]):
        event_count = len(group)
        frequency = event_count / eligible_years
        damage = group["discounted_loss_pct_gdp_years"].to_numpy()
        mean_damage = float(damage.mean())
        for cost in COSTS_PCT:
            required = cost / frequency
            rows.append(
                {
                    "annual_cost_pct_output": cost,
                    "eligible_country_years": eligible_years,
                    "complete_events": event_count,
                    "normal_events": int(group["financial_recession"].eq(0).sum()),
                    "financial_events": int(group["financial_recession"].eq(1).sum()),
                    "event_rate_per_eligible_country_year": frequency,
                    "counterfactual": path,
                    "discount_rate_pct": rate,
                    "mean_damage_pct_gdp_years": mean_damage,
                    "median_damage_pct_gdp_years": float(np.median(damage)),
                    "required_damage_pct_gdp_years_per_event": required,
                    "required_share_of_mean_damage_pct": 100 * required / mean_damage,
                }
            )

    result = pd.DataFrame(rows)
    check = result.loc[
        (result["annual_cost_pct_output"] == 0.1) & (result["discount_rate_pct"] == 3)
    ].set_index("counterfactual")
    assert int(check.loc["pre-episode linear trend", "complete_events"]) == 207
    assert int(check.loc["pre-episode linear trend", "eligible_country_years"]) == 1442
    assert (
        round(
            float(check.loc["pre-episode linear trend", "mean_damage_pct_gdp_years"]), 3
        )
        == 8.540
    )
    assert (
        round(
            float(check.loc["flat pre-downturn peak", "mean_damage_pct_gdp_years"]), 3
        )
        == 5.285
    )
    return result


def bootstrap_frontier(events: pd.DataFrame, exposure: pd.Series) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    countries = sorted(exposure.index)
    event_map = {
        (path, rate): {
            country: values["discounted_loss_pct_gdp_years"].to_numpy()
            for country, values in group.groupby("iso", sort=True)
        }
        for (path, rate), group in events.groupby(
            ["counterfactual", "discount_rate_pct"]
        )
    }
    rows = []
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = rng.choice(countries, size=len(countries), replace=True)
        eligible_years = int(exposure.reindex(sampled).sum())
        for (path, rate), country_events in event_map.items():
            samples = [
                country_events[country]
                for country in sampled
                if country in country_events
            ]
            losses = np.concatenate(samples)
            count = len(losses)
            frequency = count / eligible_years
            mean_damage = float(losses.mean())
            for cost in COSTS_PCT:
                required = cost / frequency
                rows.append(
                    {
                        "draw": draw,
                        "annual_cost_pct_output": cost,
                        "counterfactual": path,
                        "discount_rate_pct": rate,
                        "event_rate": frequency,
                        "mean_damage_pct_gdp_years": mean_damage,
                        "required_damage_pct_gdp_years_per_event": required,
                        "required_share_of_mean_damage_pct": 100
                        * required
                        / mean_damage,
                    }
                )
    return pd.DataFrame(rows)


def summarize_bootstrap(draws: pd.DataFrame) -> pd.DataFrame:
    keys = ["annual_cost_pct_output", "counterfactual", "discount_rate_pct"]
    rows = []
    for key, group in draws.groupby(keys, sort=False):
        row = dict(zip(keys, key, strict=True))
        for value in (
            "event_rate",
            "mean_damage_pct_gdp_years",
            "required_damage_pct_gdp_years_per_event",
            "required_share_of_mean_damage_pct",
        ):
            row[f"{value}_ci_low"] = float(group[value].quantile(0.025))
            row[f"{value}_ci_high"] = float(group[value].quantile(0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    panel = pd.read_stata(PANEL).sort_values(["iso", "year"]).copy()
    panel["year"] = pd.to_datetime(panel["year"]).dt.year
    exposure = eligible_exposure(panel)
    events = recession_phase_losses(panel)
    frontier = build_frontier(events, exposure)
    draws = bootstrap_frontier(events, exposure)
    uncertainty = summarize_bootstrap(draws)

    outputs = {
        "event_damage": RESULTS / "jst_brybos_phase_event_damage.csv",
        "frontier": RESULTS / "jst_brybos_phase_frontier.csv",
        "bootstrap_draws": RESULTS / "jst_brybos_phase_bootstrap.csv",
        "uncertainty": RESULTS / "jst_brybos_phase_uncertainty.csv",
    }
    events.to_csv(outputs["event_damage"], index=False, float_format="%.10f")
    frontier.to_csv(outputs["frontier"], index=False, float_format="%.10f")
    draws.to_csv(outputs["bootstrap_draws"], index=False, float_format="%.10f")
    uncertainty.to_csv(outputs["uncertainty"], index=False, float_format="%.10f")

    source_files = [
        ARCHIVE / "programs/data_Oct2012.do",
        ARCHIVE / "auxprograms/annual_brybos_loop.do",
        ARCHIVE / "auxprograms/peak_dummies_August2012Alan.do",
        ARCHIVE / "tables/table8.tex",
    ]
    metadata = {
        "analysis": "JST Bry-Boschan recession-phase break-even sensitivity",
        "command": "uv run python experiments/build_jst_brybos_frontier.py",
        "random_seed": SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "method": {
            "panel": "original-vintage author archive private.dta",
            "events": "core normal and financial Bry-Boschan peak rows paired to the next trough before the next peak",
            "exposure": "core-window country-years with five complete prior GDP observations",
            "damage": "sum positive annual output gaps from peak+1 through the next trough, against a five-year pre-peak log trend or flat peak",
            "discount_rates_pct": [0, 3, 7],
            "frontier": "annual output-equivalent cost divided by event rate, with frequency held fixed",
            "uncertainty": "country-cluster bootstrap, resampling 14 countries with replacement",
            "interpretation": "conditional break-even sensitivity, not a causal policy estimate",
        },
        "inputs": {
            "archive_zip": {
                "path": "runs/jst-2013-benchmark-01/source/WCBB_replication.zip",
                "sha256": sha256(
                    ROOT / "runs/jst-2013-benchmark-01/source/WCBB_replication.zip"
                ),
                "redistribution_license": "not identified in the archive",
            },
            "panel": {"path": str(PANEL.relative_to(ROOT)), "sha256": sha256(PANEL)},
            "original_source": {
                str(path.relative_to(ROOT)): sha256(path) for path in source_files
            },
        },
        "code": {"path": str(SCRIPT.relative_to(ROOT)), "sha256": sha256(SCRIPT)},
        "outputs": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for name, path in outputs.items()
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "code_version": "not recorded, project has no Git metadata",
        },
    }
    (RUN / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        frontier.loc[frontier["annual_cost_pct_output"].eq(0.1)].to_string(index=False)
    )
    print(
        uncertainty.loc[uncertainty["annual_cost_pct_output"].eq(0.1)].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
