"""Sensitivity of the break-even loss shares to the loss floor, window overlap, and war years.

The baseline in run_severity_analysis.py floors yearly output gaps at zero and sums them
over overlapping ten-year windows. This script recomputes the mean loss per downturn and
the share that a 0.1 percent annual cost requires under alternative loss definitions.
"""

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:  # Run as a script from experiments/.
    from run_severity_analysis import (
        MAX_HORIZON,
        RAW,
        eligible_exposure,
        load_panel,
        make_episodes,
    )
except ModuleNotFoundError:  # Imported by the tests from the project root.
    from experiments.run_severity_analysis import (
        MAX_HORIZON,
        RAW,
        eligible_exposure,
        load_panel,
        make_episodes,
    )

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "break-even-sensitivity-01"
OUTPUT = RUN / "results" / "break_even_sensitivity.csv"
EPISODE_OUTPUT = RUN / "results" / "episode_flags.csv"
ANNUAL_COST_PCT = 0.1
DISCOUNT_RATES = (0.0, 0.03)
WAR_START_YEARS = frozenset(range(1914, 1919)) | frozenset(range(1939, 1946))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gap_paths(panel: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    """Yearly output gaps (percent) for years 1 to 10 after each peak, without a floor."""
    rows = []
    for episode in episodes.itertuples(index=False):
        data = panel[panel["iso"] == episode.iso].set_index("year")["rgdpbarro"]
        start = int(episode.start_year)
        pre_years = np.arange(start - 5, start)
        future_years = np.arange(start + 1, start + MAX_HORIZON + 1)
        pre = data.reindex(pre_years)
        actual = data.reindex(future_years)
        if pre.isna().any() or actual.isna().any():
            continue
        slope, intercept = np.polyfit(pre_years, np.log(pre.to_numpy()), 1)
        trend_gap = 100 * (
            1 - np.exp(np.log(actual.to_numpy()) - (slope * future_years + intercept))
        )
        peak = float(data.loc[start])
        peak_gap = 100 * (1 - actual.to_numpy() / peak)
        for offset, (trend_value, peak_value) in enumerate(
            zip(trend_gap, peak_gap, strict=True)
        ):
            rows.append(
                {
                    "iso": episode.iso,
                    "start_year": start,
                    "horizon": offset + 1,
                    "trend": trend_value,
                    "flat_peak": peak_value,
                }
            )
    return pd.DataFrame(rows)


def flag_episodes(episodes: pd.DataFrame) -> pd.DataFrame:
    flags = episodes[["iso", "start_year"]].sort_values(["iso", "start_year"]).copy()
    next_start = flags.groupby("iso")["start_year"].shift(-1)
    flags["years_to_next_start"] = next_start - flags["start_year"]
    flags["next_start_within_10y"] = flags["years_to_next_start"].le(MAX_HORIZON)
    flags["war_start"] = flags["start_year"].isin(WAR_START_YEARS)
    kept = []
    for _, group in flags.groupby("iso", sort=False):
        last_kept = None
        for start in group["start_year"]:
            keep = last_kept is None or start > last_kept + MAX_HORIZON
            kept.append(keep)
            if keep:
                last_kept = start
    flags["non_overlapping_subset"] = kept
    return flags


def episode_losses(
    paths: pd.DataFrame, flags: pd.DataFrame, rate: float
) -> pd.DataFrame:
    merged = paths.merge(flags, on=["iso", "start_year"], validate="many_to_one")
    weights = (1 + rate) ** -merged["horizon"].astype(float)
    # A year belongs only to the latest downturn whose window covers it.
    own_year = merged["years_to_next_start"].isna() | (
        merged["horizon"] <= merged["years_to_next_start"]
    )
    out = {}
    for name in ("trend", "flat_peak"):
        floored = merged[name].clip(lower=0) * weights
        signed = merged[name] * weights
        out[f"{name}_floored"] = floored
        out[f"{name}_signed"] = signed
        out[f"{name}_floored_dedup"] = floored.where(own_year, 0.0)
        out[f"{name}_signed_dedup"] = signed.where(own_year, 0.0)
    frame = pd.concat([merged[["iso", "start_year"]], pd.DataFrame(out)], axis=1)
    return frame.groupby(["iso", "start_year"], as_index=False).sum()


def exposure_without_war(panel: pd.DataFrame) -> int:
    total = 0
    for _, group in panel.groupby("iso"):
        years = set(group.dropna(subset=["rgdpbarro"])["year"].astype(int))
        total += sum(
            all(year in years for year in range(start - 5, start + MAX_HORIZON + 1))
            for start in years
            if start not in WAR_START_YEARS
        )
    return total


def summarize(
    panel: pd.DataFrame, paths: pd.DataFrame, flags: pd.DataFrame
) -> pd.DataFrame:
    exposure_all = sum(eligible_exposure(panel).values())
    exposure_peace = exposure_without_war(panel)
    variants = (
        ("baseline: floored, overlapping, all starts", "floored", None, False),
        ("no floor (signed gaps)", "signed", None, False),
        ("de-duplicated years, floored", "floored_dedup", None, False),
        ("non-overlapping subset, floored", "floored", "non_overlapping_subset", False),
        ("war starts excluded, floored", "floored", None, True),
        ("no floor, de-duplicated, war excluded", "signed_dedup", None, True),
        ("floored, de-duplicated, war excluded", "floored_dedup", None, True),
    )
    rows = []
    for rate in DISCOUNT_RATES:
        losses = episode_losses(paths, flags, rate).merge(
            flags, on=["iso", "start_year"]
        )
        for label, measure, subset, drop_war in variants:
            sample = losses
            if subset is not None:
                sample = sample[sample[subset]]
            if drop_war:
                sample = sample[~sample["war_start"]]
            exposure = exposure_peace if drop_war else exposure_all
            frequency = len(sample) / exposure
            required = ANNUAL_COST_PCT / frequency
            for counterfactual in ("trend", "flat_peak"):
                mean_loss = float(sample[f"{counterfactual}_{measure}"].mean())
                rows.append(
                    {
                        "variant": label,
                        "discount_rate_pct": 100 * rate,
                        "counterfactual": (
                            "pre-episode linear trend"
                            if counterfactual == "trend"
                            else "flat pre-downturn peak"
                        ),
                        "episodes": len(sample),
                        "eligible_country_years": exposure,
                        "frequency_per_country_year": frequency,
                        "required_avoided_loss_pct_gdp_years": required,
                        "mean_loss_pct_gdp_years": mean_loss,
                        "required_share_of_mean_loss_pct": (
                            100 * required / mean_loss if mean_loss > 0 else np.nan
                        ),
                        "implied_annual_loss_pct_gdp": frequency * mean_loss,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    panel = load_panel()
    episodes = make_episodes(panel)
    paths = gap_paths(panel, episodes)
    flags = flag_episodes(episodes)
    complete = paths[["iso", "start_year"]].drop_duplicates()
    flags = flags.merge(complete, on=["iso", "start_year"])
    summary = summarize(panel, paths, flags)
    RUN.joinpath("results").mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT, index=False, float_format="%.6f")
    flags.to_csv(EPISODE_OUTPUT, index=False)
    loss_3 = episode_losses(paths, flags, 0.03).merge(flags, on=["iso", "start_year"])
    diagnostics = {
        "complete_episodes": len(flags),
        "share_next_start_within_10y": float(flags["next_start_within_10y"].mean()),
        "war_start_episodes": int(flags["war_start"].sum()),
        "war_start_mean_trend_loss_3pct_floored": float(
            loss_3.loc[loss_3["war_start"], "trend_floored"].mean()
        ),
        "other_mean_trend_loss_3pct_floored": float(
            loss_3.loc[~loss_3["war_start"], "trend_floored"].mean()
        ),
    }
    script = Path(__file__).resolve()
    metadata = {
        "run_id": RUN.name,
        "command": "uv run python experiments/break_even_sensitivity.py",
        "random_seed": None,
        "annual_cost_pct": ANNUAL_COST_PCT,
        "war_start_years": "1914-1918 and 1939-1945",
        "code_version": "No Git repository. SHA-256 hashes identify the executed source.",
        "source_sha256": {
            "experiments/break_even_sensitivity.py": sha256(script),
            "experiments/run_severity_analysis.py": sha256(
                ROOT / "experiments/run_severity_analysis.py"
            ),
        },
        "input_sha256": {"data/raw/JSTdatasetR6.dta": sha256(RAW / "JSTdatasetR6.dta")},
        "output_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (OUTPUT, EPISODE_OUTPUT)
        },
        "diagnostics": diagnostics,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.system() + "-" + platform.machine(),
        },
    }
    (RUN / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
