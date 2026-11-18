from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.formula.api as smf
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
RUNS = ROOT / "runs"
SEED = 17
BOOTSTRAP_DRAWS = 1000
MAX_HORIZON = 10


def load_panel() -> pd.DataFrame:
    panel, _ = pyreadstat.read_dta(RAW / "JSTdatasetR6.dta")
    panel = panel.sort_values(["iso", "year"]).copy()
    for variable in ("rgdpbarro", "gdp", "tloans", "hpnom", "iy", "cpi"):
        panel[variable] = pd.to_numeric(panel[variable], errors="coerce")
    panel["credit_gdp"] = 100 * panel["tloans"] / panel["gdp"]
    grouped = panel.groupby("iso", sort=False)
    panel["credit_growth_3y"] = grouped["credit_gdp"].diff(3)
    panel["house_price_growth_3y"] = grouped["hpnom"].transform(
        lambda value: 100 * np.log(value).diff(3)
    )
    panel["equity_price_growth_3y"] = grouped["eq_capgain"].transform(
        lambda value: 100 * np.log1p(value).rolling(3).sum()
    )
    panel["real_growth_3y"] = grouped["rgdpbarro"].transform(
        lambda value: 100 * np.log(value).diff(3)
    )
    panel["inflation_3y"] = grouped["cpi"].transform(
        lambda value: 100 * np.log(value).diff(3)
    )
    panel["decade"] = (panel["year"] // 10 * 10).astype(str)
    return panel


def make_episodes(panel: pd.DataFrame) -> pd.DataFrame:
    records = []
    for country, group in panel.groupby("iso", sort=True):
        data = group.set_index("year").sort_index()
        output = data["rgdpbarro"].dropna()
        years = output.index.to_numpy(dtype=int)
        values = output.to_numpy(dtype=float)
        if len(years) < 8:
            continue
        index = 1
        while index < len(years) - 2:
            if years[index] != years[index - 1] + 1 or values[index] >= values[index - 1]:
                index += 1
                continue
            peak_index = index - 1
            peak = values[peak_index]
            horizon_end = min(
                np.searchsorted(years, years[peak_index] + MAX_HORIZON, side="right") - 1,
                len(years) - 1,
            )
            trough_index = index
            recovery = None
            for candidate in range(index, horizon_end + 1):
                if years[candidate] != years[candidate - 1] + 1:
                    break
                if values[candidate] < values[trough_index]:
                    trough_index = candidate
                if candidate > trough_index and values[candidate] >= peak:
                    recovery = candidate
                    break
            start_year = int(years[peak_index])
            trough_year = int(years[trough_index])
            pre_years = np.arange(start_year - 5, start_year)
            pre = data.reindex(pre_years)
            trend_ok = pre["rgdpbarro"].notna().all() and (pre.index.to_numpy()[1:] - pre.index.to_numpy()[:-1] == 1).all()
            if trend_ok:
                slope, intercept = np.polyfit(pre_years, np.log(pre["rgdpbarro"]), 1)
                future_years = np.arange(start_year + 1, min(start_year + MAX_HORIZON, int(years[-1])) + 1)
                actual = data.reindex(future_years)["rgdpbarro"]
                valid = actual.notna()
                trend = slope * future_years[valid] + intercept
                gaps = 100 * (1 - np.exp(np.log(actual[valid].to_numpy()) - trend))
                valid_years = future_years[valid]
                losses_5 = gaps[future_years[valid] <= start_year + 5]
                losses_10 = gaps
                scar_5 = float(gaps[future_years[valid] == start_year + 5][0]) if (future_years[valid] == start_year + 5).any() else np.nan
                scar_10 = float(gaps[future_years[valid] == start_year + 10][0]) if (future_years[valid] == start_year + 10).any() else np.nan
                cumulative_5 = float(np.maximum(losses_5, 0).sum()) if len(losses_5) == 5 else np.nan
                cumulative_10 = float(np.maximum(losses_10, 0).sum()) if len(losses_10) == 10 else np.nan
                peak_gaps = 100 * (1 - actual.to_numpy()[valid] / peak)
                peak_losses_5 = peak_gaps[valid_years <= start_year + 5]
                cumulative_peak_5 = float(np.maximum(peak_losses_5, 0).sum()) if len(peak_losses_5) == 5 else np.nan
                cumulative_peak_10 = float(np.maximum(peak_gaps, 0).sum()) if len(peak_gaps) == 10 else np.nan
                discounted_10 = {
                    rate: float(np.dot(np.maximum(losses_10, 0), (1 + float(rate)) ** -np.arange(1, 11)))
                    if len(losses_10) == 10
                    else np.nan
                    for rate in (0, 0.03, 0.07)
                }
                discounted_peak_10 = {
                    rate: float(np.dot(np.maximum(peak_gaps, 0), (1 + float(rate)) ** -np.arange(1, 11)))
                    if len(peak_gaps) == 10
                    else np.nan
                    for rate in (0, 0.03, 0.07)
                }
            else:
                scar_5 = scar_10 = cumulative_5 = cumulative_10 = np.nan
                cumulative_peak_5 = cumulative_peak_10 = np.nan
                discounted_10 = {rate: np.nan for rate in (0, 0.03, 0.07)}
                discounted_peak_10 = {rate: np.nan for rate in (0, 0.03, 0.07)}
            before = data.reindex(range(start_year - 3, start_year))
            during = data.reindex(range(start_year + 1, min(start_year + MAX_HORIZON, int(years[-1])) + 1))
            unemp_before = before["unemp"].mean()
            unemp_max = during["unemp"].max()
            predictors = {
                "credit_growth_3y": data.reindex([start_year - 1])["credit_growth_3y"].iloc[0],
                "credit_gdp": before["credit_gdp"].mean(),
                "house_price_growth_3y": data.reindex([start_year - 1])["house_price_growth_3y"].iloc[0],
                "equity_price_growth_3y": data.reindex([start_year - 1])["equity_price_growth_3y"].iloc[0],
                "investment_share": before["iy"].mean(),
                "real_growth_3y": data.reindex([start_year - 1])["real_growth_3y"].iloc[0],
                "inflation_3y": data.reindex([start_year - 1])["inflation_3y"].iloc[0],
                "interest_rate": before["stir"].mean(),
            }
            episode_crisis = data.reindex(range(start_year, trough_year + 1))["crisisJST"].fillna(0).max()
            records.append(
                {
                    "iso": country,
                    "start_year": start_year,
                    "trough_year": trough_year,
                    "peak_to_trough_pct": 100 * (1 - values[trough_index] / peak),
                    "duration_years": trough_year - start_year,
                    "recovery_years": int(years[recovery]) - start_year if recovery is not None else np.nan,
                    "recovery_censored_10y": recovery is None and int(years[horizon_end]) >= start_year + MAX_HORIZON,
                    "scar_5y_pct": scar_5,
                    "scar_10y_pct": scar_10,
                    "cumulative_loss_5y_pct_gdp_years": cumulative_5,
                    "cumulative_loss_10y_pct_gdp_years": cumulative_10,
                    "cumulative_loss_vs_peak_5y_pct_gdp_years": cumulative_peak_5,
                    "cumulative_loss_vs_peak_10y_pct_gdp_years": cumulative_peak_10,
                    "discounted_loss_10y_0pct_pct_gdp_years": discounted_10[0],
                    "discounted_loss_10y_3pct_pct_gdp_years": discounted_10[0.03],
                    "discounted_loss_10y_7pct_pct_gdp_years": discounted_10[0.07],
                    "discounted_loss_vs_peak_10y_0pct_pct_gdp_years": discounted_peak_10[0],
                    "discounted_loss_vs_peak_10y_3pct_pct_gdp_years": discounted_peak_10[0.03],
                    "discounted_loss_vs_peak_10y_7pct_pct_gdp_years": discounted_peak_10[0.07],
                    "unemployment_increase_pp": unemp_max - unemp_before if pd.notna(unemp_before) and pd.notna(unemp_max) else np.nan,
                    "systemic_crisis": int(episode_crisis > 0),
                    "decade": str(start_year // 10 * 10),
                    **predictors,
                }
            )
            index = recovery + 1 if recovery is not None else horizon_end + 1
    return pd.DataFrame(records)


def fit_associations(episodes: pd.DataFrame) -> pd.DataFrame:
    core_predictors = [
        "credit_growth_3y",
        "credit_gdp",
        "house_price_growth_3y",
        "investment_share",
        "real_growth_3y",
        "inflation_3y",
        "interest_rate",
    ]
    specifications = [
        ("core predictors", core_predictors),
        ("plus equity returns", [*core_predictors, "equity_price_growth_3y"]),
    ]
    outcomes = [
        "peak_to_trough_pct",
        "cumulative_loss_5y_pct_gdp_years",
        "cumulative_loss_10y_pct_gdp_years",
        "cumulative_loss_vs_peak_5y_pct_gdp_years",
        "cumulative_loss_vs_peak_10y_pct_gdp_years",
        "recovery_years",
        "scar_5y_pct",
        "scar_10y_pct",
        "unemployment_increase_pp",
    ]
    rows = []
    for outcome in outcomes:
        for specification, predictors in specifications:
            required = [outcome, "iso", "decade", *predictors]
            sample = episodes.dropna(subset=required)
            if len(sample) < 40:
                continue
            formula = f"{outcome} ~ " + " + ".join(predictors) + " + C(iso) + C(decade)"
            fit = smf.ols(formula, data=sample).fit(
                cov_type="cluster", cov_kwds={"groups": sample["iso"]}
            )
            for variable in predictors:
                rows.append(
                    {
                        "outcome": outcome,
                        "predictor": variable,
                        "specification": specification,
                        "coefficient": fit.params[variable],
                        "cluster_se": fit.bse[variable],
                        "p_value": 2 * student_t.sf(abs(fit.params[variable] / fit.bse[variable]), df=sample["iso"].nunique() - 1),
                        "ci_low_95": fit.params[variable] - student_t.ppf(0.975, df=sample["iso"].nunique() - 1) * fit.bse[variable],
                        "ci_high_95": fit.params[variable] + student_t.ppf(0.975, df=sample["iso"].nunique() - 1) * fit.bse[variable],
                        "n_episodes": len(sample),
                        "countries": sample["iso"].nunique(),
                        "r_squared": fit.rsquared,
                        "sample": "all downturns",
                    }
                )
    return pd.DataFrame(rows)


def eligible_exposure(panel: pd.DataFrame) -> dict[str, int]:
    exposure = {}
    for country, group in panel.groupby("iso"):
        output = group.set_index("year")["rgdpbarro"].dropna()
        available_years = set(output.index.astype(int))
        exposure[country] = sum(
            all(year in available_years for year in range(start - 5, start + MAX_HORIZON + 1))
            for start in available_years
        )
    return exposure


def build_frontier(episodes: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    exposure = eligible_exposure(panel)
    rows = []
    for discount_rate in (0, 0.03, 0.07):
        for counterfactual, prefix in (("pre-episode linear trend", "discounted_loss_10y"), ("flat pre-downturn peak", "discounted_loss_vs_peak_10y")):
            complete = episodes.dropna(subset=[f"{prefix}_{int(discount_rate * 100)}pct_pct_gdp_years"])
            damage = complete[f"{prefix}_{int(discount_rate * 100)}pct_pct_gdp_years"].to_numpy(dtype=float)
            annual_frequency = len(complete) / sum(exposure.values())
            mean_damage = float(np.mean(damage))
            for growth_cost in (0.05, 0.1, 0.25, 0.5, 1.0):
                required_damage = growth_cost / annual_frequency
                for percentile in (50, 75, 90, 95):
                    reference_damage = float(np.quantile(damage, percentile / 100))
                    rows.append(
                        {
                            "annual_growth_cost_pct": growth_cost,
                            "downturn_frequency_per_country_year": annual_frequency,
                            "damage_percentile": percentile,
                            "discount_rate_pct": discount_rate * 100,
                            "counterfactual": counterfactual,
                            "reference_damage_pct_gdp_years": reference_damage,
                            "required_damage_avoided_pct_gdp_years_per_downturn": required_damage,
                            "required_share_of_reference_damage_pct": 100 * required_damage / reference_damage if reference_damage > 0 else np.nan,
                            "mean_damage_pct_gdp_years": mean_damage,
                            "frequency_held_fixed": True,
                        }
                    )
    return pd.DataFrame(rows)


def bootstrap_frontier(episodes: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    countries = np.array(sorted(panel["iso"].unique()))
    exposure = eligible_exposure(panel)
    by_country = {country: group for country, group in episodes.groupby("iso")}
    rng = np.random.default_rng(SEED)
    rows = []
    for draw in range(BOOTSTRAP_DRAWS):
        selected = rng.choice(countries, size=len(countries), replace=True)
        sample = pd.concat([by_country[country] for country in selected if country in by_country])
        for discount_rate in (0, 0.03, 0.07):
            for counterfactual, prefix in (("pre-episode linear trend", "discounted_loss_10y"), ("flat pre-downturn peak", "discounted_loss_vs_peak_10y")):
                complete = sample.dropna(subset=[f"{prefix}_{int(discount_rate * 100)}pct_pct_gdp_years"])
                damage = complete[f"{prefix}_{int(discount_rate * 100)}pct_pct_gdp_years"].to_numpy()
                if len(damage) == 0:
                    continue
                frequency = len(complete) / sum(exposure[country] for country in selected)
                mean_damage = float(damage.mean())
                for percentile in (50, 75, 90, 95):
                    reference_damage = float(np.quantile(damage, percentile / 100))
                    for growth_cost in (0.05, 0.1, 0.25, 0.5, 1.0):
                        required = growth_cost / frequency
                        rows.append(
                            {
                                "draw": draw,
                                "annual_growth_cost_pct": growth_cost,
                                "discount_rate_pct": discount_rate * 100,
                                "counterfactual": counterfactual,
                                "damage_percentile": percentile,
                                "frequency_per_country_year": frequency,
                                "required_damage_pct_gdp_years": required,
                                "required_share_of_reference_damage_pct": 100 * required / reference_damage if reference_damage > 0 else np.nan,
                                "required_share_of_mean_damage_pct": 100 * required / mean_damage if mean_damage > 0 else np.nan,
                            }
                        )
    draws = pd.DataFrame(rows)
    keys = ["annual_growth_cost_pct", "discount_rate_pct", "counterfactual", "damage_percentile"]
    summaries = draws.groupby(keys).agg(
        bootstrap_draws=("draw", "nunique"),
        frequency_ci_low=("frequency_per_country_year", lambda value: value.quantile(0.025)),
        frequency_ci_high=("frequency_per_country_year", lambda value: value.quantile(0.975)),
        required_damage_ci_low=("required_damage_pct_gdp_years", lambda value: value.quantile(0.025)),
        required_damage_ci_high=("required_damage_pct_gdp_years", lambda value: value.quantile(0.975)),
        required_share_ci_low=("required_share_of_reference_damage_pct", lambda value: value.quantile(0.025)),
        required_share_ci_high=("required_share_of_reference_damage_pct", lambda value: value.quantile(0.975)),
        required_share_of_mean_ci_low=("required_share_of_mean_damage_pct", lambda value: value.quantile(0.025)),
        required_share_of_mean_ci_high=("required_share_of_mean_damage_pct", lambda value: value.quantile(0.975)),
    ).reset_index()
    return summaries


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    panel = load_panel()
    episodes = make_episodes(panel)
    assert len(episodes) > 50
    assert episodes["peak_to_trough_pct"].gt(0).all()
    assert episodes["recovery_years"].dropna().ge(0).all()
    estimates = fit_associations(episodes)
    frontier = build_frontier(episodes, panel)
    frontier_uncertainty = bootstrap_frontier(episodes, panel)
    episodes.to_csv(RESULTS / "downturn_episodes.csv", index=False)
    estimates.to_csv(RESULTS / "downturn_severity_associations.csv", index=False)
    frontier.to_csv(RESULTS / "severity_insurance_frontier.csv", index=False)
    frontier_uncertainty.to_csv(RESULTS / "severity_insurance_frontier_uncertainty.csv", index=False)
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    selected = frontier.loc[(frontier["discount_rate_pct"] == 3) & (frontier["counterfactual"] == "pre-episode linear trend")]
    for percentile, group in selected.groupby("damage_percentile"):
        axis.plot(
            group["annual_growth_cost_pct"],
            group["required_share_of_reference_damage_pct"],
            marker="o",
            label=f"{percentile}th percentile loss",
        )
    axis.axhline(100, color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="Annual output-equivalent growth cost (%)", ylabel="Damage reduction required (%)", title="Severity-insurance break-even frontier")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(RESULTS / "severity_insurance_frontier.png", dpi=180)
    plt.close(figure)
    print(f"Downturn episodes: {len(episodes)}")
    print(f"Countries: {episodes['iso'].nunique()}")
    print(f"Systemic-crisis episodes: {episodes['systemic_crisis'].sum()}")
    print(f"Median peak-to-trough decline: {episodes['peak_to_trough_pct'].median():.3f}%")
    print(estimates.loc[(estimates["outcome"] == "cumulative_loss_5y_pct_gdp_years") & (estimates["predictor"] == "credit_growth_3y")].to_string(index=False))
    print(frontier.loc[(frontier["annual_growth_cost_pct"] == 0.1) & (frontier["damage_percentile"] == 50)].to_string(index=False))


if __name__ == "__main__":
    main()
