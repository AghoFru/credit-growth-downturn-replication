"""Holm adjustment of the saved credit-growth p-values, by family of tests."""

from pathlib import Path

import pandas as pd
from scipy.stats import t as student_t
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
SEVERITY = ROOT / "results/downturn_severity_associations.csv"
EVENT_RULES = ROOT / "runs/jst-r6-event-rules-01/results/event_rule_associations.csv"
OUTPUT = ROOT / "results/holm_adjusted_pvalues.csv"


def severity_families() -> pd.DataFrame:
    frame = pd.read_csv(SEVERITY)
    frame = frame[frame["predictor"] == "credit_growth_3y"].copy()
    frame["family"] = "severity outcomes, " + frame["specification"]
    frame["label"] = frame["outcome"]
    return frame[
        ["family", "label", "coefficient", "cluster_se", "p_value", "countries"]
    ]


def event_rule_family() -> pd.DataFrame:
    frame = pd.read_csv(EVENT_RULES)
    # Same inference as the source script: t with (clusters - 1) degrees of freedom.
    statistic = (frame["coefficient"] / frame["cluster_se"]).abs()
    frame["p_value"] = 2 * student_t.sf(statistic, frame["countries"] - 1)
    frame["family"] = "event-rule comparison"
    frame["label"] = frame["sample"] + " | " + frame["outcome"]
    return frame[
        ["family", "label", "coefficient", "cluster_se", "p_value", "countries"]
    ]


def main() -> None:
    tests = pd.concat([severity_families(), event_rule_family()], ignore_index=True)
    if tests["p_value"].isna().any():
        raise ValueError("Missing p-values in saved results")
    tests["family_size"] = tests.groupby("family")["p_value"].transform("size")
    tests["holm_p_value"] = tests.groupby("family")["p_value"].transform(
        lambda values: multipletests(values, method="holm")[1]
    )
    tests.to_csv(OUTPUT, index=False, float_format="%.6f")
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
