# Credit Growth and Downturn Severity: A Replication and Fragility Note

This repository checks whether the credit-severity link in Jordà, Schularick, and Taylor
(2013), "When Credit Bites Back", holds on revised data. It re-estimates their Table 8,
reruns the relation on release 6 of the Jordà-Schularick-Taylor Macrohistory Database, and
converts a restraint cost into the avoided loss that it would need to break even.

Paper: `paper/output/pdf/A Replication and Fragility Note on Credit Growth and Downturn Severity.pdf`

## Claim

An independent Python implementation of JST (2013) Table 8 matches the authors' archived
output at three decimals. On JST release 6 data, the association between pre-downturn
credit growth and downturn severity is imprecise and sensitive to data vintage, episode
dating, and controls. A 0.1% annual output cost requires 0.87 GDP-percent-years of avoided
loss per downturn to break even. That number is arithmetic by construction: the cost
divided by the observed downturn frequency of 0.114 per country-year.

| Result | Value |
| --- | --- |
| Table 8 check | 20 of 20 coefficients and standard errors match the archived `table8.tex` at three decimals, 121 events |
| Vintage, 116 common events | Credit interactions at horizons 1 to 3 move toward zero. None changes sign. No formal difference test |
| Five-year loss, annual-decline rule | -1.097 (SE 0.446), raw p 0.029, Holm p 0.229 |
| Five-year loss, JST chronology | -0.013 (SE 0.432) |
| Five-year loss, matched dates | -0.401 (SE 0.501) |
| Nine severity outcomes, 144 to 178 episodes | All Holm-adjusted p values are 1.000 |
| Break-even for a 0.1% annual cost | 0.87 GDP-percent-years per downturn |

The nine-outcome design uses three-year credit growth with house-price, other macro,
country, and decade controls. It is not the JST design.

## Limits

- The Table 8 check compares against the archive's `table8.tex`, in Python, not Stata. It
  was not checked against the printed 2013 journal table.
- Inference uses 14 to 18 country clusters.
- The vintage comparison cannot separate data revision from a possible change in the credit
  definition. The dating comparison cannot separate the dating rule from sample composition.
- All estimates are conditional associations. The note makes no causal or policy claim and
  does not estimate how any policy changes downturn loss.
- The note does not express 0.87 as a share of measured loss. The loss measure floors yearly
  gaps at zero, counts overlapping ten-year windows, and includes war-year downturns. Even
  after corrections it implies a trend loss of 4.5% to 11.9% of GDP each year. The
  share-of-loss columns in `results/severity_insurance_frontier.csv` and
  `results/severity_insurance_frontier_uncertainty.csv` are therefore not results of the
  note. `runs/break-even-sensitivity-01/` has the corrected loss measures.
- The severity outcomes use the same floored, overlapping loss windows.

## Reproduce

The data are not in this repository. `experiments/acquire_data.py` downloads JST release 6
and the JST (2013) replication archive, checks each file against a pinned SHA-256 hash,
stops on a mismatch, and writes `data/source_manifest.json`. See `data/README.md` for the
sources.

```sh
uv sync
uv run python experiments/acquire_data.py
uv run python experiments/run_severity_analysis.py
uv run python experiments/reproduce_jst_table8.py
uv run python experiments/compare_jst2013_r6.py
uv run python experiments/compare_jst_r6_event_rules.py
uv run python experiments/decompose_jst_r6_specification.py
uv run python experiments/build_jst_brybos_frontier.py
uv run python experiments/adjust_multiple_tests.py
uv run python experiments/break_even_sensitivity.py
uv run python -m unittest discover -s tests -v
```

The test suite has 12 tests. Each run folder under `runs/` records its command, input and
code hashes, outputs, and environment.

Build the paper from `paper/`:

```sh
pdflatex -jobname="A Replication and Fragility Note on Credit Growth and Downturn Severity" -output-directory=output/pdf paper.tex
```

## License

The code is under the MIT License (`LICENSE`). Result files derived from JST data are under
CC BY-NC-SA 4.0 (`results/DATA_NOTICE.md`). The JST database is CC BY-NC-SA 4.0. The JST
(2013) replication archive states no license, so it is not included.
