# Data notice

These result files are derived from the Jordà-Schularick-Taylor Macrohistory Database,
release 6. That database is licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). These derived files
are shared under the same license: CC BY-NC-SA 4.0. Use them for non-commercial purposes
only. Give credit, and share any adaptations under the same license.

Cite the database as: Jordà, Ò., Schularick, M., and Taylor, A. M. (2017). Macrofinancial
history and the new business cycle facts. NBER Macroeconomics Annual 31, 213-263.

Files covered:

- `results/downturn_episodes.csv`
- `results/downturn_severity_associations.csv`
- `results/severity_insurance_frontier.csv`
- `results/severity_insurance_frontier_uncertainty.csv`
- `results/holm_adjusted_pvalues.csv`
- `runs/severity-insurance-01/` result hashes and metadata
- `runs/jst-r6-vintage-compare-01/results/`
- `runs/jst-r6-event-rules-01/results/`
- `runs/jst-r6-spec-bridge-01/results/`
- `runs/jst-brybos-frontier-01/results/`
- `runs/break-even-sensitivity-01/results/`

Some of these files also use event dates from the Jordà, Schularick, and Taylor (2013)
replication archive. That archive states no license, so it is not included here.
`runs/jst-2013-benchmark-01/results/jst_table8_reproduction.csv` contains only regression
estimates from that archive's panel.

The source code is under the MIT License in `LICENSE`. The MIT License does not cover these
data files.

The share-of-loss columns in `results/severity_insurance_frontier.csv` and
`results/severity_insurance_frontier_uncertainty.csv` are not results of the note. They use
a loss measure that floors yearly gaps at zero and counts overlapping windows.
