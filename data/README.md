# Data

The paper uses two third-party sources. The project does not redistribute either of them.
Download them with:

```sh
uv run python experiments/acquire_data.py
```

The script downloads each file, checks it against the pinned SHA-256 hash below, and stops
if the hash differs. It then extracts the replication archive and writes
`data/source_manifest.json` with the URL, hash, and license of each file. `.gitignore`
excludes the downloaded files.

## Sources

| Local path | Source | SHA-256 | License |
| --- | --- | --- | --- |
| `data/raw/JSTdatasetR6.dta` | [Jordà-Schularick-Taylor Macrohistory Database](https://www.macrohistory.net/database/), release 6. 18 advanced economies, 1870 to 2020, 2,718 country-years. | `b0ebb74a8d1b5b1bc9033fc46a6dcc578736afff8ce1e1086b7840f2649e79b3` | CC BY-NC-SA 4.0 |
| `runs/jst-2013-benchmark-01/source/WCBB_replication.zip` | "When Credit Bites Back" replication archive, version of 14 August 2015, from the [Schularick datasets page](https://www.moritzschularick.com/academic/datasets-and-codes/). It has the original panel `private.dta`, the Stata programs, and `tables/table8.tex`. | `5e61b6492115bf2adc13ac63f86bb4e7846dda47c5b0c07dab687ed1dd298899` | No license stated. Do not redistribute. |

The JST release 6 file was first downloaded on 4 November 2026. On 18 November 2026, a
fresh download from the same URL gave the same hash. The archive was downloaded and
checked on 18 November 2026 with the same result.

Cite the database as Jordà, Schularick, and Taylor (2017), "Macrofinancial History and the
New Business Cycle Facts", NBER Macroeconomics Annual 31, 213-263. Cite the archive as the
replication files for Jordà, Schularick, and Taylor (2013), "When Credit Bites Back",
Journal of Money, Credit and Banking 45(S2), 3-28.

## Limits

The JST database is revised between releases. Release 6 `tloans` is the closest series to
the archive's `loans1`, but the two definitions are not confirmed to be identical. The panel
is unbalanced, and each analysis uses complete cases, so samples differ by outcome.

Result files derived from JST data are under CC BY-NC-SA 4.0. See
`results/DATA_NOTICE.md`.

`uv run python experiments/acquire_data.py --include-unused` also downloads two files that
the paper does not use: BEA depreciation rates and the Kim and Kung redeployability index.
