"""Download the third-party source files, check their SHA-256 hashes, and record them.

The project does not redistribute these files. Each one keeps its owner's license.
By default, only the two sources that the paper uses are downloaded: JST release 6 and
the JST (2013) replication archive. Use --include-unused for the BEA and Kim and Kung files
that only out-of-scope work uses.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "source_manifest.json"
WCBB_DIR = ROOT / "runs" / "jst-2013-benchmark-01" / "source"
DOWNLOAD_TIMEOUT_SECONDS = 60
MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024


@dataclass(frozen=True)
class Source:
    relative_path: str
    url: str
    sha256: str
    license: str
    used_by_paper: bool


SOURCES = (
    Source(
        "data/raw/JSTdatasetR6.dta",
        "https://www.macrohistory.net/app/download/9834512469/JSTdatasetR6.dta?t=1720600177",
        "b0ebb74a8d1b5b1bc9033fc46a6dcc578736afff8ce1e1086b7840f2649e79b3",
        "CC BY-NC-SA 4.0, Jorda-Schularick-Taylor Macrohistory Database release 6",
        True,
    ),
    Source(
        "runs/jst-2013-benchmark-01/source/WCBB_replication.zip",
        "https://www.moritzschularick.com/app/download/12556526299/"
        "WCBB_replication_14aug2015.zip.zip?t=1763503201",
        "5e61b6492115bf2adc13ac63f86bb4e7846dda47c5b0c07dab687ed1dd298899",
        "No license stated in the archive or on the download page. Not redistributed.",
        True,
    ),
    Source(
        "data/raw/DetailNonres_rate.xlsx",
        "https://apps.bea.gov/national/FA2004/Details/xls/DetailNonres_rate.xlsx",
        "017d0dccc30f6ad933c17871411fb6f43df3b6856369b0c4227e330a323d70db",
        "U.S. government work (BEA)",
        False,
    ),
    Source(
        "data/raw/redeployability_data_RFS_2024v.zip",
        "https://www.dropbox.com/scl/fi/rxpges91vpgijcx47jq04/"
        "redeployability_data_RFS_2024v.zip?rlkey=qkp3sd3j5ujguutxkd0t170n3&dl=1",
        "e96f66c02376a4695016edd159ac466a9a7abc49cb789aad91f031900534b9ea",
        "No license stated on the author download page. Not redistributed.",
        False,
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(path: Path) -> None:
    if path.stat().st_size == 0:
        raise ValueError(f"Downloaded file is empty: {path.name}")
    if path.suffix in {".zip", ".xlsx"}:
        with ZipFile(path) as archive:
            invalid = archive.testzip()
            if invalid is not None:
                raise ValueError(f"File has invalid data: {path.name} ({invalid})")
    elif path.suffix == ".dta":
        pyreadstat.read_dta(path, metadataonly=True)


def check_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: expected {expected}, got {actual}. "
            "The source may have changed. Do not use it until you review the new version."
        )


def download(destination: Path, url: str, expected_sha256: str) -> None:
    if destination.exists():
        validate(destination)
        check_hash(destination, expected_sha256)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.download{destination.suffix}"
    )
    request = Request(url, headers={"User-Agent": "research-data-client/0.1"})
    try:
        with (
            urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
            temporary.open("wb") as output,
        ):
            copy_bounded(response, output)
        validate(temporary)
        check_hash(temporary, expected_sha256)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def copy_bounded(source, output) -> None:
    total_bytes = 0
    for block in iter(lambda: source.read(1 << 20), b""):
        total_bytes += len(block)
        if total_bytes > MAX_DOWNLOAD_BYTES:
            raise ValueError(f"Download exceeds {MAX_DOWNLOAD_BYTES} bytes")
        output.write(block)


def extract_member(archive_path: Path, member: str, target_dir: Path) -> None:
    if (target_dir / member).exists():
        return
    try:
        with ZipFile(archive_path) as archive:
            archive.extract(member, target_dir)
    except Exception:
        (target_dir / member).unlink(missing_ok=True)
        raise


def extract_all(archive_path: Path, target_dir: Path) -> None:
    with ZipFile(archive_path) as archive:
        for name in archive.namelist():
            resolved = (target_dir / name).resolve()
            if not resolved.is_relative_to(target_dir.resolve()):
                raise ValueError(f"Unsafe archive member path: {name}")
        archive.extractall(target_dir)


def selected_sources(include_unused: bool) -> tuple[Source, ...]:
    return tuple(source for source in SOURCES if source.used_by_paper or include_unused)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-unused",
        action="store_true",
        help="Also download the BEA and Kim and Kung files, which the paper does not use.",
    )
    arguments = parser.parse_args(argv)
    sources = selected_sources(arguments.include_unused)
    for source in sources:
        download(ROOT / source.relative_path, source.url, source.sha256)
    if arguments.include_unused:
        extract_member(
            RAW / "redeployability_data_RFS_2024v.zip",
            "redeployability_data_RFS_2024v.xlsx",
            RAW,
        )
        validate(RAW / "redeployability_data_RFS_2024v.xlsx")
    wcbb_archive = WCBB_DIR / "archive"
    if not (wcbb_archive / "WCBB_replication_14aug2015").exists():
        extract_all(WCBB_DIR / "WCBB_replication.zip", wcbb_archive)
    manifest = [
        {
            "path": source.relative_path,
            "url": source.url,
            "sha256": source.sha256,
            "license": source.license,
            "used_by_paper": source.used_by_paper,
        }
        for source in sources
    ]
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
