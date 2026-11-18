import hashlib
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch
from zipfile import BadZipFile, ZipFile

from experiments import acquire_data


def valid_zip() -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("data.txt", "valid")
    return buffer.getvalue()


class AcquisitionValidationTests(TestCase):
    def test_validate_rejects_empty_files(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "empty.xlsx"
            path.touch()
            with self.assertRaisesRegex(ValueError, "empty"):
                acquire_data.validate(path)

    def test_validate_rejects_invalid_workbooks(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.xlsx"
            path.write_bytes(b"not a workbook")
            with self.assertRaises(BadZipFile):
                acquire_data.validate(path)

    def test_download_removes_invalid_partial_file(self) -> None:
        with TemporaryDirectory() as directory:
            with (
                patch.object(acquire_data, "urlopen", return_value=BytesIO(b"bad")),
                self.assertRaises(BadZipFile),
            ):
                acquire_data.download(
                    Path(directory) / "source.xlsx", "https://example.invalid", "0" * 64
                )
            self.assertFalse((Path(directory) / "source.xlsx").exists())
            self.assertFalse((Path(directory) / ".source.download.xlsx").exists())

    def test_download_installs_only_valid_file(self) -> None:
        with TemporaryDirectory() as directory:
            payload = valid_zip()
            with patch.object(acquire_data, "urlopen", return_value=BytesIO(payload)):
                acquire_data.download(
                    Path(directory) / "source.xlsx",
                    "https://example.invalid",
                    hashlib.sha256(payload).hexdigest(),
                )
            self.assertTrue((Path(directory) / "source.xlsx").is_file())
            self.assertFalse((Path(directory) / ".source.download.xlsx").exists())

    def test_download_rejects_hash_mismatch(self) -> None:
        with TemporaryDirectory() as directory:
            with (
                patch.object(
                    acquire_data, "urlopen", return_value=BytesIO(valid_zip())
                ),
                self.assertRaisesRegex(ValueError, "SHA-256 mismatch"),
            ):
                acquire_data.download(
                    Path(directory) / "source.xlsx", "https://example.invalid", "0" * 64
                )
            self.assertFalse((Path(directory) / "source.xlsx").exists())
            self.assertFalse((Path(directory) / ".source.download.xlsx").exists())

    def test_download_rejects_oversized_response(self) -> None:
        with TemporaryDirectory() as directory:
            with (
                patch.object(acquire_data, "MAX_DOWNLOAD_BYTES", 10),
                patch.object(acquire_data, "urlopen", return_value=BytesIO(b"x" * 11)),
                self.assertRaisesRegex(ValueError, "exceeds"),
            ):
                acquire_data.download(
                    Path(directory) / "source.xlsx", "https://example.invalid", "0" * 64
                )
            self.assertFalse((Path(directory) / ".source.download.xlsx").exists())

    def test_extract_all_rejects_path_traversal(self) -> None:
        with TemporaryDirectory() as directory:
            archive_path = Path(directory) / "unsafe.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("../escape.txt", "x")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                acquire_data.extract_all(archive_path, Path(directory) / "out")
            self.assertFalse((Path(directory) / "escape.txt").exists())

    def test_existing_source_files_match_pinned_hashes(self) -> None:
        for source in acquire_data.SOURCES:
            path = acquire_data.ROOT / source.relative_path
            if not path.exists():
                self.skipTest(f"{source.relative_path} not downloaded")
            self.assertEqual(acquire_data.sha256(path), source.sha256)

    def test_default_selection_is_only_the_papers_sources(self) -> None:
        default = acquire_data.selected_sources(include_unused=False)
        self.assertEqual(
            [source.relative_path for source in default],
            [
                "data/raw/JSTdatasetR6.dta",
                "runs/jst-2013-benchmark-01/source/WCBB_replication.zip",
            ],
        )
        self.assertEqual(len(acquire_data.selected_sources(include_unused=True)), 4)


if __name__ == "__main__":
    main()
