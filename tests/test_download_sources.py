import gzip
import json
import subprocess
import sys
import threading
import zipfile

import pytest

from scripts.ripple.download_sources import (
    digest,
    download,
    summarize_gtopdb,
    validate,
    verify_bindingdb,
)


def test_rejects_success_status_html_login_response(tmp_path):
    path = tmp_path / "features.tsv"
    path.write_text("<!doctype html><html>Log in</html>")
    with pytest.raises(ValueError, match="HTML"):
        validate(path, "tsv")


def test_truncated_gzip_is_not_a_complete_source(tmp_path):
    path = tmp_path / "structure.cif.gz"
    path.write_bytes(gzip.compress(b"data_3J5P\n#\n")[:-5])
    with pytest.raises(EOFError):
        validate(path, "cif_gzip")


def test_modified_cached_file_is_not_reused_or_overwritten(tmp_path):
    path = tmp_path / "features.tsv"
    path.write_text("Entry\tSequence\nP12345\tACDE\n")
    metadata = {
        "url": "https://example.org/features",
        "status": "complete",
        "sha256": digest(path),
    }
    path.with_name(path.name + ".metadata.json").write_text(json.dumps(metadata))
    path.write_text("changed")
    with pytest.raises(ValueError, match="Unverified existing file"):
        download(tmp_path, path.name, metadata["url"])
    assert path.read_text() == "changed"


def test_bindingdb_official_checksum_is_required_to_match(tmp_path):
    parent = tmp_path / "bindingdb"
    parent.mkdir()
    path = parent / "BindingDB_All_202610_tsv.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("BindingDB_All.tsv", "measurement\n")
    checksum = path.with_suffix(".md5")
    checksum.write_text("0" * 32)
    with pytest.raises(ValueError, match="MD5 mismatch"):
        verify_bindingdb(tmp_path, "202610")
    checksum.write_text(digest(path, "md5"))
    assert path.name in verify_bindingdb(tmp_path, "202610")


def test_different_gtopdb_releases_cannot_form_one_snapshot(tmp_path):
    parent = tmp_path / "gtopdb"
    parent.mkdir()
    (parent / "targets.csv").write_text('"# GtoPdb Version: 2026.3.1"\nTarget ID\n1\n')
    (parent / "interactions.csv").write_text(
        '"# GtoPdb Version: 2026.4"\nTarget ID\n1\n'
    )
    with pytest.raises(ValueError, match="release mismatch"):
        summarize_gtopdb(tmp_path)


def test_transport_logs_larger_than_pipe_buffer_do_not_block(tmp_path, monkeypatch):
    real_popen = subprocess.Popen
    timers = []

    def noisy_transport(command, **kwargs):
        script = (
            "import pathlib, sys; "
            "sys.stderr.write('x' * 1048576); sys.stderr.flush(); "
            "pathlib.Path(sys.argv[1]).write_text('Entry\\nP12345\\n'); "
            "print('200\\nhttps://example.org/features')"
        )
        process = real_popen(
            [sys.executable, "-c", script, str(tmp_path / "features.tsv.part")],
            **kwargs,
        )
        timer = threading.Timer(5, process.kill)
        timer.start()
        timers.append(timer)
        return process

    monkeypatch.setattr(subprocess, "Popen", noisy_transport)
    try:
        result = download(
            tmp_path,
            "features.tsv",
            "https://example.org/features",
            "tsv",
            min_free_gb=0,
        )
    finally:
        for timer in timers:
            timer.cancel()
    assert result["status"] == "complete"
    assert (tmp_path / "features.tsv").read_text() == "Entry\nP12345\n"
