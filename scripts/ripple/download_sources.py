"""Collect public source snapshots with integrity checks and per-file provenance.

Uses Python's standard library and curl; no model or GPU dependencies are needed.
Completed files are reused only when their recorded URL and SHA256 still match.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 - server uses Python 3.10


def digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def validate(path, kind):
    if not path.stat().st_size:
        raise ValueError("Empty response")
    details = {}
    if kind == "zip":
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"ZIP CRC failure: {bad}")
            details["archive_members"] = archive.namelist()
        return details
    opener = gzip.open if kind in {"gzip", "cif_gzip", "xml_gzip"} else open
    with opener(path, "rb") as handle:
        first = handle.read(8192)
        if kind in {"gzip", "cif_gzip", "xml_gzip"}:
            for _ in iter(lambda: handle.read(1024 * 1024), b""):
                pass
    if kind != "html" and (
        b"<html" in first.lower() or b"<!doctype html" in first.lower()
    ):
        raise ValueError("Received HTML instead of source data")
    if kind == "json":
        json.loads(path.read_text())
    if kind == "cif_gzip" and not first.lstrip().startswith(b"data_"):
        raise ValueError("Invalid mmCIF response")
    if kind == "xml_gzip" and b"<" not in first:
        raise ValueError("Invalid XML response")
    if kind in {"csv", "tsv", "text", "gzip"}:
        details["first_line"] = first.decode(
            "utf-8-sig", errors="replace"
        ).splitlines()[0][:500]
    return details


def download(
    raw, relative, url, kind="text", min_free_gb=15, max_file_gb=8, quiet=False
):
    path = raw / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = path.with_name(path.name + ".metadata.json")
    previous = json.loads(metadata.read_text()) if metadata.exists() else {}
    if path.exists():
        if (
            previous.get("url") == url
            and previous.get("status") == "complete"
            and digest(path) == previous.get("sha256")
        ):
            return previous
        if previous:
            previous.update(
                status="invalid",
                error="Cached file, URL or SHA256 does not match",
                checked_at_utc=now(),
            )
            dump(metadata, previous)
        raise ValueError(f"Unverified existing file; inspect before replacing: {path}")
    part = path.with_name(path.name + ".part")
    headers = path.with_name(path.name + ".headers.part")
    record = {
        "path": relative,
        "url": url,
        "started_at_utc": now(),
        "status": "downloading",
    }
    dump(metadata, record)
    command = [
        "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry",
        "2",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "20",
        "--max-time",
        "1800",
        "--speed-limit",
        "1024",
        "--speed-time",
        "60",
        "--max-filesize",
        str(int(max_file_gb * 2**30)),
        "--dump-header",
        str(headers),
        "--output",
        str(part),
        "--write-out",
        "%{http_code}\n%{url_effective}\n",
        url,
    ]
    use_aria = (
        kind == "zip" and relative.startswith("bindingdb/") and shutil.which("aria2c")
    )
    if use_aria:
        # BindingDB archives have versioned URLs and an official checksum.
        command = [
            "aria2c",
            "--continue=true",
            "--file-allocation=none",
            "--split=4",
            "--max-connection-per-server=4",
            "--min-split-size=8M",
            "--max-tries=5",
            "--retry-wait=5",
            "--timeout=60",
            "--connect-timeout=20",
            "--summary-interval=0",
            "--show-console-readout=false",
            "--console-log-level=warn",
            "--download-result=hide",
            "--auto-file-renaming=false",
            "--allow-overwrite=true",
            "--dir",
            str(path.parent),
            "--out",
            part.name,
            url,
        ]
    record["transport"] = "aria2c" if use_aria else "curl"
    process = None
    try:
        with (
            tempfile.TemporaryFile(mode="w+t") as output_file,
            tempfile.TemporaryFile(mode="w+t") as error_file,
        ):
            if shutil.disk_usage(raw).free < min_free_gb * 2**30:
                raise OSError(f"Less than {min_free_gb} GiB free; download paused")
            process = subprocess.Popen(
                command, stdout=output_file, stderr=error_file, text=True
            )
            while process.poll() is None:
                if shutil.disk_usage(raw).free < min_free_gb * 2**30:
                    raise OSError(f"Less than {min_free_gb} GiB free; download paused")
                if part.exists() and part.stat().st_size > max_file_gb * 2**30:
                    raise OSError("Download exceeds per-file size limit")
                time.sleep(1)
            process.wait()
            output_file.seek(0)
            error_file.seek(0)
            output, error = output_file.read(), error_file.read()
        if process.returncode:
            raise RuntimeError(
                f"{record['transport']} {process.returncode}: {(error + output).strip()[-1500:]}"
            )
        record.update(validate(part, kind))
        record.update(
            status="complete",
            retrieved_at_utc=now(),
            bytes=part.stat().st_size,
            sha256=digest(part),
        )
        if not use_aria:
            record.update(
                http_status=output.splitlines()[0], final_url=output.splitlines()[1]
            )
        if headers.exists():
            record["response_headers"] = [
                line
                for line in headers.read_text(errors="replace").splitlines()
                if line.lower().startswith(
                    (
                        "last-modified:",
                        "etag:",
                        "content-length:",
                        "content-type:",
                        "x-uniprot-release:",
                    )
                )
            ]
        part.replace(path)
        dump(metadata, record)
        headers.unlink(missing_ok=True)
        if not quiet:
            print(f"{now()} OK {relative} ({record['bytes']:,} bytes)", flush=True)
        return record
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        record.update(status="failed", error=str(error), finished_at_utc=now())
        dump(metadata, record)
        raise


def collect(raw, jobs, workers=3, min_free_gb=15):
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                download, raw, *job, min_free_gb=min_free_gb, quiet=len(jobs) > 50
            ): job[0]
            for job in jobs
        }
        for number, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except (
                OSError,
                ValueError,
                RuntimeError,
                EOFError,
                zipfile.BadZipFile,
            ) as error:
                failure = {"path": futures[future], "error": str(error)}
                failures.append(failure)
                print(f"FAILED {failure}", flush=True)
            if len(jobs) > 50 and (number % 100 == 0 or number == len(jobs)):
                print(
                    f"{now()} Completed {number}/{len(jobs)}, failures={len(failures)}",
                    flush=True,
                )
    return failures


def core_jobs(release):
    jobs = []
    gtop = "https://www.guidetopharmacology.org/DATA/"
    for name in (
        "interactions.csv",
        "ligands.csv",
        "targets_and_families.csv",
        "targets.csv",
        "ligand_id_mapping.csv",
        "file_descriptions.txt",
    ):
        jobs.append(
            ("gtopdb/" + name, gtop + name, "csv" if name.endswith("csv") else "text")
        )
    bdb = "https://www.bindingdb.org/rwd/bind/downloads/"
    for prefix in ("BindingDB_All", "BindingDB_Assays", "BindingDB_rsid_eaids"):
        name = f"{prefix}_{release}_tsv"
        jobs.extend(
            [
                (f"bindingdb/{name}.zip", f"{bdb}{name}.zip", "zip"),
                (f"bindingdb/{name}.md5", f"{bdb}{name}.md5", "text"),
            ]
        )
    bio = "https://zhanggroup.org/BioLiP/"
    for name, remote in (
        ("BioLiP.txt.gz", "download/BioLiP.txt.gz"),
        ("ligand.tsv.gz", "data/ligand.tsv.gz"),
        ("protein.fasta.gz", "data/protein.fasta.gz"),
        ("readme.txt", "download/readme.txt"),
        ("readme_ligand.txt", "download/readme_ligand.txt"),
    ):
        jobs.append(
            ("biolip/" + name, bio + remote, "gzip" if name.endswith("gz") else "text")
        )
    jobs.append(
        (
            "sifts/pdb_chain_uniprot.csv.gz",
            "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz",
            "gzip",
        )
    )
    return jobs


def verify_bindingdb(raw, release):
    result = {}
    for prefix in ("BindingDB_All", "BindingDB_Assays", "BindingDB_rsid_eaids"):
        base = raw / "bindingdb" / f"{prefix}_{release}_tsv"
        archive, checksum = base.with_suffix(".zip"), base.with_suffix(".md5")
        if archive.exists() and checksum.exists():
            tokens = checksum.read_text().replace("=", " ").split()
            expected = [
                token.lower()
                for token in tokens
                if len(token) == 32
                and all(c in "0123456789abcdefABCDEF" for c in token)
            ]
            actual = digest(archive, "md5")
            if actual not in expected:
                raise ValueError(f"Official MD5 mismatch: {archive}")
            result[archive.name] = actual
    return result


def register_pdbbind(raw, checksums):
    for name, expected in zip(("index.tar.gz", "P-L.tar.gz"), checksums, strict=True):
        path = raw / "pdbbind" / name
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"PDBbind transfer checksum mismatch: {name}")
        validate(path, "gzip")
        dump(
            path.with_name(path.name + ".metadata.json"),
            {
                "path": "pdbbind/" + name,
                "url": "https://static.pdbbind-plus.org.cn/v2020R1/" + name,
                "version": "2020.R1",
                "status": "complete",
                "bytes": path.stat().st_size,
                "sha256": actual,
                "verified_at_utc": now(),
                "acquired_via": "User-provided local archive transferred over SSH",
                "original_retrieval_at_utc": None,
            },
        )


def summarize_gtopdb(raw):
    releases = set()
    counts = {}
    for path in sorted((raw / "gtopdb").glob("*.csv")):
        with path.open(encoding="utf-8-sig") as handle:
            releases.add(next(handle).strip())
            reader = csv.DictReader(handle)
            count = 0
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Malformed GtoPdb record in {path}")
                count += 1
            counts[path.name] = count
    if len(releases) != 1:
        raise ValueError(f"GtoPdb release mismatch: {sorted(releases)}")
    result = {"release": releases.pop(), "record_counts": counts}
    dump(raw / "gtopdb/summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--bindingdb-release", default="202610")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--min-free-gb", type=float, default=15)
    parser.add_argument(
        "--import-pdbbind",
        nargs=2,
        metavar=("INDEX_SHA256", "PL_SHA256"),
        help="Register copied PDBbind files after matching their local SHA256 hashes",
    )
    args = parser.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    if args.import_pdbbind:
        register_pdbbind(args.raw, args.import_pdbbind)
    failures = collect(
        args.raw, core_jobs(args.bindingdb_release), args.workers, args.min_free_gb
    )
    checksums = verify_bindingdb(args.raw, args.bindingdb_release)
    gtopdb = summarize_gtopdb(args.raw)
    records = [
        json.loads(path.read_text())
        for path in sorted(args.raw.rglob("*.metadata.json"))
    ]
    pdbbind_records = [r for r in records if r["path"].startswith("pdbbind/")]
    summary = {
        "generated_at_utc": now(),
        "files": records,
        "failures": failures,
        "bindingdb_verified_md5": checksums,
        "gtopdb": gtopdb,
        "pdbbind": {
            "status": "complete"
            if len(pdbbind_records) == 2
            and all(r["status"] == "complete" for r in pdbbind_records)
            else "not_verified",
            "url": "https://www.pdbbind-plus.org.cn/download",
            "note": "Official downloads require login; previously acquired local archives can be imported with SHA256 verification.",
        },
    }
    dump(args.raw / "download_manifest.json", summary)
    print(
        json.dumps(
            {
                "complete": sum(r["status"] == "complete" for r in records),
                "failures": failures,
                "bindingdb_verified_md5": checksums,
            },
            indent=2,
        )
    )
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
