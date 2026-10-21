"""Read-only source utilities and explicit nullable staging contracts."""

import csv
import hashlib
import io
import json
import math
import re
import zipfile
from contextlib import contextmanager

from scripts.ripple.source_alignment import TableWriter, schema

ENDPOINTS = {"Kd": 0, "Ki": 1, "IC50": 2, "EC50": 3}


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_paths(raw, out):
    raw, out = raw.resolve(strict=True), out.resolve()
    if raw == out or raw in out.parents or out in raw.parents:
        raise ValueError("raw and output must be separate, non-overlapping directories")
    if out.exists() and any(p.is_symlink() for p in out.rglob("*")):
        raise ValueError("Output symlinks are not allowed")
    return raw, out


def raw_snapshot(raw):
    entries = {}
    for path in sorted([raw, *raw.rglob("*")]):
        if path.is_symlink():
            raise ValueError(f"Raw symlinks require explicit audit: {path}")
        st = path.stat()
        entry = {"mode": st.st_mode, "mtime_ns": st.st_mtime_ns, "size": st.st_size}
        if path.is_file():
            entry["sha256"] = file_hash(path)
        entries[str(path.relative_to(raw))] = entry
    return entries


def ids(value):
    return sorted(
        {
            x
            for x in re.split(r"[,;|\s]+", value.strip())
            if x and x not in {"None", "NA"}
        }
    )


def concentration(raw, unit="nM"):
    match = re.fullmatch(
        r"\s*(<=|>=|<|>|=|~)?\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*", raw
    )
    factors = {"M": 0, "mM": -3, "uM": -6, "nM": -9, "pM": -12, "fM": -15}
    if not match or unit not in factors:
        return {"measurement_status": "unparsed"}
    value = float(match[2])
    if not math.isfinite(value) or value <= 0:
        return {"measurement_status": "nonpositive_or_nonfinite"}
    relation = match[1] or "="
    log_m = math.log10(value) + factors[unit]
    return {
        "relation": relation,
        "value_original": value,
        "unit_original": unit,
        "affinity_value": -log_m if relation == "=" else None,
        "measurement_status": "exact" if relation == "=" else "censored_or_approximate",
    }


@contextmanager
def zipped_rows(path, member):
    with (
        zipfile.ZipFile(path) as archive,
        archive.open(member) as binary,
        io.TextIOWrapper(binary) as text,
    ):
        yield csv.DictReader(text, delimiter="\t")


def gtop_rows(path):
    with path.open(encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        first = next(reader)
        header = next(reader) if first[0].startswith("#") else first
        for number, values in enumerate(reader, 1):
            if not values:
                continue
            if len(values) != len(header):
                raise ValueError(f"{path}:{number}: inconsistent field count")
            yield number, dict(zip(header, values))


def table(path, fields, rows):
    count = 0
    with TableWriter(path, schema(**fields)) as writer:
        for row in rows:
            writer.add(row)
            count += 1
    return count


def effect_direction(kind, action):
    positive = {
        "Activation",
        "Agonist",
        "Full agonist",
        "Positive",
        "Potentiation",
        "Irreversible agonist",
    }
    negative = {
        "Inhibition",
        "Antagonist",
        "Inverse agonist",
        "Negative",
        "Pore blocker",
        "Irreversible inhibition",
        "Feedback inhibition",
        "Voltage-dependent inhibition",
    }
    type_direction = {
        "Activator": 0,
        "Agonist": 0,
        "Antagonist": 1,
        "Inhibitor": 1,
        "Channel blocker": 1,
        "Gating inhibitor": 1,
    }
    if action in {
        "Mixed",
        "Biphasic",
        "Partial agonist",
        "Biased agonist",
        "Slows inactivation",
    }:
        return None, "context_dependent"
    direction = 0 if action in positive else 1 if action in negative else None
    expected = type_direction.get(kind)
    if direction is not None and expected is not None and direction != expected:
        return None, "type_action_conflict"
    if direction is not None:
        return direction, "explicit_action"
    # Binding alone and an absent action are not functional direction evidence.
    return None, "direction_not_explicit"
