"""File-ingest helpers for the dashboard's upload flow (Flask-free, testable).

Takes uploaded bytes and lays them out in the conventional per-assignment input
layout the graders expect:

    datasets/<ASSIGN>/
    ├── reference.ipynb     (kind="reference")
    ├── description.txt     (kind="description")
    ├── config.yaml         (kind="config")
    └── submissions/*.ipynb (uploaded individually or inside a .zip)

All names are sanitized and confined to the assignment dir (no path traversal).
Functions take raw bytes so they can be unit-tested without an HTTP layer.
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from pathlib import Path

# Single-file inputs: kind → destination filename.
META_FILES = {
    "reference": "reference.ipynb",
    "description": "description.txt",
    "config": "config.yaml",
}


def safe_assign(name: str) -> str:
    """A safe assignment key: keep word chars / dash, drop the rest."""
    s = re.sub(r"[^A-Za-z0-9_-]", "", (name or "").strip())
    if not s:
        raise ValueError("invalid assignment name")
    return s


def secure_name(name: str) -> str:
    """Basename only, with unsafe characters replaced — no traversal."""
    base = os.path.basename((name or "").replace("\\", "/"))
    cleaned = re.sub(r"[^A-Za-z0-9._ ()-]", "_", base).strip()
    return cleaned or "file"


def dataset_dir(datasets_root: Path, assign: str) -> Path:
    return datasets_root / safe_assign(assign)


def save_meta(datasets_root: Path, assign: str, kind: str, blob: bytes) -> str:
    """Save a single-file input (reference / description / config). Returns name."""
    fname = META_FILES.get(kind)
    if not fname:
        raise ValueError(f"unknown file kind {kind!r}")
    d = dataset_dir(datasets_root, assign)
    d.mkdir(parents=True, exist_ok=True)
    (d / fname).write_bytes(blob)
    return fname


def _extract_zip_ipynb(blob: bytes, subdir: Path) -> list[str]:
    saved: list[str] = []
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for member in z.namelist():
            if member.endswith("/") or member.startswith("__MACOSX"):
                continue
            if not member.lower().endswith(".ipynb"):
                continue
            data = z.read(member)
            name = secure_name(Path(member).name)
            (subdir / name).write_bytes(data)
            saved.append(name)
    return saved


def add_submissions(datasets_root: Path, assign: str, items: list[tuple[str, bytes]]) -> list[str]:
    """Add uploaded submissions. Each item is (filename, bytes); a .zip is
    expanded to its .ipynb members, a .ipynb is saved as-is. Returns saved names."""
    subdir = dataset_dir(datasets_root, assign) / "submissions"
    subdir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for filename, blob in items:
        ext = Path(filename).suffix.lower()
        if ext == ".zip":
            try:
                saved += _extract_zip_ipynb(blob, subdir)
            except zipfile.BadZipFile:
                continue
        elif ext == ".ipynb":
            name = secure_name(filename)
            (subdir / name).write_bytes(blob)
            saved.append(name)
    return saved


def dataset_status(datasets_root: Path, assign: str) -> dict:
    """What inputs exist for an assignment (so the UI knows what's gradeable)."""
    d = dataset_dir(datasets_root, assign)
    subs = d / "submissions"
    return {
        "assignment": safe_assign(assign),
        "submissions": len(list(subs.glob("*.ipynb"))) if subs.is_dir() else 0,
        "has_reference": (d / "reference.ipynb").exists(),
        "has_config": (d / "config.yaml").exists(),
        "has_description": (d / "description.txt").exists(),
    }


def list_datasets(datasets_root: Path) -> list[dict]:
    if not datasets_root.is_dir():
        return []
    out = []
    for d in sorted(datasets_root.iterdir()):
        if d.is_dir() and (d / "submissions").is_dir():
            out.append(dataset_status(datasets_root, d.name))
    return out
