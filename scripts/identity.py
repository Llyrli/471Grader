"""Best-effort extraction of a student's name and ID from a submission.

Filename first (most reliable), then the notebook's header lines. Heuristic —
review the output; it maps the anonymous ``anon-NNN`` ids back to real students,
it is not authoritative.
"""

from __future__ import annotations

import re

# Student IDs here are 8–11 digit numbers (e.g. 22571086, 3220111241). The
# lookarounds reject digit runs embedded in decimals/longer numbers, so a
# coordinate like -0.41166073 is NOT mistaken for an ID.
_ID_RE = re.compile(r"(?<![\d.])\d{8,11}(?![\d.])")
# Course/assignment noise to drop when guessing a name.
_NOISE = {
    "hw", "hwk", "homework", "assignment", "me", "problem", "prob",
    "sol", "solution", "final", "submission", "ipynb", "np", "array",
    "image", "png", "jpg", "phantom", "begin", "end",
}


def _looks_like_name_token(tok: str) -> bool:
    if len(tok) < 2:                                # drops single letters (e, g)
        return False
    if any(ch.isdigit() for ch in tok):            # drops HW3, 471HW3, ids
        return False
    if tok.lower() in _NOISE:
        return False
    return all(ch.isalpha() for ch in tok)         # letters only (no code symbols)


def _name_tokens(s: str) -> list[str]:
    s = re.sub(r"\([^)]*\)", " ", s)               # drop "(1)"
    s = re.sub(r"[_\-.]+", " ", s)                 # separators → space
    return [t for t in s.split() if _looks_like_name_token(t)]


def _name_from_filename(filename: str) -> str:
    base = re.sub(r"\.[A-Za-z0-9]+$", "", filename)
    return " ".join(_name_tokens(base)).strip()


def _id_from(text: str) -> str | None:
    m = _ID_RE.search(text)
    return m.group(0) if m else None


def _name_from_text(text: str) -> str:
    """Only trust a header line that ALSO carries a student id (Name + ID)."""
    for raw in text.splitlines()[:40]:
        line = raw.strip().lstrip("#").strip()
        if not line or len(line) > 60 or not _ID_RE.search(line):
            continue
        tokens = _name_tokens(line)
        if 1 <= len(tokens) <= 4:
            return " ".join(tokens)
    return ""


def extract_identity(filename: str, text: str = "") -> dict[str, str | None]:
    """Return {'name': str, 'student_no': str|None}."""
    name = _name_from_filename(filename)
    student_no = _id_from(filename) or (_id_from(text[:4000]) if text else None)
    if not name and text:
        name = _name_from_text(text[:4000])
    return {"name": name, "student_no": student_no}
