"""Data layer for the JN Grader review dashboard.

Reads the on-disk grading outputs under ``workspace/<ASSIGN>/`` (scored JSON +
run_manifest) and normalizes BOTH engines' schemas into one shape the web UI can
render. Also persists human review decisions to ``workspace/<ASSIGN>/decisions/``
— the visual equivalent of the pipeline's human-approval node.

No Flask dependency here, so it is unit-testable on its own.

Engine schemas normalized:
  - general (score_general.py): ``problems: [{name, max, score, feedback}]``
  - numeric (score_notebooks.py): ``Q*_result/process/score`` + ``diagnostics``
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Engine-aware normalization
# ---------------------------------------------------------------------------

def _is_general(rec: dict) -> bool:
    return isinstance(rec.get("problems"), list)


def _numeric_questions(rec: dict) -> list[str]:
    qs = sorted({k.split("_")[0] for k in rec if k.endswith("_score") and k[0] == "Q"})
    return qs


def normalize_summary(rec: dict, sid: str) -> dict:
    """One row for the submission table (engine-agnostic)."""
    if _is_general(rec):
        engine = "general"
        max_score = int(rec.get("max_score") or sum(p.get("max", 0) for p in rec["problems"]))
    else:
        engine = "numeric"
        qs = _numeric_questions(rec)
        max_score = int(rec.get("max_score") or (len(qs) * 10) or 30)
    final = int(rec.get("final_score") or 0)
    return {
        "student_id": rec.get("student_id", sid),
        "source_file": rec.get("source_file", ""),
        "name": rec.get("name") or "",
        "student_no": rec.get("student_no") or "",
        "engine": engine,
        "final_score": final,
        "max_score": max_score,
        "pct": round(100.0 * final / max_score, 1) if max_score else 0.0,
        "status": rec.get("status", "AUTO"),
        "review_reasons": rec.get("review_reasons", []),
    }


def normalize_detail(rec: dict, sid: str) -> dict:
    """Full per-submission detail: items + feedback + diagnostics/invariants."""
    out = normalize_summary(rec, sid)
    feedback = rec.get("feedback", {}) or {}
    items: list[dict] = []
    if _is_general(rec):
        for p in rec["problems"]:
            items.append({"name": p.get("name"), "max": p.get("max"),
                          "score": p.get("score"), "feedback": p.get("feedback", "")})
    else:
        diags = rec.get("diagnostics", {}) or {}
        autodet = rec.get("autograde_detail", {}) or {}
        for q in _numeric_questions(rec):
            ag = autodet.get(q, {})
            items.append({
                "name": q, "max": 10,
                "score": rec.get(f"{q}_score"),
                "result": rec.get(f"{q}_result_score"),
                "process": rec.get(f"{q}_process_score"),
                "feedback": feedback.get(q, ""),
                "diagnostic": diags.get(q, {}),
                "physics": [p for p in (ag.get("physics") or []) if p.get("status") == "fail"],
                "fields": [p for p in (ag.get("fields") or []) if p.get("status") == "fail"],
                "first_divergence": ag.get("first_divergence"),
            })
    out["items"] = items
    out["overall_feedback"] = feedback.get("overall", "")
    out["confidence"] = rec.get("confidence")
    out["execution_status"] = rec.get("execution_status")
    out["scored_at"] = rec.get("scored_at")
    return out


# ---------------------------------------------------------------------------
# Assignment / submission listing
# ---------------------------------------------------------------------------

def _scored_dir(root: Path, assign: str) -> Path:
    return root / assign / "scored"


def list_assignments(root: Path) -> list[dict]:
    """Every assignment under workspace/ that has scored output, with a summary."""
    out = []
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        sdir = d / "scored"
        if not sdir.is_dir():
            continue
        recs = [r for r in (_read_json(f) for f in sdir.glob("*_scored.json")) if r]
        if not recs:
            continue
        summaries = [normalize_summary(r, "") for r in recs]
        n = len(summaries)
        avg_pct = round(sum(s["pct"] for s in summaries) / n, 1) if n else 0.0
        abstain = sum(1 for s in summaries if s["status"] == "ABSTAIN")
        decided = len(load_decisions(root, d.name))
        out.append({
            "assignment": d.name,
            "submissions": n,
            "avg_pct": avg_pct,
            "abstain": abstain,
            "reviewed": decided,
            "engine": summaries[0]["engine"] if summaries else "unknown",
            "has_manifest": (d / "run_manifest.json").exists(),
        })
    return out


def assignment_detail(root: Path, assign: str) -> dict:
    sdir = _scored_dir(root, assign)
    rows = []
    for f in sorted(sdir.glob("*_scored.json")):
        rec = _read_json(f)
        if rec:
            rows.append(normalize_summary(rec, f.stem.replace("_scored", "")))
    rows.sort(key=lambda r: r["final_score"])
    manifest = _read_json(root / assign / "run_manifest.json")
    decisions = load_decisions(root, assign)
    for r in rows:
        r["reviewed"] = r["student_id"] in decisions
    return {"assignment": assign, "submissions": rows,
            "manifest": manifest, "decisions": decisions}


def submission_detail(root: Path, assign: str, sid: str) -> dict | None:
    f = _scored_dir(root, assign) / f"{sid}_scored.json"
    rec = _read_json(f)
    if rec is None:
        return None
    detail = normalize_detail(rec, sid)
    decisions = load_decisions(root, assign)
    detail["decision"] = decisions.get(detail["student_id"])
    return detail


# ---------------------------------------------------------------------------
# Human review decisions (the approval node, persisted)
# ---------------------------------------------------------------------------

def _decisions_dir(root: Path, assign: str) -> Path:
    return root / assign / "decisions"


def load_decisions(root: Path, assign: str) -> dict[str, dict]:
    d = _decisions_dir(root, assign)
    out: dict[str, dict] = {}
    if not d.is_dir():
        return out
    for f in d.glob("*.json"):
        rec = _read_json(f)
        if rec and rec.get("student_id"):
            out[rec["student_id"]] = rec
    return out


def save_decision(
    root: Path, assign: str, sid: str,
    decision: str, final_score: int | None = None,
    note: str = "", reviewer: str = "",
    now: str | None = None,
) -> dict:
    """Persist a human review decision for one submission.

    ``decision`` is 'approve' (accept the AI score) or 'override' (use
    ``final_score``). Stored under workspace/<assign>/decisions/<sid>.json so the
    dashboard and any downstream export can see what a human decided."""
    if decision not in ("approve", "override"):
        raise ValueError(f"decision must be 'approve' or 'override', got {decision!r}")
    d = _decisions_dir(root, assign)
    d.mkdir(parents=True, exist_ok=True)
    rec = {
        "student_id": sid,
        "decision": decision,
        "final_score": final_score,
        "note": note,
        "reviewer": reviewer,
        "reviewed_at": now or datetime.now(timezone.utc).isoformat(),
    }
    (d / f"{sid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return rec
