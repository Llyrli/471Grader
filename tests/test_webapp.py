"""Tests for the review dashboard: data layer (no Flask) + Flask routes."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webapp"))

import data as D  # noqa: E402


def _write(d: Path, name: str, obj: dict):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(obj), encoding="utf-8")


def _general_rec(sid, final, max_score=65, status="AUTO"):
    return {"student_id": sid, "source_file": f"{sid}.ipynb", "status": status,
            "max_score": max_score, "final_score": final, "review_reasons": [],
            "problems": [{"name": "P2", "max": 20, "score": min(final, 20), "feedback": "ok"},
                         {"name": "P3", "max": 45, "score": max(0, final - 20), "feedback": "meh"}],
            "feedback": {"overall": "fine"}}


def _numeric_rec(sid, status="AUTO"):
    return {"student_id": sid, "status": status, "execution_status": "success",
            "Q1_result_score": 3, "Q1_process_score": 7, "Q1_score": 10,
            "Q2_result_score": 0, "Q2_process_score": 4, "Q2_score": 4,
            "final_score": 14, "confidence": 0.8, "review_reasons": [],
            "feedback": {"Q1": "good", "Q2": "wrong assembly", "overall": "ok"},
            "diagnostics": {"Q2": {"error_class": "physics_modeling",
                                   "first_divergence": "global_stiffness", "fix": "fix k_e"}},
            "autograde_detail": {"Q2": {"first_divergence": "global_stiffness",
                                        "physics": [{"name": "sym", "type": "symmetric",
                                                     "status": "fail", "detail": "asym"}],
                                        "fields": []}}}


def test_normalize_general_and_numeric():
    g = D.normalize_summary(_general_rec("anon-001", 52), "anon-001")
    assert g["engine"] == "general" and g["max_score"] == 65 and g["pct"] == 80.0
    n = D.normalize_summary(_numeric_rec("anon-002"), "anon-002")
    assert n["engine"] == "numeric" and n["max_score"] == 20 and n["final_score"] == 14


def test_numeric_detail_surfaces_diagnostics_and_invariants():
    d = D.normalize_detail(_numeric_rec("anon-002"), "anon-002")
    q2 = next(i for i in d["items"] if i["name"] == "Q2")
    assert q2["diagnostic"]["error_class"] == "physics_modeling"
    assert q2["first_divergence"] == "global_stiffness"
    assert q2["physics"][0]["name"] == "sym"  # only failing invariants kept


def test_list_assignments_and_detail(tmp_path):
    _write(tmp_path / "HW3" / "scored", "anon-001_scored.json", _general_rec("anon-001", 60))
    _write(tmp_path / "HW3" / "scored", "anon-002_scored.json", _general_rec("anon-002", 10, status="ABSTAIN"))
    lst = D.list_assignments(tmp_path)
    assert len(lst) == 1
    a = lst[0]
    assert a["assignment"] == "HW3" and a["submissions"] == 2 and a["abstain"] == 1

    det = D.assignment_detail(tmp_path, "HW3")
    # sorted ascending by score → the abstained low one first
    assert det["submissions"][0]["student_id"] == "anon-002"


def test_save_and_load_decision_roundtrip(tmp_path):
    _write(tmp_path / "HW3" / "scored", "anon-002_scored.json", _general_rec("anon-002", 10, status="ABSTAIN"))
    rec = D.save_decision(tmp_path, "HW3", "anon-002", "override", final_score=35,
                          note="rubric exception", reviewer="zx", now="2026-06-30T00:00:00Z")
    assert rec["decision"] == "override" and rec["final_score"] == 35
    loaded = D.load_decisions(tmp_path, "HW3")
    assert loaded["anon-002"]["note"] == "rubric exception"
    # reflected in assignment detail + submission detail
    det = D.assignment_detail(tmp_path, "HW3")
    assert det["submissions"][0]["reviewed"] is True
    sub = D.submission_detail(tmp_path, "HW3", "anon-002")
    assert sub["decision"]["final_score"] == 35


def test_save_decision_rejects_bad_value(tmp_path):
    with pytest.raises(ValueError):
        D.save_decision(tmp_path, "HW3", "anon-001", "delete")


def test_flask_routes_smoke(tmp_path, monkeypatch):
    flask = pytest.importorskip("flask")
    _write(tmp_path / "HW3" / "scored", "anon-001_scored.json", _general_rec("anon-001", 60))
    monkeypatch.setenv("JN_WORKSPACE", str(tmp_path))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webapp"))
    import importlib
    import app as appmod
    importlib.reload(appmod)
    client = appmod.app.test_client()

    assert client.get("/api/assignments").get_json()[0]["assignment"] == "HW3"
    sub = client.get("/api/assignments/HW3/submissions/anon-001").get_json()
    assert sub["final_score"] == 60
    assert client.get("/api/assignments/HW3/submissions/zzz").status_code == 404
    r = client.post("/api/assignments/HW3/submissions/anon-001/decision",
                    json={"decision": "approve", "reviewer": "zx"})
    assert r.get_json()["decision"] == "approve"
    assert client.post("/api/assignments/HW3/submissions/anon-001/decision",
                       json={"decision": "bogus"}).status_code == 400
