"""Unit tests for the pipeline orchestrator (stage selection, manifest, gate)."""

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pipeline  # noqa: E402


def _opts(**kw):
    base = dict(from_stage=None, to_stage=None, rubric=None, memory=None, ingest=False,
                max_score=None, provider=None, model=None, base_url=None, api_key=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


# ---- stage selection -------------------------------------------------------

def test_select_stages_full_and_sliced():
    assert pipeline.select_stages("numeric", None, None) == ["preprocess", "score", "report"]
    assert pipeline.select_stages("general", None, None) == ["score", "report"]
    assert pipeline.select_stages("numeric", "score", "score") == ["score"]
    assert pipeline.select_stages("numeric", "score", None) == ["score", "report"]


def test_select_stages_rejects_inverted_range():
    with pytest.raises(ValueError):
        pipeline.select_stages("numeric", "report", "preprocess")


# ---- command construction --------------------------------------------------

def test_build_numeric_score_cmd_includes_review_queue_and_llm():
    p = pipeline.Paths("HW2")
    cmd = pipeline.build_stage_cmd("score", "numeric", p,
                                   _opts(provider="openai", model="m", memory=Path("mem.json")))
    s = " ".join(cmd)
    assert "score_notebooks.py" in s
    assert "--review-queue" in s and "review_queue" in s
    assert "--memory" in s and "mem.json" in s
    assert "--provider openai" in s and "--model m" in s


def test_build_general_score_cmd():
    p = pipeline.Paths("HW3")
    cmd = pipeline.build_stage_cmd("score", "general", p, _opts())
    s = " ".join(cmd)
    assert "score_general.py" in s and "--reference" in s and "--config" in s


def test_unknown_stage_raises():
    with pytest.raises(ValueError):
        pipeline.build_stage_cmd("bogus", "numeric", pipeline.Paths("HW2"), _opts())


# ---- review summary --------------------------------------------------------

def test_summarize_review_splits_auto_and_abstain(tmp_path):
    (tmp_path / "anon-001_scored.json").write_text(json.dumps(
        {"student_id": "anon-001", "status": "AUTO"}))
    (tmp_path / "anon-002_scored.json").write_text(json.dumps(
        {"student_id": "anon-002", "status": "ABSTAIN", "review_reasons": ["low_confidence"],
         "confidence": 0.4}))
    (tmp_path / "anon-003_scored.json").write_text(json.dumps(
        {"student_id": "anon-003"}))  # general engine: no status → AUTO
    s = pipeline.summarize_review(tmp_path)
    assert s["total"] == 3 and s["auto"] == 2 and s["abstain"] == 1
    assert s["review_queue"][0]["student_id"] == "anon-002"
    assert s["review_queue"][0]["reasons"] == ["low_confidence"]


# ---- full run with an injected runner -------------------------------------

def _fake_runner_ok(cmd):
    return 0, "ok", ""


def test_run_pipeline_all_ok_and_exit_codes(tmp_path, monkeypatch):
    # Point the scored dir at a tmp path with all-AUTO records.
    scored = tmp_path / "scored"
    scored.mkdir()
    (scored / "a_scored.json").write_text(json.dumps({"student_id": "a", "status": "AUTO"}))

    real_paths = pipeline.Paths
    monkeypatch.setattr(pipeline, "Paths",
                        lambda assign, repo=pipeline.REPO_ROOT: _patched_paths(real_paths, assign, scored))

    m = pipeline.run_pipeline("HW2", "numeric", _opts(), runner=_fake_runner_ok, clock=lambda: 0.0)
    assert [s["stage"] for s in m["stages"]] == ["preprocess", "score", "report"]
    assert all(s["status"] == "ok" for s in m["stages"])
    assert m["status"] == "done"
    assert pipeline.exit_code_for(m) == pipeline.EXIT_OK


def test_run_pipeline_needs_review(tmp_path, monkeypatch):
    scored = tmp_path / "scored"
    scored.mkdir()
    (scored / "a_scored.json").write_text(json.dumps(
        {"student_id": "a", "status": "ABSTAIN", "review_reasons": ["x"]}))
    real_paths = pipeline.Paths
    monkeypatch.setattr(pipeline, "Paths",
                        lambda assign, repo=pipeline.REPO_ROOT: _patched_paths(real_paths, assign, scored))
    m = pipeline.run_pipeline("HW2", "numeric", _opts(), runner=_fake_runner_ok, clock=lambda: 0.0)
    assert m["status"] == "needs_review"
    assert pipeline.exit_code_for(m) == pipeline.EXIT_NEEDS_REVIEW


def test_run_pipeline_stops_on_stage_failure(monkeypatch):
    def failing(cmd):
        return (1, "", "boom") if "score_notebooks.py" in " ".join(cmd) else (0, "", "")
    m = pipeline.run_pipeline("HW2", "numeric", _opts(), runner=failing, clock=lambda: 0.0)
    assert m["status"] == "failed" and m["failed_stage"] == "score"
    # report stage must NOT run after score failed
    assert [s["stage"] for s in m["stages"]] == ["preprocess", "score"]
    assert pipeline.exit_code_for(m) == pipeline.EXIT_STAGE_FAILED


def test_ingest_stage_appended(monkeypatch):
    seen = []
    def rec(cmd):
        seen.append(cmd)
        return 0, "", ""
    # avoid scored-dir read affecting status
    monkeypatch.setattr(pipeline, "summarize_review", lambda d: {"total": 0, "auto": 0, "abstain": 0, "review_queue": []})
    m = pipeline.run_pipeline("HW3", "general", _opts(ingest=True), runner=rec, clock=lambda: 0.0)
    assert [s["stage"] for s in m["stages"]] == ["score", "report", "ingest"]
    assert any("db_ingest.py" in " ".join(c) for c in seen)


def _patched_paths(real_paths_cls, assign, scored_dir):
    p = real_paths_cls(assign)
    p.scored = scored_dir
    return p
