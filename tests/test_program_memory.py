"""Unit tests for program_memory: deterministic sedimentation + block formatting."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import program_memory as pm  # noqa: E402


def _write(d: Path, name: str, obj: dict) -> None:
    (d / name).write_text(json.dumps(obj), encoding="utf-8")


def test_fold_numeric_groups_by_class_and_locus():
    store = pm.empty_store("ME471")
    scored = {
        "status": "AUTO",
        "diagnostics": {
            "Q1": {"error_class": "none", "first_divergence": None},
            "Q3": {"error_class": "physics_modeling", "first_divergence": "global_stiffness",
                   "explanation": "wrong constitutive law", "fix": "use E*A/L"},
        },
    }
    assert pm._fold_numeric(scored, "HW2", store) == 1
    sig = "physics_modeling::global_stiffness"
    assert sig in store["error_patterns"]
    p = store["error_patterns"][sig]
    assert p["count"] == 1 and p["assignments"] == ["HW2"]
    assert p["fix_hint"] == "use E*A/L"
    # 'none' must not become a pattern.
    assert all("none" not in s for s in store["error_patterns"])


def test_fold_numeric_counts_repeat_and_skips_abstain():
    store = pm.empty_store("ME471")
    rec = {"status": "AUTO", "diagnostics": {
        "Q2": {"error_class": "coding", "first_divergence": "displacement", "explanation": "x"}}}
    pm._fold_numeric(rec, "HW2", store)
    pm._fold_numeric(rec, "HW4", store)
    p = store["error_patterns"]["coding::displacement"]
    assert p["count"] == 2
    assert set(p["assignments"]) == {"HW2", "HW4"}
    # ABSTAIN records are not trusted enough to shape memory.
    assert pm._fold_numeric({"status": "ABSTAIN", "diagnostics": {
        "Q1": {"error_class": "coding", "first_divergence": "displacement"}}}, "HW5", store) == 0
    assert store["error_patterns"]["coding::displacement"]["count"] == 2


def test_fold_general_collects_only_deductions():
    store = pm.empty_store("ME471")
    scored = {"problems": [
        {"name": "P1", "max": 10, "score": 10, "feedback": "perfect"},      # full marks → ignored
        {"name": "P3", "max": 35, "score": 28, "feedback": "missing numeric matrix in (f),(g)"},
    ]}
    assert pm._fold_general(scored, "HW3", store) == 1
    assert "HW3::P3" in store["problem_notes"]
    assert "HW3::P1" not in store["problem_notes"]


def test_format_block_ranks_by_frequency_and_is_advisory():
    store = pm.empty_store("ME471")
    store["error_patterns"] = {
        "coding::displacement": {"error_class": "coding", "locus": "displacement",
                                 "count": 2, "examples": ["e1"], "fix_hint": ""},
        "physics_modeling::global_stiffness": {"error_class": "physics_modeling",
                                                "locus": "global_stiffness", "count": 9,
                                                "examples": ["bad assembly"], "fix_hint": "fix K"},
    }
    pm.add_convention(store, "Deduct for code with no markdown explanation.", "HW3")
    block = pm.format_memory_block(store)
    assert "PROGRAM MEMORY" in block and "NOT override" in block
    # Most frequent pattern appears before the rarer one.
    assert block.index("global_stiffness") < block.index("displacement")
    assert "Deduct for code with no markdown" in block


def test_empty_store_yields_empty_block():
    assert pm.format_memory_block(pm.empty_store("ME471")) == ""
    assert pm.load_block(None) == ""


def test_roundtrip_sediment_dir(tmp_path):
    scored_dir = tmp_path / "HW2" / "scored"
    scored_dir.mkdir(parents=True)
    _write(scored_dir, "anon-001_scored.json", {"status": "AUTO", "diagnostics": {
        "Q1": {"error_class": "physics_modeling", "first_divergence": "bc_reduced",
               "explanation": "BC wrong", "fix": "fix node 0"}}})
    _write(scored_dir, "anon-002_scored.json", {"problems": [
        {"name": "P2", "max": 20, "score": 5, "feedback": "incomplete"}]})

    store = pm.empty_store("ME471")
    stats = pm.sediment_dir(scored_dir, "HW2", store)
    assert stats["files"] == 2
    assert stats["numeric_patterns"] == 1
    assert stats["general_deductions"] == 1

    store_path = tmp_path / "ME471.json"
    pm.save_store(store_path, store)
    reloaded = pm.load_store(store_path)
    assert reloaded["error_patterns"]["physics_modeling::bc_reduced"]["count"] == 1
    assert reloaded["updated_at"] is not None
    assert pm.load_block(store_path)  # non-empty block from a populated store


def test_add_convention_dedups():
    store = pm.empty_store("ME471")
    assert pm.add_convention(store, "rule A", "HW2") is True
    assert pm.add_convention(store, "rule A", "HW3") is False
    assert len(store["conventions"]) == 1
