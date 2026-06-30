"""Physics-signal integration in score_notebooks: findings block + gate trigger."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import score_notebooks as sn  # noqa: E402


def _ag(passed, physics=None, first_div=None):
    return {"passed": passed, "first_divergence": first_div,
            "checkpoints": [], "physics": physics or []}


def test_findings_block_flags_physics_contradiction_on_pass():
    ag = {"Q1": _ag(True, physics=[
        {"name": "sym", "type": "symmetric", "status": "fail", "detail": "‖M−Mᵀ‖=3"}])}
    block = sn._findings_block(ag, ["Q1"])
    assert "PASS" in block
    assert "PHYSICS CONTRADICTION" in block and "sym" in block


def test_findings_block_corroborates_on_fail():
    ag = {"Q1": _ag(False, first_div="global_stiffness", physics=[
        {"name": "psd", "type": "psd", "status": "fail", "detail": "min eig=-2"}])}
    block = sn._findings_block(ag, ["Q1"])
    assert "corroborates" in block and "psd" in block


def test_findings_block_omits_na_and_pass_checks():
    ag = {"Q1": _ag(True, physics=[
        {"name": "fin", "type": "finite", "status": "pass", "detail": "ok"},
        {"name": "missing", "type": "finite", "status": "na", "detail": "absent"}])}
    block = sn._findings_block(ag, ["Q1"])
    assert "CONTRADICTION" not in block  # only failures are surfaced


def test_gate_abstains_on_passed_but_physics_violated():
    autograde = {"Q1": _ag(True, physics=[
        {"name": "sym", "type": "symmetric", "status": "fail", "detail": "x"}])}
    scored = {"confidence": 0.9, "diagnostics": {"Q1": {"confidence": 0.9, "located": True}}}
    out = sn.gate(scored, autograde, ["Q1"], exec_status="success", llm_failed=False)
    assert out["status"] == "ABSTAIN"
    assert any("passed_but_physics_violated" in r for r in out["review_reasons"])


def test_gate_auto_when_physics_clean():
    autograde = {"Q1": _ag(True, physics=[
        {"name": "sym", "type": "symmetric", "status": "pass", "detail": "ok"}])}
    scored = {"confidence": 0.9, "diagnostics": {"Q1": {"confidence": 0.9, "located": True}}}
    out = sn.gate(scored, autograde, ["Q1"], exec_status="success", llm_failed=False)
    assert out["status"] == "AUTO"
    assert out["review_reasons"] == []
