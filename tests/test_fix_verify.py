"""Tests for fix → re-execute → verify. The deterministic core needs no LLM."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import fix_verify as FV  # noqa: E402


# ---- run_and_check: execution is the judge ---------------------------------

def test_runs_and_matches_expected():
    code = "import numpy as np\nu = np.array([0.0, 0.5, 1.0])\n"
    r = FV.run_and_check(code, expected=[0.0, 0.5, 1.0])
    assert r["runs"] is True and r["matched"] is True


def test_runs_but_wrong_answer():
    code = "import numpy as np\nu = np.array([0.0, 0.9, 1.0])\n"
    r = FV.run_and_check(code, expected=[0.0, 0.5, 1.0])
    assert r["runs"] is True and r["matched"] is False


def test_does_not_run():
    r = FV.run_and_check("u = undefined_symbol + 1", expected=[1.0])
    assert r["runs"] is False and r["matched"] is False
    assert "NameError" in r["error"]


def test_no_expected_target_matched_is_none():
    r = FV.run_and_check("x = 1 + 1")
    assert r["runs"] is True and r["matched"] is None


def test_answer_found_by_namespace_scan_any_var_name():
    # the answer lives in `result`, not a conventional name → still found
    code = "import numpy as np\nresult = np.array([1.0, 2.0])\n"
    assert FV.run_and_check(code, expected=[1.0, 2.0])["matched"] is True


# ---- propose_fix + attempt_fix with a fake client --------------------------

class FakeClient:
    """Returns a fixed response, or a sequence of responses (one per call)."""
    def __init__(self, response):
        self.responses = response if isinstance(response, list) else None
        self.response = None if isinstance(response, list) else response
        self.calls = 0
        self.last_user = None

    def complete(self, system, user, max_tokens=1600):
        self.last_user = user
        self.calls += 1
        if self.responses is not None:
            return self.responses[min(self.calls - 1, len(self.responses) - 1)]
        return self.response


def test_propose_fix_strips_code_fences():
    c = FakeClient("```python\nu = 1\n```")
    assert FV.propose_fix(c, "buggy", "Q failed") == "u = 1"


def test_attempt_fix_verified_when_corrected_code_matches():
    good = "import numpy as np\nu = np.array([0.0, 0.5, 1.0])"
    fr = FV.attempt_fix(FakeClient(good), "buggy student code",
                        "Q1 failed at displacement", expected=[0.0, 0.5, 1.0])
    assert fr["fix_attempted"] and fr["fix_runs"] and fr["fix_verified"]


def test_attempt_fix_not_verified_when_still_wrong():
    wrong = "import numpy as np\nu = np.array([9.0, 9.0, 9.0])"
    fr = FV.attempt_fix(FakeClient(wrong), "code", "Q1 failed", expected=[0.0, 0.5, 1.0])
    assert fr["fix_runs"] and fr["fix_verified"] is False


def test_attempt_fix_reports_when_code_does_not_run():
    fr = FV.attempt_fix(FakeClient("u = nope"), "code", "Q1 failed", expected=[1.0])
    assert fr["fix_runs"] is False and fr["fix_verified"] is False


def test_attempt_fix_open_ended_verified_falls_back_to_runs():
    # no expected target → "runs through" counts as verified
    fr = FV.attempt_fix(FakeClient("x = 42"), "code", "P2 unclear", expected=None)
    assert fr["fix_runs"] and fr["fix_verified"] is True


def test_attempt_fix_handles_proposal_error():
    class Boom:
        def complete(self, *a, **k):
            raise RuntimeError("api down")
    fr = FV.attempt_fix(Boom(), "code", "Q1", expected=[1.0])
    assert fr["fix_attempted"] and not fr["fix_runs"] and not fr["fix_verified"]


# ---- iterative self-repair loop --------------------------------------------

def test_loop_retries_until_verified():
    # first attempt wrong, second attempt correct → verified on iteration 2
    wrong = "import numpy as np\nu = np.array([9.0, 9.0, 9.0])"
    good = "import numpy as np\nu = np.array([0.0, 0.5, 1.0])"
    c = FakeClient([wrong, good])
    fr = FV.attempt_fix(c, "code", "Q1 failed", expected=[0.0, 0.5, 1.0], max_iterations=3)
    assert fr["fix_verified"] is True and fr["iterations"] == 2 and not fr["exhausted"]
    assert len(fr["attempts"]) == 2
    # the retry prompt fed back the previous failure
    assert "PREVIOUS ATTEMPT" in c.last_user


def test_loop_exhausts_then_flags_for_review():
    wrong = "import numpy as np\nu = np.array([9.0, 9.0, 9.0])"
    c = FakeClient([wrong])  # always wrong
    fr = FV.attempt_fix(c, "code", "Q1 failed", expected=[0.0, 0.5, 1.0], max_iterations=3)
    assert fr["fix_verified"] is False and fr["exhausted"] is True
    assert fr["iterations"] == 3 and c.calls == 3   # used the full budget


def test_loop_stops_early_on_first_success():
    good = "import numpy as np\nu = np.array([0.0, 0.5, 1.0])"
    c = FakeClient([good])
    fr = FV.attempt_fix(c, "code", "Q1", expected=[0.0, 0.5, 1.0], max_iterations=5)
    assert fr["iterations"] == 1 and c.calls == 1 and fr["fix_verified"]
