"""score_general selective-grading gate: lenient JSON parse + abstain on failure."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import nbformat  # noqa: E402
import score_general as sg  # noqa: E402


# ---- lenient JSON extraction ----------------------------------------------

def test_extract_json_plain():
    assert sg._extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    assert sg._extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_wrapped_in_prose():
    raw = 'Sure, here is the grading:\n{"P2": {"score": 5}}\nHope that helps!'
    assert sg._extract_json(raw) == {"P2": {"score": 5}}


def test_extract_json_unsalvageable_raises():
    with pytest.raises(json.JSONDecodeError):
        sg._extract_json("not json at all")


# ---- fake client + minimal notebook ---------------------------------------

class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, system, user, max_tokens=1500):
        r = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        if isinstance(r, Exception):
            raise r
        return r


def _nb(tmp_path):
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell("# Problem 2"),
                nbformat.v4.new_code_cell("x = 1")]
    p = tmp_path / "sub.ipynb"
    nbformat.write(nb, str(p))
    return p


_CFG = {"assignment": "T", "max_score": 20,
        "problems": [{"name": "P2", "points": 20, "type": "llm", "desc": "d"}]}


def test_abstains_when_llm_call_raises(tmp_path):
    client = FakeClient(RuntimeError("boom"), RuntimeError("boom"))
    rec = sg.score_one(_nb(tmp_path), "anon-001", client, _CFG, "", "ref", "ans")
    assert rec["status"] == "ABSTAIN"
    assert "llm_failed" in rec["review_reasons"]


def test_abstains_when_json_unparseable_even_after_retry(tmp_path):
    # This is exactly the HW3 (3) case: malformed JSON twice → abstain, not 0.
    client = FakeClient("not json", "still not json")
    rec = sg.score_one(_nb(tmp_path), "anon-001", client, _CFG, "", "ref", "ans")
    assert rec["status"] == "ABSTAIN" and client.calls == 2


def test_abstains_when_problem_missing_from_result(tmp_path):
    client = FakeClient('{"overall": "ok but no P2 key"}')
    rec = sg.score_one(_nb(tmp_path), "anon-001", client, _CFG, "", "ref", "ans")
    assert rec["status"] == "ABSTAIN"
    assert any("missing_problem_scores" in r for r in rec["review_reasons"])


def test_auto_when_result_well_formed(tmp_path):
    client = FakeClient('{"P2": {"score": 17, "feedback": "good"}, "overall": "ok"}')
    rec = sg.score_one(_nb(tmp_path), "anon-001", client, _CFG, "", "ref", "ans")
    assert rec["status"] == "AUTO" and rec["review_reasons"] == []
    assert rec["final_score"] == 17


def test_retry_recovers_bad_first_response(tmp_path):
    # First response unparseable, retry returns valid JSON → AUTO (no abstain).
    client = FakeClient("garbage", '{"P2": {"score": 12, "feedback": "ok"}, "overall": "x"}')
    rec = sg.score_one(_nb(tmp_path), "anon-001", client, _CFG, "", "ref", "ans")
    assert rec["status"] == "AUTO" and rec["final_score"] == 12 and client.calls == 2
