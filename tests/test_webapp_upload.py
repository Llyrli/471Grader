"""Tests for the upload/grade workflow: files.py, runner.py, and the routes."""

import importlib
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

WEBAPP = Path(__file__).resolve().parents[1] / "webapp"
sys.path.insert(0, str(WEBAPP))

import files as F  # noqa: E402
import runner as R  # noqa: E402


# ---- files.py --------------------------------------------------------------

def test_safe_assign_and_secure_name():
    assert F.safe_assign(" HW-3_x ") == "HW-3_x"
    with pytest.raises(ValueError):
        F.safe_assign("///")
    assert F.secure_name("../../etc/passwd") == "passwd"
    assert F.secure_name("a b(1).ipynb") == "a b(1).ipynb"


def test_save_meta_and_submissions(tmp_path):
    F.save_meta(tmp_path, "HW9", "reference", b"{}")
    assert (tmp_path / "HW9" / "reference.ipynb").read_bytes() == b"{}"
    with pytest.raises(ValueError):
        F.save_meta(tmp_path, "HW9", "bogus", b"x")
    saved = F.add_submissions(tmp_path, "HW9", [("s1.ipynb", b"{}"), ("note.txt", b"x")])
    assert saved == ["s1.ipynb"]  # non-ipynb ignored
    assert (tmp_path / "HW9" / "submissions" / "s1.ipynb").exists()


def test_add_submissions_extracts_zip(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.ipynb", b"{}")
        z.writestr("sub/b.ipynb", b"{}")
        z.writestr("__MACOSX/c.ipynb", b"junk")
        z.writestr("readme.txt", b"x")
    saved = F.add_submissions(tmp_path, "HW9", [("pack.zip", buf.getvalue())])
    assert sorted(saved) == ["a.ipynb", "b.ipynb"]  # macosx + txt skipped


def test_dataset_status_and_list(tmp_path):
    F.add_submissions(tmp_path, "HW9", [("s1.ipynb", b"{}")])
    F.save_meta(tmp_path, "HW9", "config", b"x")
    st = F.dataset_status(tmp_path, "HW9")
    assert st["submissions"] == 1 and st["has_config"] and not st["has_reference"]
    assert F.list_datasets(tmp_path)[0]["assignment"] == "HW9"


# ---- runner.py -------------------------------------------------------------

def test_build_cmd_and_status_mapping():
    cmd = R.build_cmd("HW9", "general", llm={"provider": "openai", "model": "m",
                                             "base_url": "http://x"}, memory="m.json")
    s = " ".join(cmd)
    assert "pipeline.py" in s and "--assign HW9" in s and "--engine general" in s
    assert "--provider openai" in s and "--memory m.json" in s
    assert R._status_from_rc(None) == "running"
    assert R._status_from_rc(0) == "done"
    assert R._status_from_rc(10) == "done_review"
    assert R._status_from_rc(1) == "failed"


class FakeProc:
    def __init__(self, rc=None):
        self.rc = rc

    def poll(self):
        return self.rc


def _fake_popen(rc):
    def f(cmd, stdout=None, stderr=None, cwd=None):
        if stdout:
            stdout.write("pipeline log line\n")
            stdout.flush()
        return FakeProc(rc)
    return f


def test_runner_start_status_and_guard(tmp_path):
    run = R.GradeRunner(popen=_fake_popen(None))   # stays "running"
    log = tmp_path / "ws" / "HW9" / "grade.log"
    man = tmp_path / "ws" / "HW9" / "run_manifest.json"
    st = run.start("HW9", ["echo"], log, man)
    assert st["status"] == "running" and run.is_running("HW9")
    # second start while running → guarded
    with pytest.raises(RuntimeError):
        run.start("HW9", ["echo"], log, man)


def test_runner_done_reads_manifest(tmp_path):
    log = tmp_path / "HW9" / "grade.log"
    man = tmp_path / "HW9" / "run_manifest.json"
    man.parent.mkdir(parents=True)
    man.write_text(json.dumps({"status": "needs_review", "stages": [{"stage": "score", "status": "ok"}],
                               "review": {"total": 3, "auto": 2, "abstain": 1}}))
    run = R.GradeRunner(popen=_fake_popen(10))      # exit 10 → done_review
    run.start("HW9", ["echo"], log, man)
    st = run.status("HW9")
    assert st["status"] == "done_review"
    assert st["review"]["abstain"] == 1 and st["manifest_status"] == "needs_review"
    assert "pipeline log line" in "\n".join(st["log_tail"])


# ---- Flask routes ----------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("JN_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("JN_DATASETS", str(tmp_path / "datasets"))
    sys.path.insert(0, str(WEBAPP))
    import app as appmod
    importlib.reload(appmod)
    appmod.RUNNER = R.GradeRunner(popen=_fake_popen(None))  # never spawns real procs
    return appmod, appmod.app.test_client()


def test_route_upload_and_datasets(client):
    appmod, c = client
    data = {
        "reference": (io.BytesIO(b"{}"), "reference.ipynb"),
        "submissions": (io.BytesIO(b"{}"), "stud1.ipynb"),
    }
    r = c.post("/api/assignments/HW9/upload", data=data, content_type="multipart/form-data")
    body = r.get_json()
    assert body["assignment"] == "HW9"
    assert body["saved"]["submissions"] == ["stud1.ipynb"]
    assert body["status"]["has_reference"] is True
    # appears in datasets listing
    ds = c.get("/api/datasets").get_json()
    assert ds and ds[0]["assignment"] == "HW9" and ds[0]["submissions"] == 1


def test_route_grade_and_run_status(client):
    appmod, c = client
    c.post("/api/assignments/HW9/upload", data={"submissions": (io.BytesIO(b"{}"), "s.ipynb")},
           content_type="multipart/form-data")
    r = c.post("/api/assignments/HW9/grade", json={"engine": "general"})
    assert r.status_code == 200 and r.get_json()["status"] == "running"
    # second grade while running → 409
    assert c.post("/api/assignments/HW9/grade", json={"engine": "general"}).status_code == 409
    assert c.get("/api/assignments/HW9/run").get_json()["status"] == "running"


def test_route_grade_rejects_bad_engine(client):
    appmod, c = client
    assert c.post("/api/assignments/HW9/grade", json={"engine": "bogus"}).status_code == 400
