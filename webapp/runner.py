"""Background grading-run manager for the dashboard's "Start grading" button.

Runs `scripts/pipeline.py` as a detached subprocess (LLM grading is slow), tracks
one run per assignment in memory, and reports status by polling the process +
reading the run manifest it writes. Flask-free; the process factory is injectable
so the state machine is unit-testable without spawning anything.

pipeline.py exit codes map to run status:
    0  → done (all AUTO)            10 → done_review (some abstained → human queue)
    1  → failed                     other → failed
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = REPO_ROOT / "scripts" / "pipeline.py"


def build_cmd(assign: str, engine: str, llm: dict | None = None, memory: str | None = None) -> list[str]:
    """Argv for a pipeline run. Pure — unit-testable."""
    cmd = [sys.executable, str(PIPELINE), "--assign", assign, "--engine", engine]
    llm = llm or {}
    if llm.get("provider"):
        cmd += ["--provider", llm["provider"]]
    if llm.get("model"):
        cmd += ["--model", llm["model"]]
    if llm.get("base_url"):
        cmd += ["--base-url", llm["base_url"]]
    if memory:
        cmd += ["--memory", memory]
    return cmd


def _status_from_rc(rc: int | None) -> str:
    if rc is None:
        return "running"
    if rc == 0:
        return "done"
    if rc == 10:
        return "done_review"
    return "failed"


class GradeRunner:
    """Tracks one background grading run per assignment."""

    def __init__(self, popen=subprocess.Popen, repo_root: Path = REPO_ROOT):
        self._popen = popen
        self._repo_root = repo_root
        self._runs: dict[str, dict] = {}

    def is_running(self, assign: str) -> bool:
        r = self._runs.get(assign)
        return bool(r and r["proc"].poll() is None)

    def start(self, assign: str, cmd: list[str], log_path: Path, manifest_path: Path) -> dict:
        if self.is_running(assign):
            raise RuntimeError("a grading run is already in progress for this assignment")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate a stale manifest so status() doesn't report an old run as this one.
        logf = open(log_path, "w", encoding="utf-8")
        proc = self._popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(self._repo_root))
        self._runs[assign] = {"proc": proc, "log": log_path, "manifest": manifest_path,
                              "cmd": cmd}
        return self.status(assign)

    def status(self, assign: str) -> dict:
        r = self._runs.get(assign)
        if not r:
            return {"assignment": assign, "status": "idle"}
        rc = r["proc"].poll()
        status = _status_from_rc(rc)
        out: dict = {"assignment": assign, "status": status, "returncode": rc}
        # Tail the log for live feedback.
        try:
            if r["log"].exists():
                lines = r["log"].read_text(encoding="utf-8", errors="replace").splitlines()
                out["log_tail"] = [ln for ln in lines if "httpx" not in ln][-12:]
        except Exception:
            pass
        # When finished, attach the manifest's review summary if present.
        if status != "running":
            try:
                import json
                if r["manifest"].exists():
                    m = json.loads(r["manifest"].read_text(encoding="utf-8"))
                    out["manifest_status"] = m.get("status")
                    out["review"] = m.get("review")
                    out["stages"] = [{"stage": s["stage"], "status": s["status"]}
                                     for s in m.get("stages", [])]
            except Exception:
                pass
        return out
