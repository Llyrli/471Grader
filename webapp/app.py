"""JN Grader review dashboard — a small Flask app over the grading outputs.

A grader can browse assignments and submissions, see scores / feedback /
deterministic diagnostics, and — most importantly — work the **abstention queue**:
the submissions the confidence gate flagged for human review. Approving or
overriding a score here persists a decision to workspace/<assign>/decisions/,
making this the visual form of the pipeline's human-approval node.

Run:
    pip install flask
    python webapp/app.py            # serves http://127.0.0.1:5000
    # custom workspace / port:
    JN_WORKSPACE=workspace JN_PORT=5000 python webapp/app.py

All data is read live from the workspace dir — no database required (the task
bank stays optional). The data layer (data.py) is Flask-free and unit-tested.
"""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

import data as D
import files as F
from runner import GradeRunner, build_cmd

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ.get("JN_WORKSPACE", REPO_ROOT / "workspace"))
DATASETS = Path(os.environ.get("JN_DATASETS", REPO_ROOT / "datasets"))
STATIC = Path(__file__).resolve().parent / "static"

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB upload cap
RUNNER = GradeRunner()


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/static/<path:fname>")
def static_files(fname: str):
    return send_from_directory(STATIC, fname)


@app.get("/api/assignments")
def api_assignments():
    return jsonify(D.list_assignments(WORKSPACE))


@app.get("/api/datasets")
def api_datasets():
    """Uploaded assignments (inputs), with how many are graded so far."""
    graded = {a["assignment"]: a for a in D.list_assignments(WORKSPACE)}
    out = []
    for ds in F.list_datasets(DATASETS):
        g = graded.get(ds["assignment"], {})
        ds["scored"] = g.get("submissions", 0)
        ds["abstain"] = g.get("abstain", 0)
        ds["running"] = RUNNER.is_running(ds["assignment"])
        out.append(ds)
    return jsonify(out)


@app.post("/api/assignments/<assign>/upload")
def api_upload(assign: str):
    """Multipart upload: submissions (multiple, or a .zip) + optional
    reference / description / config single files."""
    try:
        assign = F.safe_assign(assign)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    saved: dict = {}
    for kind in F.META_FILES:
        f = request.files.get(kind)
        if f and f.filename:
            saved[kind] = F.save_meta(DATASETS, assign, kind, f.read())
    subs = request.files.getlist("submissions")
    items = [(f.filename, f.read()) for f in subs if f and f.filename]
    if items:
        saved["submissions"] = F.add_submissions(DATASETS, assign, items)
    return jsonify({"assignment": assign, "saved": saved,
                    "status": F.dataset_status(DATASETS, assign)})


@app.post("/api/assignments/<assign>/grade")
def api_grade(assign: str):
    """Kick off a background pipeline run (preprocess→score→report)."""
    try:
        assign = F.safe_assign(assign)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    body = request.get_json(force=True, silent=True) or {}
    engine = body.get("engine", "general")
    if engine not in ("numeric", "general"):
        return jsonify({"error": "engine must be 'numeric' or 'general'"}), 400
    cmd = build_cmd(assign, engine, llm=body.get("llm"), memory=body.get("memory"))
    log_path = WORKSPACE / assign / "grade.log"
    manifest = WORKSPACE / assign / "run_manifest.json"
    try:
        st = RUNNER.start(assign, cmd, log_path, manifest)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify(st)


@app.get("/api/assignments/<assign>/run")
def api_run_status(assign: str):
    return jsonify(RUNNER.status(F.safe_assign(assign)))


@app.get("/api/assignments/<assign>")
def api_assignment(assign: str):
    return jsonify(D.assignment_detail(WORKSPACE, assign))


@app.get("/api/assignments/<assign>/submissions/<sid>")
def api_submission(assign: str, sid: str):
    detail = D.submission_detail(WORKSPACE, assign, sid)
    if detail is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(detail)


@app.post("/api/assignments/<assign>/submissions/<sid>/decision")
def api_decision(assign: str, sid: str):
    body = request.get_json(force=True, silent=True) or {}
    try:
        rec = D.save_decision(
            WORKSPACE, assign, sid,
            decision=body.get("decision", ""),
            final_score=body.get("final_score"),
            note=body.get("note", ""),
            reviewer=body.get("reviewer", ""),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(rec)


def main():
    port = int(os.environ.get("JN_PORT", "5000"))
    app.run(host=os.environ.get("JN_HOST", "127.0.0.1"), port=port, debug=False)


if __name__ == "__main__":
    main()
