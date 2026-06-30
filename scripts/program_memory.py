"""Program memory — sediment course grading conventions + high-frequency error
patterns from past scored assignments, and reuse them consistently across new
assignments.

Closing the loop (P5):
    grade  →  sediment (this script)  →  next assignment grades WITH memory  →  re-sediment

The store is a portable per-course JSON file (`workspace/memory/<course>.json`).
It is built DETERMINISTICALLY from the structured signals the pipeline already
emits — no LLM is required to sediment:

  - numeric engine  (score_notebooks.py): per-question `diagnostics` carry an
    `error_class` + deterministic `first_divergence` locus → grouped into error
    patterns keyed by (error_class, locus), counted, with representative
    explanations/fixes.
  - general engine  (score_general.py): per-`problems` deductions (score < max)
    are collected as recurring deduction notes per problem.

An OPTIONAL `--distill` pass uses the LLM only to compress the collected raw
feedback corpus into a few concise, reusable convention bullets — it never
invents facts and never affects scores; it only summarizes what graders already
wrote.

At grading time, `format_memory_block()` renders a compact, FREQUENCY-RANKED
block that both graders inject AFTER the deterministic findings, framed as
advisory priors that MUST NOT override execution facts (determinism first).

Usage:
    # sediment one or more assignments' scored outputs into the course memory
    python program_memory.py sediment --course ME471 \
        --scored workspace/HW2/scored workspace/HW4/scored \
        --store workspace/memory/ME471.json

    # inspect the accumulated memory
    python program_memory.py show --store workspace/memory/ME471.json

    # (optional) LLM-distill free-text feedback into convention bullets
    python program_memory.py sediment --course ME471 \
        --scored workspace/HW4/scored --store workspace/memory/ME471.json \
        --distill --base-url https://api.deepseek.com --model deepseek-chat
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("program_memory")

# How many of each category to surface in the injected grading block.
DEFAULT_TOP_PATTERNS = 8
DEFAULT_TOP_CONVENTIONS = 8
# Keep at most this many representative free-text examples per pattern in the store.
MAX_EXAMPLES_PER_PATTERN = 5
# An error pattern is only "frequent" enough to inject once seen this many times.
MIN_PATTERN_COUNT = 1


# ---------------------------------------------------------------------------
# Store I/O
# ---------------------------------------------------------------------------

def empty_store(course: str) -> dict[str, Any]:
    return {
        "course": course,
        "updated_at": None,
        "sources": [],            # assignment dirs/keys folded in so far
        "conventions": [],        # [{text, source, kind}]
        "error_patterns": {},     # signature -> pattern dict
        "problem_notes": {},      # "<assignment>::<Pname>" -> {count, max, examples[]}
    }


def load_store(path: Path, course: str | None = None) -> dict[str, Any]:
    if path.exists():
        store = json.loads(path.read_text(encoding="utf-8"))
        # Tolerate older/empty stores.
        store.setdefault("conventions", [])
        store.setdefault("error_patterns", {})
        store.setdefault("problem_notes", {})
        store.setdefault("sources", [])
        if course:
            store["course"] = course
        return store
    return empty_store(course or "course")


def save_store(path: Path, store: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    store["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Sediment — deterministic aggregation from scored JSON
# ---------------------------------------------------------------------------

def _signature(error_class: str, locus: str | None) -> str:
    return f"{error_class}::{locus or 'unlocated'}"


def _fold_numeric(scored: dict[str, Any], assignment: str, store: dict[str, Any]) -> int:
    """Fold a numeric-engine scored record (Q*/diagnostics) into error patterns.

    Only AUTO records contribute (ABSTAIN ones were not trusted enough to grade,
    so their diagnoses shouldn't shape the course memory). Returns #patterns hit.
    """
    if scored.get("status") == "ABSTAIN":
        return 0
    diags = scored.get("diagnostics", {})
    if not isinstance(diags, dict):
        return 0
    patterns = store["error_patterns"]
    hits = 0
    for q, d in diags.items():
        if not isinstance(d, dict):
            continue
        ec = d.get("error_class", "none")
        if ec in ("none", "unknown", None):
            continue
        sig = _signature(ec, d.get("first_divergence"))
        p = patterns.get(sig)
        if p is None:
            p = patterns[sig] = {
                "signature": sig,
                "error_class": ec,
                "locus": d.get("first_divergence"),
                "count": 0,
                "assignments": [],
                "examples": [],
                "fix_hint": "",
            }
        p["count"] += 1
        hits += 1
        if assignment not in p["assignments"]:
            p["assignments"].append(assignment)
        expl = (d.get("explanation") or "").strip()
        if expl and len(p["examples"]) < MAX_EXAMPLES_PER_PATTERN and expl not in p["examples"]:
            p["examples"].append(expl)
        if not p["fix_hint"]:
            fix = (d.get("fix") or "").strip()
            if fix:
                p["fix_hint"] = fix
    return hits


def _fold_general(scored: dict[str, Any], assignment: str, store: dict[str, Any]) -> int:
    """Fold a general-engine scored record (problems[]) into per-problem deduction
    notes. A deduction = score < max; its feedback is the recurring reason. Returns
    #deductions collected.
    """
    problems = scored.get("problems")
    if not isinstance(problems, list):
        return 0
    notes = store["problem_notes"]
    hits = 0
    for p in problems:
        if not isinstance(p, dict):
            continue
        score, mx = p.get("score"), p.get("max")
        if score is None or mx is None or score >= mx:
            continue  # full marks → nothing to learn from
        fb = (p.get("feedback") or "").strip()
        if not fb:
            continue
        key = f"{assignment}::{p.get('name', '?')}"
        n = notes.get(key)
        if n is None:
            n = notes[key] = {"count": 0, "max": mx, "examples": []}
        n["count"] += 1
        hits += 1
        if len(n["examples"]) < MAX_EXAMPLES_PER_PATTERN and fb not in n["examples"]:
            n["examples"].append(fb)
    return hits


def sediment_dir(scored_dir: Path, assignment: str, store: dict[str, Any]) -> dict[str, int]:
    """Fold every *_scored.json in a directory into the store. Dispatches by
    schema (numeric vs general). Returns a small stats dict."""
    files = sorted(scored_dir.glob("*_scored.json"))
    stats = {"files": 0, "numeric_patterns": 0, "general_deductions": 0, "skipped": 0}
    for f in files:
        try:
            scored = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("  skip unreadable %s: %s", f.name, exc)
            stats["skipped"] += 1
            continue
        stats["files"] += 1
        if "diagnostics" in scored or any(k.endswith("_result_score") for k in scored):
            stats["numeric_patterns"] += _fold_numeric(scored, assignment, store)
        elif "problems" in scored:
            stats["general_deductions"] += _fold_general(scored, assignment, store)
    if assignment not in store["sources"]:
        store["sources"].append(assignment)
    return stats


def add_convention(store: dict[str, Any], text: str, source: str, kind: str = "convention") -> bool:
    """Append a convention bullet, de-duplicated by text. Returns True if added."""
    text = (text or "").strip()
    if not text:
        return False
    if any(c.get("text") == text for c in store["conventions"]):
        return False
    store["conventions"].append({"text": text, "source": source, "kind": kind})
    return True


# ---------------------------------------------------------------------------
# Optional LLM distillation (summarize free text → convention bullets)
# ---------------------------------------------------------------------------

DISTILL_SYSTEM = """\
You compress a corpus of grader feedback into a SHORT list of reusable grading
conventions and recurring student mistakes for this course. You MUST only
summarize patterns clearly present in the feedback — do NOT invent rules, do NOT
assign scores. Output ONLY valid JSON, no markdown:
{"conventions": ["<concise rule or recurring-mistake bullet>", ...]}
Aim for 3-8 bullets, each one sentence, generic enough to reuse on future
assignments in the same course."""


def distill_conventions(store: dict[str, Any], client: Any, max_chars: int = 8000) -> int:
    """Use the LLM to distill collected feedback into convention bullets.

    Advisory-only: summarizes existing grader text, never changes scores. Returns
    the number of new convention bullets added.
    """
    corpus: list[str] = []
    for p in store["error_patterns"].values():
        corpus.extend(p.get("examples", []))
    for n in store["problem_notes"].values():
        corpus.extend(n.get("examples", []))
    corpus = [c for c in corpus if c]
    if not corpus:
        logger.info("  nothing to distill (no feedback collected yet)")
        return 0
    blob = "\n- ".join(corpus)[:max_chars]
    user = f"Course: {store.get('course')}\nGrader feedback corpus:\n- {blob}"
    raw = client.complete(DISTILL_SYSTEM, user, max_tokens=600)
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        parsed = json.loads(raw.strip())
    except Exception as exc:
        logger.warning("  distill output unparseable: %s", exc)
        return 0
    added = 0
    for bullet in parsed.get("conventions", []):
        if add_convention(store, str(bullet), source="distilled", kind="distilled"):
            added += 1
    return added


# ---------------------------------------------------------------------------
# Retrieve / format — the block injected into grading prompts
# ---------------------------------------------------------------------------

def ranked_patterns(store: dict[str, Any], top_k: int, min_count: int = MIN_PATTERN_COUNT) -> list[dict]:
    pats = [p for p in store.get("error_patterns", {}).values() if p.get("count", 0) >= min_count]
    pats.sort(key=lambda p: p.get("count", 0), reverse=True)
    return pats[:top_k]


def pattern_text(p: dict) -> str:
    """Text used to embed/match an error pattern for semantic recall."""
    parts = [p.get("error_class", ""), p.get("locus") or "", p.get("fix_hint", "")]
    parts += p.get("examples", []) or []
    return " ".join(x for x in parts if x)


def relevant_patterns(
    store: dict[str, Any],
    query: str,
    embedder: Any,
    top_k: int,
    min_count: int = MIN_PATTERN_COUNT,
) -> list[dict]:
    """Error patterns most SEMANTICALLY similar to `query` (e.g. the current
    assignment's description / reference), via cosine over embeddings.

    Cross-assignment "similar-cause" recall: instead of only the globally most
    frequent patterns, surface the ones whose error text matches what THIS
    assignment is about. Ties broken by frequency. Built once per grading batch,
    so embedding N patterns is cheap (and free with the offline `hash` embedder).
    """
    from embeddings import cosine
    pats = [p for p in store.get("error_patterns", {}).values() if p.get("count", 0) >= min_count]
    if not pats:
        return []
    qv = embedder.embed_one(query or "")
    texts = [pattern_text(p) for p in pats]
    vecs = embedder.embed(texts)
    scored = [(cosine(qv, v), p.get("count", 0), p) for v, p in zip(vecs, pats)]
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [p for _, _, p in scored[:top_k]]


def format_memory_block(
    store: dict[str, Any],
    top_patterns: int = DEFAULT_TOP_PATTERNS,
    top_conventions: int = DEFAULT_TOP_CONVENTIONS,
    min_count: int = MIN_PATTERN_COUNT,
    query: str | None = None,
    embedder: Any = None,
) -> str:
    """Render a compact memory block for prompt injection.

    Patterns are ranked by FREQUENCY by default, or by SEMANTIC RELEVANCE to
    `query` when both `query` and `embedder` are given (cross-assignment
    similar-cause recall). Returns "" when the store has nothing actionable.
    The block is framed as advisory priors that must not override deterministic
    execution findings.
    """
    convs = store.get("conventions", [])[:top_conventions]
    semantic = bool(query and embedder)
    pats = (relevant_patterns(store, query, embedder, top_patterns, min_count)
            if semantic else ranked_patterns(store, top_patterns, min_count))
    if not convs and not pats:
        return ""

    lines = [
        "[PROGRAM MEMORY — COURSE GRADING CONVENTIONS & COMMON ERROR PATTERNS]",
        "Advisory priors sedimented from prior grading in this course, for "
        "CONSISTENCY across assignments. These are NOT ground truth: they must "
        "NOT override the deterministic execution findings above, and must not "
        "by themselves change a pass/fail. Use them only to inform wording, "
        "error classification, and partial-credit calibration.",
    ]
    if convs:
        lines.append("\nEstablished conventions:")
        for c in convs:
            lines.append(f"  - {c.get('text')}")
    if pats:
        header = ("\nError patterns most relevant to this assignment:" if semantic
                  else "\nFrequent error patterns in this course (most common first):")
        lines.append(header)
        for p in pats:
            locus = p.get("locus") or "unlocated"
            head = f"  - [{p.get('error_class')} @ {locus}] seen {p.get('count')}×"
            if p.get("fix_hint"):
                head += f"; typical fix: {p['fix_hint']}"
            lines.append(head)
            ex = p.get("examples") or []
            if ex:
                lines.append(f"      e.g. {ex[0]}")
    return "\n".join(lines)


def load_block(
    store_path: Path | None,
    query: str | None = None,
    embed_provider: str = "hash",
    embed_model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **kw,
) -> str:
    """Convenience for graders: load a store file and format its block; returns
    "" if the path is None/missing/empty. Never raises on a bad store.

    When `query` is given, patterns are ranked by SEMANTIC RELEVANCE to it
    (cross-assignment similar-cause recall) using `embed_provider` (default the
    offline `hash` embedder, so it works with no API). Falls back to frequency
    ranking if the embedder can't be built."""
    if not store_path:
        return ""
    try:
        if not Path(store_path).exists():
            logger.warning("memory store not found: %s — grading without memory", store_path)
            return ""
        store = load_store(Path(store_path))
        embedder = None
        if query:
            try:
                from embeddings import Embedder
                embedder = Embedder(provider=embed_provider, model=embed_model,
                                    api_key=api_key, base_url=base_url)
            except Exception as exc:
                logger.warning("memory embedder unavailable (%s) — frequency ranking", exc)
        return format_memory_block(store, query=query, embedder=embedder, **kw)
    except Exception as exc:
        logger.warning("could not load memory store %s: %s — grading without memory", store_path, exc)
        return ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_sediment(args: argparse.Namespace) -> None:
    store = load_store(args.store, course=args.course)
    total = {"files": 0, "numeric_patterns": 0, "general_deductions": 0, "skipped": 0}
    for scored_dir in args.scored:
        if not scored_dir.is_dir():
            logger.warning("not a directory, skipping: %s", scored_dir)
            continue
        assignment = args.assignment or scored_dir.parent.name or scored_dir.name
        # Folding is additive; guard against double-counting on re-runs. Pass
        # --force to re-fold (e.g. after re-grading) — start from a fresh store
        # if you want exact counts.
        if assignment in store["sources"] and not args.force:
            logger.info("skip [%s] — already folded (use --force to re-fold)", assignment)
            continue
        stats = sediment_dir(scored_dir, assignment, store)
        logger.info("folded %-22s files=%d numeric=%d general=%d",
                    f"[{assignment}]", stats["files"],
                    stats["numeric_patterns"], stats["general_deductions"])
        for k in total:
            total[k] += stats[k]

    if args.distill:
        try:
            from llm_client import DEFAULT_ANTHROPIC_MODEL, LLMClient
            import os
            api_key = args.api_key or os.environ.get("LLM_API_KEY")
            if args.provider == "anthropic":
                api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            model = args.model or (DEFAULT_ANTHROPIC_MODEL if args.provider == "anthropic" else "deepseek-chat")
            client = LLMClient(provider=args.provider, api_key=api_key,
                               base_url=args.base_url, model=model)
            added = distill_conventions(store, client)
            logger.info("distilled %d convention bullet(s)", added)
        except Exception as exc:
            logger.warning("distillation skipped: %s", exc)

    save_store(args.store, store)
    logger.info(
        "Memory updated: %s  (sources=%d, patterns=%d, conventions=%d, problem-notes=%d)",
        args.store, len(store["sources"]), len(store["error_patterns"]),
        len(store["conventions"]), len(store["problem_notes"]),
    )


def _cmd_show(args: argparse.Namespace) -> None:
    store = load_store(args.store)
    print(f"Course: {store.get('course')}   updated: {store.get('updated_at')}")
    print(f"Sources: {', '.join(store.get('sources', [])) or '(none)'}")
    if getattr(args, "query", None):
        print(f"Semantic query: {args.query!r}  (embedder: {args.embed_provider})")
    print(f"Collected: {len(store.get('error_patterns', {}))} error pattern(s), "
          f"{len(store.get('conventions', []))} convention(s), "
          f"{len(store.get('problem_notes', {}))} problem-note group(s)")

    notes = store.get("problem_notes", {})
    if notes:
        print("\nPer-problem deduction notes (run `sediment --distill` to turn these "
              "into reusable conventions):")
        for key, n in sorted(notes.items(), key=lambda kv: kv[1].get("count", 0), reverse=True):
            print(f"  - {key}: {n.get('count')} deduction(s) below max {n.get('max')}")

    print("\n--- injected block preview ---\n")
    block = load_block(args.store, query=getattr(args, "query", None),
                       embed_provider=getattr(args, "embed_provider", "hash"))
    print(block or "(empty — no conventions/patterns to inject yet; numeric-engine "
                   "diagnostics or `--distill` populate this)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Course program memory: sediment + reuse grading patterns.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sediment", help="Fold scored outputs into the course memory store.")
    s.add_argument("--course", required=True, help="Course key, e.g. ME471")
    s.add_argument("--scored", nargs="+", type=Path, required=True,
                   help="One or more scored/ directories to fold in")
    s.add_argument("--store", type=Path, required=True, help="Course memory JSON file")
    s.add_argument("--assignment", default=None,
                   help="Override the assignment label (default: each scored dir's parent name)")
    s.add_argument("--force", action="store_true",
                   help="Re-fold a source already in the store (default: skip to avoid double-counting)")
    s.add_argument("--distill", action="store_true",
                   help="Also LLM-distill collected feedback into convention bullets (advisory only)")
    s.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    s.add_argument("--api-key", default=None)
    s.add_argument("--base-url", default="https://api.deepseek.com")
    s.add_argument("--model", default=None)
    s.set_defaults(func=_cmd_sediment)

    sh = sub.add_parser("show", help="Print the store + the block it would inject.")
    sh.add_argument("--store", type=Path, required=True)
    sh.add_argument("--query", default=None,
                    help="Rank patterns by semantic relevance to this text (e.g. an "
                         "assignment description) instead of by frequency")
    sh.add_argument("--embed-provider", choices=["hash", "openai"], default="hash",
                    help="Embedder for --query (default: hash, offline)")
    sh.set_defaults(func=_cmd_show)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    args.func(args)


if __name__ == "__main__":
    main()
