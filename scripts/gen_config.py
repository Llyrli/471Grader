"""Turn an assignment description.txt into a general grading config.yaml via LLM.

The LLM reads the description and emits the per-problem structure (name, points,
type, desc); this script then injects the standard explanation criterion and
writes a clean config.yaml ready for score_general.py.

Usage:
    python gen_config.py datasets/HW4/description.txt --key ME471-HW4 \
        --output datasets/HW4/config.yaml \
        --base-url https://api.deepseek.com --model deepseek-chat
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import yaml

from llm_client import DEFAULT_ANTHROPIC_MODEL, LLMClient

logger = logging.getLogger("gen_config")

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"

EXPLANATION_CRITERION = (
    "Explanation & interpretation: the student should explain their approach in "
    "markdown between code cells (what each step computes and why), at a level "
    "comparable to the reference solution. Within each problem's points, reward "
    "clear interpretation and deduct for code with little or no explanation, even "
    "when the numbers are correct."
)

SYSTEM_PROMPT = """\
You convert a homework assignment description into a grading configuration.

Output ONLY valid YAML (no markdown fences) with exactly these keys:

problems:
  - {name: P1, points: <int>, type: <llm|link>, desc: "<concise summary>"}
  - {name: P2, points: <int>, type: <llm|link>, desc: "..."}
  ...

Rules:
- One entry per problem in the description, named P1, P2, P3, ... in order.
- points = the points stated for that problem (read them from the text).
- type = "link" ONLY if the problem asks the student to submit a link/URL or to
  upload/submit somewhere other than inside the notebook (i.e. nothing in the
  notebook to grade). Otherwise type = "llm".
- desc = 1-2 sentence summary of what the problem asks, including its sub-parts
  (a), (b), ... if any. Keep it concrete and short.
- Output YAML only. No prose, no code fences, no extra keys.
"""


def generate(description: str, client: LLMClient) -> dict:
    raw = client.complete(SYSTEM_PROMPT, description, max_tokens=1500).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("yaml"):
            raw = raw[4:]
        raw = raw.strip()
    return yaml.safe_load(raw)


def build_config(key: str, problems: list[dict], max_score: int | None) -> dict:
    total = sum(int(p.get("points", 0)) for p in problems)
    return {
        "assignment": key,
        "max_score": int(max_score) if max_score else total,
        "criteria": [EXPLANATION_CRITERION],
        "problems": [
            {
                "name": p.get("name", f"P{i+1}"),
                "points": int(p.get("points", 0)),
                "type": p.get("type", "llm"),
                "desc": (p.get("desc", "") or "").strip(),
            }
            for i, p in enumerate(problems)
        ],
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="LLM: description.txt -> config.yaml")
    p.add_argument("description_file", type=Path)
    p.add_argument("--key", required=True, help="Assignment key, e.g. ME471-HW4")
    p.add_argument("--output", "-o", type=Path, required=True)
    p.add_argument("--max-score", type=int, default=None,
                   help="Override total (default: sum of problem points)")
    p.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not args.description_file.exists():
        logger.error("description not found: %s", args.description_file)
        raise SystemExit(1)

    api_key = args.api_key or os.environ.get("LLM_API_KEY")
    if args.provider == "anthropic":
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    model = args.model or (DEFAULT_ANTHROPIC_MODEL if args.provider == "anthropic" else DEFAULT_MODEL)
    client = LLMClient(provider=args.provider, api_key=api_key, base_url=args.base_url, model=model)

    description = args.description_file.read_text(encoding="utf-8")
    logger.info("Generating config from %s (provider=%s, model=%s) …",
                args.description_file.name, args.provider, model)
    parsed = generate(description, client)
    problems = parsed.get("problems") if isinstance(parsed, dict) else None
    if not problems:
        logger.error("LLM did not return a 'problems' list. Raw: %s", parsed)
        raise SystemExit(1)

    cfg = build_config(args.key, problems, args.max_score)

    header = (
        f"# {args.key} — generated from {args.description_file.name} by gen_config.py.\n"
        f"# REVIEW before grading: check points, type (llm/link), and desc.\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False, width=100)

    logger.info("Wrote %s", args.output)
    for p in cfg["problems"]:
        logger.info("  %s: %d pts, type=%s", p["name"], p["points"], p["type"])
    logger.info("  total max_score = %d", cfg["max_score"])


if __name__ == "__main__":
    main()
