"""Generate a process-based grading rubric from an assignment description.

Uses an LLM (OpenAI-compatible API) to produce a structured YAML rubric
covering FE process criteria for Q1/Q2/Q3 (7 process points each).

Usage:
    python generate_rubric.py <description.txt> --output rubric.yaml
    python generate_rubric.py <description.txt> --output rubric.yaml \\
        --api-key <key> \\
        --base-url https://api.siliconflow.cn/v1 \\
        --model Qwen/Qwen2.5-72B-Instruct

The generated rubric.yaml is then passed to score_notebooks.py via --rubric.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
from openai import OpenAI

logger = logging.getLogger("generate_rubric")

DEFAULT_MODEL    = "Qwen/Qwen2.5-72B-Instruct"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"

# ---------------------------------------------------------------------------
# System prompt (from tellclaude2.txt)
# ---------------------------------------------------------------------------

RUBRIC_SYSTEM_PROMPT = """\
You are an expert course instructor designing a grading rubric for a finite \
element (FE) homework assignment.

The instructor has already implemented automatic grading for correctness \
(e.g., unit tests).
Your task is to generate a structured PROCESS-BASED rubric only.

Do NOT generate result-based scoring.
Do NOT include correctness from test results.
The correctness component is already handled separately.

Your rubric must evaluate only the quality and correctness of the finite \
element implementation process.

==================================================
YOUR TASK
==================================================

From the assignment description, generate a structured rubric that:

1. Covers all required FE process components.
2. Focuses on mathematical and algorithmic correctness.
3. Does NOT enforce implementation style.
4. Is independent of any specific reference solution structure.
5. Is suitable for assigning up to 7 process points per question.

For each question (Q1, Q2, Q3), identify:
- Required technical components
- Key algorithmic steps
- Critical mathematical formulations
- Boundary condition logic
- Post-processing requirements (if applicable)

==================================================
RUBRIC DESIGN PRINCIPLES
==================================================

Your rubric must:
- Be structured and hierarchical.
- Be technical and specific.
- Avoid vague statements like "code is good".
- Focus on FE pipeline logic:
    * problem setup
    * element formulation
    * assembly
    * boundary conditions
    * solver
    * post-processing
    * physical consistency
- Allow partial credit based on conceptual correctness.

==================================================
OUTPUT FORMAT (STRICT YAML)
==================================================

Return ONLY valid YAML in this format:

questions:
  - name: Q1
    process_points: 7
    criteria:
      - description: "..."
        weight: integer
      - description: "..."
        weight: integer

  - name: Q2
    process_points: 7
    criteria:
      - description: "..."
        weight: integer
      - description: "..."
        weight: integer

  - name: Q3
    process_points: 7
    criteria:
      - description: "..."
        weight: integer
      - description: "..."
        weight: integer

Rules:
- Weights under each question must sum to 7.
- Do not include correctness scoring.
- Do not include total score beyond process_points.
- Do not include markdown formatting.
- Output YAML only.

==================================================
CONSTRAINTS
==================================================

- Do NOT copy implementation details.
- Do NOT assume a specific coding structure.
- Do NOT require identical implementation to any reference solution.
- Abstract only the mathematical and procedural requirements.

You are generating a grading rubric, not grading the submission.\
"""


# ---------------------------------------------------------------------------
# Rubric validation
# ---------------------------------------------------------------------------

def validate_rubric(rubric: dict) -> list[str]:
    """Return a list of validation error messages (empty = valid)."""
    errors = []
    questions = rubric.get("questions")
    if not isinstance(questions, list):
        return ["'questions' key missing or not a list"]

    for q in questions:
        name = q.get("name", "?")
        criteria = q.get("criteria", [])
        if not criteria:
            errors.append(f"{name}: no criteria defined")
            continue
        total = sum(int(c.get("weight", 0)) for c in criteria)
        if total != 7:
            errors.append(f"{name}: weights sum to {total}, expected 7")

    return errors


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_rubric(
    description: str,
    client: OpenAI,
    model: str,
) -> dict:
    """Call LLM with the assignment description and return parsed rubric dict."""
    response = client.chat.completions.create(
        model=model,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": RUBRIC_SYSTEM_PROMPT},
            {"role": "user",   "content": description},
        ],
    )
    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("yaml"):
            raw = raw[4:]
        raw = raw.strip()

    return yaml.safe_load(raw)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a process-based rubric from an assignment description.",
    )
    parser.add_argument("description_file", type=Path,
                        help="Text file containing the assignment description")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Output path for the generated rubric.yaml")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (overrides LLM_API_KEY env var)")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL,
                        help=f"OpenAI-compatible base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not args.description_file.exists():
        logger.error("Description file not found: %s", args.description_file)
        raise SystemExit(1)

    api_key = args.api_key or os.environ.get("LLM_API_KEY")
    if not api_key:
        logger.error("No API key. Pass --api-key or set LLM_API_KEY.")
        raise SystemExit(1)

    description = args.description_file.read_text(encoding="utf-8")
    logger.info("Loaded description: %d chars", len(description))

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    logger.info("Calling LLM (%s) …", args.model)

    rubric = generate_rubric(description, client, args.model)

    # Validate
    errors = validate_rubric(rubric)
    if errors:
        logger.warning("Rubric validation warnings:")
        for e in errors:
            logger.warning("  %s", e)
    else:
        logger.info("Rubric validated OK.")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        yaml.dump(rubric, f, allow_unicode=True, sort_keys=False)
    logger.info("Rubric saved → %s", args.output)

    # Print summary
    for q in rubric.get("questions", []):
        logger.info("  %s: %d criteria, weights sum=%d",
                    q["name"],
                    len(q.get("criteria", [])),
                    sum(c.get("weight", 0) for c in q.get("criteria", [])))


if __name__ == "__main__":
    main()
