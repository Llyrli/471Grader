#!/usr/bin/env python3
"""Eval harness — measure AI grading against N human-scored notebooks (ground truth).

Replaces the legacy `calibrate.py` (which used the obsolete
`dimension_scores`/`weighted_total` schema). This harness speaks the CURRENT
scored schema (Qi_score / final_score / diagnostics / status) and measures the
three things this grader's design actually claims:

  1. SCORE AGREEMENT — how close are AI scores to human scores?
       per-question & final MAD, exact/within-tolerance agreement, quadratic
       weighted Cohen's κ (QWK) over Qi_score.

  2. DEVIATION-LOCALIZATION HIT RATE — when the AI says a question is wrong and
       localizes the first divergence, does that match where the human says the
       error is? (Requires human labels; see schema below.) Plus error-class
       hit rate. This is the deterministic-signal quality metric.

  3. ABSTENTION / SELECTIVE-GRADING QUALITY — does abstaining actually filter the
       cases the AI would have gotten wrong?
         - AUTO accuracy  : agreement rate on NON-abstained samples (should be high)
         - Abstain precision: fraction of abstained samples the AI *would* have
                              gotten wrong (|Δfinal| > tol) — abstention was justified
         - Abstain recall   : fraction of all would-be-wrong samples that were
                              abstained (caught before publishing a bad grade)

Ground-truth files: one JSON per student in --human, named `<student_id>-human.json`
(or `<student_id>_human.json`). Minimal schema (a teacher fills this in):

    {
      "student_id": "anon-001",
      "Q1_score": 10, "Q2_score": 4, "Q3_score": 4,   // per-question 0..10
      "final_score": 18,                               // optional; else summed
      "labels": {                                      // optional, for hit-rate
        "Q3": {"first_divergence": "global_stiffness",
               "error_class": "physics_modeling"}
      }
    }

Paired against the AI's `<student_id>_scored.json` in --scored.

Usage:
    python eval_harness.py --human <human_dir> --scored <scored_dir> \
        [--output <report.md>] [--questions Q1 Q2 Q3] [--score-tol 1.0]
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("eval_harness")

# Agreement thresholds (report-only flags; tune per course).
QWK_GOOD = 0.70
MAD_GOOD = 1.0          # per-question points
FINAL_TOL_DEFAULT = 1.0  # |AI - human| final-score points counted as "agreement"


# ---------------------------------------------------------------------------
# Loading / pairing
# ---------------------------------------------------------------------------

def _student_id_from_human(path: Path) -> str:
    stem = path.stem
    for suffix in ("-human", "_human"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def load_pairs(human_dir: Path, scored_dir: Path) -> list[tuple[dict, dict]]:
    """Pair each human-scored file with the AI scored file for the same student."""
    pairs: list[tuple[dict, dict]] = []
    human_files = sorted(human_dir.glob("*-human.json")) + sorted(human_dir.glob("*_human.json"))
    for hp in human_files:
        sid = _student_id_from_human(hp)
        ai_path = scored_dir / f"{sid}_scored.json"
        if not ai_path.exists():
            logger.warning("No AI scored file for %s (expected %s)", sid, ai_path.name)
            continue
        try:
            human = json.loads(hp.read_text(encoding="utf-8"))
            ai = json.loads(ai_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Skipping %s: %s", sid, exc)
            continue
        human.setdefault("student_id", sid)
        pairs.append((human, ai))
    logger.info("Paired %d human/AI submissions", len(pairs))
    return pairs


def _final(record: dict, questions: list[str]) -> float:
    if "final_score" in record and record["final_score"] is not None:
        return float(record["final_score"])
    return float(sum(record.get(f"{q}_score", 0) or 0 for q in questions))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def quadratic_weighted_kappa(human: list[int], ai: list[int], min_s: int, max_s: int) -> float:
    """QWK over integer scores in [min_s, max_s]."""
    n_cat = max_s - min_s + 1
    if not human or n_cat < 2:
        return float("nan")
    O = np.zeros((n_cat, n_cat))
    for h, a in zip(human, ai):
        O[int(round(h)) - min_s, int(round(a)) - min_s] += 1
    O /= O.sum()
    row, col = O.sum(1), O.sum(0)
    E = np.outer(row, col)
    W = np.zeros((n_cat, n_cat))
    for i in range(n_cat):
        for j in range(n_cat):
            W[i, j] = (i - j) ** 2 / (n_cat - 1) ** 2
    denom = (W * E).sum()
    return 1.0 - (W * O).sum() / denom if denom else 1.0


def mad(human: list[float], ai: list[float]) -> float:
    return float(np.mean(np.abs(np.array(human) - np.array(ai)))) if human else float("nan")


@dataclass
class QuestionAgreement:
    question: str
    n: int
    mad: float
    qwk: float
    exact_rate: float


def per_question_agreement(pairs, questions) -> list[QuestionAgreement]:
    out = []
    for q in questions:
        h = [float(hm.get(f"{q}_score", 0) or 0) for hm, _ in pairs]
        a = [float(ai.get(f"{q}_score", 0) or 0) for _, ai in pairs]
        if not h:
            continue
        exact = float(np.mean([1.0 if round(x) == round(y) else 0.0 for x, y in zip(h, a)]))
        out.append(QuestionAgreement(
            question=q, n=len(h), mad=round(mad(h, a), 3),
            qwk=round(quadratic_weighted_kappa([int(round(x)) for x in h],
                                               [int(round(y)) for y in a], 0, 10), 3),
            exact_rate=round(exact, 3),
        ))
    return out


def localization_hit_rate(pairs, questions) -> dict[str, Any]:
    """Among questions the human labeled with a locus, how often does the AI's
    deterministic first_divergence (and error_class) match?"""
    locus_total = locus_hit = 0
    cls_total = cls_hit = 0
    misses: list[str] = []
    for hm, ai in pairs:
        labels = hm.get("labels", {}) or {}
        diags = ai.get("diagnostics", {}) or {}
        for q in questions:
            lab = labels.get(q)
            if not lab:
                continue
            ai_diag = diags.get(q, {})
            if lab.get("first_divergence"):
                locus_total += 1
                if ai_diag.get("first_divergence") == lab["first_divergence"]:
                    locus_hit += 1
                else:
                    misses.append(
                        f"{hm.get('student_id')}/{q}: human={lab['first_divergence']} "
                        f"ai={ai_diag.get('first_divergence')}"
                    )
            if lab.get("error_class"):
                cls_total += 1
                if ai_diag.get("error_class") == lab["error_class"]:
                    cls_hit += 1
    return {
        "locus_total": locus_total,
        "locus_hit": locus_hit,
        "locus_rate": round(locus_hit / locus_total, 3) if locus_total else None,
        "class_total": cls_total,
        "class_hit": cls_hit,
        "class_rate": round(cls_hit / cls_total, 3) if cls_total else None,
        "misses": misses,
    }


def abstention_analysis(pairs, questions, tol: float) -> dict[str, Any]:
    """Selective-grading quality: does abstention catch the would-be-wrong ones?"""
    auto_ok = auto_n = 0
    abst_justified = abst_n = 0
    would_be_wrong = caught = 0
    for hm, ai in pairs:
        h_final = _final(hm, questions)
        a_final = _final(ai, questions)
        wrong = abs(h_final - a_final) > tol
        abstained = ai.get("status") == "ABSTAIN"
        if wrong:
            would_be_wrong += 1
            if abstained:
                caught += 1
        if abstained:
            abst_n += 1
            if wrong:
                abst_justified += 1
        else:
            auto_n += 1
            if not wrong:
                auto_ok += 1
    return {
        "auto_n": auto_n,
        "auto_accuracy": round(auto_ok / auto_n, 3) if auto_n else None,
        "abstain_n": abst_n,
        "abstain_precision": round(abst_justified / abst_n, 3) if abst_n else None,
        "would_be_wrong": would_be_wrong,
        "abstain_recall": round(caught / would_be_wrong, 3) if would_be_wrong else None,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _flag(ok: bool) -> str:
    return "✅" if ok else "❌"


def render_report(pairs, questions, tol: float) -> str:
    h_final = [_final(hm, questions) for hm, _ in pairs]
    a_final = [_final(ai, questions) for _, ai in pairs]
    final_mad = mad(h_final, a_final)
    within = float(np.mean([1.0 if abs(x - y) <= tol else 0.0 for x, y in zip(h_final, a_final)])) if h_final else float("nan")
    final_qwk = quadratic_weighted_kappa([int(round(x)) for x in h_final],
                                         [int(round(y)) for y in a_final], 0, 10 * len(questions))

    pq = per_question_agreement(pairs, questions)
    loc = localization_hit_rate(pairs, questions)
    ab = abstention_analysis(pairs, questions, tol)

    L = [
        "# Eval Harness Report",
        "",
        f"- 日期: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        f"- 配对样本: {len(pairs)}  |  最终分容差: ±{tol}",
        "",
        "## 1. 分数一致性 (vs 人工 ground truth)",
        "",
        "| 指标 | 值 | 阈值 | 结果 |",
        "|---|---|---|---|",
        f"| 最终分 MAD | {final_mad:.3f} | ≤ {MAD_GOOD*len(questions):.1f} | {_flag(final_mad <= MAD_GOOD*len(questions))} |",
        f"| 最终分一致率 (±{tol}) | {within*100:.1f}% | — | — |",
        f"| 最终分 QWK | {final_qwk:.3f} | ≥ {QWK_GOOD} | {_flag(final_qwk >= QWK_GOOD)} |",
        "",
        "### 逐题一致性",
        "",
        "| 题 | n | MAD | QWK | 精确一致率 |",
        "|---|---|---|---|---|",
    ]
    for a in pq:
        L.append(f"| {a.question} | {a.n} | {a.mad} | {a.qwk} | {a.exact_rate*100:.0f}% |")

    L += [
        "",
        "## 2. 错因定位命中率 (确定性信号质量)",
        "",
    ]
    if loc["locus_total"]:
        L += [
            f"- **偏差定位命中率:** {loc['locus_hit']}/{loc['locus_total']} = "
            f"**{loc['locus_rate']*100:.1f}%**",
        ]
        if loc["class_total"]:
            L.append(f"- **错因分类命中率:** {loc['class_hit']}/{loc['class_total']} = "
                     f"{loc['class_rate']*100:.1f}%")
        if loc["misses"]:
            L.append("- 未命中:")
            L += [f"  - {m}" for m in loc["misses"]]
    else:
        L.append("- (无人工 locus 标注 — 在 human JSON 的 `labels` 字段补充以启用此指标)")

    L += [
        "",
        "## 3. 弃判 / 选择性评分质量",
        "",
        "| 指标 | 值 | 含义 |",
        "|---|---|---|",
        f"| AUTO 样本数 | {ab['auto_n']} | 自动给分 |",
        f"| AUTO 准确率 | {_pct(ab['auto_accuracy'])} | 自动给分中与人工一致(±{tol})的比例,越高越好 |",
        f"| 弃判样本数 | {ab['abstain_n']} | 转人工 |",
        f"| 弃判精度 | {_pct(ab['abstain_precision'])} | 弃判样本中确实会判错的比例(弃判是否值得) |",
        f"| 应弃判总数 | {ab['would_be_wrong']} | 与人工差异 > ±{tol} 的样本 |",
        f"| 弃判召回 | {_pct(ab['abstain_recall'])} | 会判错的样本中被成功拦截的比例 |",
        "",
        "> 选择性评分的目标: **AUTO 准确率高** 且 **弃判召回高** —— 自信的判得准,",
        "> 判不准的都转人工。",
        "",
        "---",
        "*Generated by eval_harness.py*",
    ]
    return "\n".join(L)


def _pct(v) -> str:
    return f"{v*100:.1f}%" if isinstance(v, (int, float)) else "n/a"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--human", type=Path, required=True,
                        help="Directory of <student_id>-human.json ground-truth files")
    parser.add_argument("--scored", type=Path, required=True,
                        help="Directory of <student_id>_scored.json AI files")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Markdown report path (default: <scored>/../eval_report.md)")
    parser.add_argument("--questions", nargs="+", default=["Q1", "Q2", "Q3"],
                        help="Question keys (default: Q1 Q2 Q3)")
    parser.add_argument("--score-tol", type=float, default=FINAL_TOL_DEFAULT, dest="tol",
                        help=f"Final-score agreement tolerance (default: {FINAL_TOL_DEFAULT})")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    if not args.human.is_dir() or not args.scored.is_dir():
        logger.error("--human and --scored must be existing directories")
        raise SystemExit(1)

    pairs = load_pairs(args.human, args.scored)
    if not pairs:
        logger.error("No paired samples found.")
        raise SystemExit(1)

    report = render_report(pairs, args.questions, args.tol)
    out_path = args.output or (args.scored.parent / "eval_report.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Eval report → %s", out_path)
    print(report)


if __name__ == "__main__":
    main()
