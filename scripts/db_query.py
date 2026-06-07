"""Query / aggregate the JN Grader task bank.

Subcommands:
    list      List assignments with submission counts and average score.
    stats     Per-question and overall stats for one assignment (--key).
    errors    Most common failing question across an assignment (--key).
    top       Lowest/highest submissions for an assignment (--key).

Usage:
    python db_query.py list
    python db_query.py stats --key ME471-HW2
    python db_query.py errors --key ME471-HW2
"""

from __future__ import annotations

import argparse

import db_common


def _assignment_id(cur, key: str) -> int | None:
    cur.execute("SELECT id FROM assignments WHERE key = %s", (key,))
    row = cur.fetchone()
    return row[0] if row else None


def cmd_list(conn, _args) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.key, a.title, a.max_score,
                   COUNT(s.id)            AS submissions,
                   ROUND(AVG(r.final_score), 1) AS avg_score
            FROM assignments a
            LEFT JOIN submissions s ON s.assignment_id = a.id
            LEFT JOIN results r ON r.submission_id = s.id
            GROUP BY a.id ORDER BY a.created_at
            """
        )
        rows = cur.fetchall()
    if not rows:
        print("(no assignments)")
        return
    print(f"{'KEY':<16}{'SUBS':>6}{'AVG':>8}  TITLE")
    for key, title, mx, subs, avg in rows:
        avg_s = f"{avg}/{mx}" if avg is not None else "-"
        print(f"{key:<16}{subs:>6}{avg_s:>8}  {title or ''}")


def _has_problem_scores(cur, aid: int) -> bool:
    cur.execute(
        """
        SELECT 1 FROM problem_scores ps
        JOIN results r ON r.id = ps.result_id
        JOIN submissions s ON s.id = r.submission_id
        WHERE s.assignment_id = %s LIMIT 1
        """,
        (aid,),
    )
    return cur.fetchone() is not None


def cmd_stats(conn, args) -> None:
    with conn.cursor() as cur:
        aid = _assignment_id(cur, args.key)
        if aid is None:
            print(f"assignment not found: {args.key}")
            return
        if _has_problem_scores(cur, aid):
            cur.execute(
                """
                SELECT COUNT(DISTINCT s.id),
                       ROUND(AVG(r.final_score),1), MIN(r.final_score), MAX(r.final_score),
                       MAX(r.max_score)
                FROM results r JOIN submissions s ON s.id = r.submission_id
                WHERE s.assignment_id = %s
                """,
                (aid,),
            )
            n, avg, lo, hi, mx = cur.fetchone()
            print(f"=== {args.key} — {n} submission(s) ===")
            print(f"final: avg {avg}  min {lo}  max {hi}  (out of {mx})")
            cur.execute(
                """
                SELECT ps.name, ROUND(AVG(ps.score),1), MAX(ps.max), MIN(ps.score), MAX(ps.score)
                FROM problem_scores ps
                JOIN results r ON r.id = ps.result_id
                JOIN submissions s ON s.id = r.submission_id
                WHERE s.assignment_id = %s
                GROUP BY ps.name ORDER BY ps.name
                """,
                (aid,),
            )
            print(f"{'prob':<6}{'avg':>7}{'max':>6}{'min':>6}{'hi':>5}")
            for name, a, m, lo2, hi2 in cur.fetchall():
                print(f"{name:<6}{a:>7}{m:>6}{lo2:>6}{hi2:>5}")
            return
        cur.execute(
            """
            SELECT COUNT(*),
                   ROUND(AVG(final_score),1), MIN(final_score), MAX(final_score),
                   ROUND(AVG(q1_score),1), ROUND(AVG(q2_score),1), ROUND(AVG(q3_score),1),
                   ROUND(AVG(q1_result),2), ROUND(AVG(q2_result),2), ROUND(AVG(q3_result),2),
                   ROUND(AVG(q1_process),1), ROUND(AVG(q2_process),1), ROUND(AVG(q3_process),1)
            FROM results r JOIN submissions s ON s.id = r.submission_id
            WHERE s.assignment_id = %s
            """,
            (aid,),
        )
        (n, avg, lo, hi, q1, q2, q3,
         q1r, q2r, q3r, q1p, q2p, q3p) = cur.fetchone()
    if not n:
        print(f"no scored submissions for {args.key}")
        return
    print(f"=== {args.key} — {n} submission(s) ===")
    print(f"final: avg {avg}  min {lo}  max {hi}")
    print(f"{'':<6}{'avg':>6}{'result(0-3)':>13}{'process(0-7)':>14}")
    for q, a, r, p in [("Q1", q1, q1r, q1p), ("Q2", q2, q2r, q2p), ("Q3", q3, q3r, q3p)]:
        print(f"{q:<6}{a:>6}{r:>13}{p:>14}")


def cmd_errors(conn, args) -> None:
    with conn.cursor() as cur:
        aid = _assignment_id(cur, args.key)
        if aid is None:
            print(f"assignment not found: {args.key}")
            return
        # count autograde failures per question (result score == 0)
        cur.execute(
            """
            SELECT
              SUM(CASE WHEN q1_result = 0 THEN 1 ELSE 0 END),
              SUM(CASE WHEN q2_result = 0 THEN 1 ELSE 0 END),
              SUM(CASE WHEN q3_result = 0 THEN 1 ELSE 0 END),
              COUNT(*)
            FROM results r JOIN submissions s ON s.id = r.submission_id
            WHERE s.assignment_id = %s
            """,
            (aid,),
        )
        q1f, q2f, q3f, n = cur.fetchone()
    if not n:
        print(f"no scored submissions for {args.key}")
        return
    print(f"=== {args.key} — autograde failure rate (n={n}) ===")
    for q, f in [("Q1", q1f), ("Q2", q2f), ("Q3", q3f)]:
        pct = 100.0 * f / n
        print(f"{q}: {f}/{n} failed ({pct:.0f}%)")


def cmd_top(conn, args) -> None:
    with conn.cursor() as cur:
        aid = _assignment_id(cur, args.key)
        if aid is None:
            print(f"assignment not found: {args.key}")
            return
        cur.execute(
            """
            SELECT s.student_id, COALESCE(s.student_name,''), COALESCE(s.student_no,''),
                   r.final_score, s.source_file
            FROM results r JOIN submissions s ON s.id = r.submission_id
            WHERE s.assignment_id = %s
            ORDER BY r.final_score %s LIMIT %s
            """ % ("%s", "ASC" if args.lowest else "DESC", "%s"),
            (aid, args.n),
        )
        rows = cur.fetchall()
    label = "lowest" if args.lowest else "highest"
    print(f"=== {args.key} — {label} {len(rows)} ===")
    for sid, name, no, score, src in rows:
        who = (f"{name} {no}".strip()) or src
        print(f"{sid:<12} {score:>3}  {who}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Query the JN Grader task bank.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("stats", "errors"):
        sp = sub.add_parser(name)
        sp.add_argument("--key", required=True)
    sp = sub.add_parser("top")
    sp.add_argument("--key", required=True)
    sp.add_argument("--n", type=int, default=5)
    sp.add_argument("--lowest", action="store_true")
    args = p.parse_args(argv)

    conn = db_common.connect()
    {"list": cmd_list, "stats": cmd_stats, "errors": cmd_errors, "top": cmd_top}[args.cmd](conn, args)
    conn.close()


if __name__ == "__main__":
    main()
