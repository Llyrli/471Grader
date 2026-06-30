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
import logging

import db_common

logger = logging.getLogger("db_query")


def _assignment_id(cur, key: str) -> int | None:
    cur.execute("SELECT id FROM assignments WHERE key = %s", (key,))
    row = cur.fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# Semantic similarity (pgvector)
# ---------------------------------------------------------------------------

def _assignment_text(cur, aid: int) -> str:
    """Text used to embed an assignment: title + description."""
    cur.execute("SELECT COALESCE(title,''), COALESCE(description,'') FROM assignments WHERE id = %s", (aid,))
    row = cur.fetchone()
    if not row:
        return ""
    return f"{row[0]}\n\n{row[1]}".strip()


def _embed_assignment(conn, embedder, aid: int) -> list[float]:
    """Compute and persist the embedding for one assignment; return the vector."""
    from embeddings import to_pgvector
    with conn.cursor() as cur:
        text = _assignment_text(cur, aid)
        vec = embedder.embed_one(text)
        cur.execute("UPDATE assignments SET embedding = %s::vector WHERE id = %s",
                    (to_pgvector(vec), aid))
    conn.commit()
    return vec


def cmd_embed(conn, args) -> None:
    """Backfill assignments.embedding for one (--key) or all (--all) assignments."""
    from embeddings import build_embedder_from_args
    embedder = build_embedder_from_args(args)
    with conn.cursor() as cur:
        if args.all:
            cur.execute("SELECT id, key FROM assignments ORDER BY created_at")
            targets = cur.fetchall()
        else:
            aid = _assignment_id(cur, args.key)
            if aid is None:
                print(f"assignment not found: {args.key}")
                return
            targets = [(aid, args.key)]
    for aid, key in targets:
        _embed_assignment(conn, embedder, aid)
        print(f"embedded {key} (provider={embedder.provider}, dim={embedder.dim})")
    print(f"done: {len(targets)} assignment(s) embedded")


def cmd_similar(conn, args) -> None:
    """Nearest assignments to --key by cosine distance over embeddings.

    If the query assignment (or others) lack an embedding, compute it on the fly
    with the chosen provider so the command works even before a backfill.
    """
    from embeddings import build_embedder_from_args, to_pgvector
    embedder = build_embedder_from_args(args)
    with conn.cursor() as cur:
        aid = _assignment_id(cur, args.key)
        if aid is None:
            print(f"assignment not found: {args.key}")
            return
        # Ensure every assignment has an embedding (auto-backfill missing ones).
        cur.execute("SELECT id FROM assignments WHERE embedding IS NULL")
        missing = [r[0] for r in cur.fetchall()]
    for mid in missing:
        _embed_assignment(conn, embedder, mid)

    with conn.cursor() as cur:
        cur.execute("SELECT embedding FROM assignments WHERE id = %s", (aid,))
        qvec = cur.fetchone()[0]
        cur.execute(
            """
            SELECT key, title, 1 - (embedding <=> %s::vector) AS similarity
            FROM assignments
            WHERE id != %s AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (qvec, aid, qvec, args.n),
        )
        rows = cur.fetchall()
    print(f"=== assignments most similar to '{args.key}' (provider={embedder.provider}) ===")
    if not rows:
        print("(no other embedded assignments to compare)")
        return
    for key, title, sim in rows:
        print(f"{sim:>6.3f}  {key:<16} {title or ''}")


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


def _add_embed_args(sp) -> None:
    """Shared embedding-provider flags for embed/similar."""
    sp.add_argument("--embed-provider", choices=["hash", "openai"], default="hash",
                    help="Embedding source: 'hash' (offline, default) or 'openai' "
                         "(OpenAI-compatible /embeddings)")
    sp.add_argument("--embed-model", default=None, help="Embedding model (openai provider)")
    sp.add_argument("--embed-dim", type=int, default=1024,
                    help="Embedding dimensionality (must match vector(1024) column; default 1024)")
    sp.add_argument("--api-key", default=None, help="API key for openai embeddings (or LLM_API_KEY)")
    sp.add_argument("--base-url", default=None, help="Base URL for openai-compatible embeddings")


def main(argv=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
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

    sp = sub.add_parser("embed", help="Backfill assignment embeddings (pgvector).")
    g = sp.add_mutually_exclusive_group(required=True)
    g.add_argument("--key", help="Embed a single assignment")
    g.add_argument("--all", action="store_true", help="Embed all assignments")
    _add_embed_args(sp)

    sp = sub.add_parser("similar", help="Find assignments semantically similar to --key.")
    sp.add_argument("--key", required=True)
    sp.add_argument("--n", type=int, default=5)
    _add_embed_args(sp)

    args = p.parse_args(argv)

    conn = db_common.connect()
    {"list": cmd_list, "stats": cmd_stats, "errors": cmd_errors, "top": cmd_top,
     "embed": cmd_embed, "similar": cmd_similar}[args.cmd](conn, args)
    conn.close()


if __name__ == "__main__":
    main()
