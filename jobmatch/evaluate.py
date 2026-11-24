"""Eval harness: how good is each search mode, measured against labeled queries?

Metrics (all computed on the top-k results, k=5 by default):
- recall@k : of the jobs a human marked relevant, what fraction showed up?
- NDCG@k   : like recall, but rewards putting the best matches higher up and
             uses the grades (2 = strong, 1 = partial).
- MRR      : 1 / rank of the first relevant result (how quickly you hit one).
"""
import argparse
import json
import math
from pathlib import Path

from .data import load_eval, load_jobs
from .embedders import get_embedder
from .search import MODES, JobSearch

ROOT = Path(__file__).resolve().parent.parent


def recall_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    relevant = {d for d, g in rel.items() if g > 0}
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def _dcg(grades: list[int]) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    gains = [rel.get(d, 0) for d in ranked[:k]]
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(gains) / idcg if idcg > 0 else 0.0


def mrr(ranked: list[str], rel: dict[str, int], k: int) -> float:
    for i, d in enumerate(ranked[:k]):
        if rel.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def collapse_labels(rel: dict[str, int], engine: JobSearch) -> dict[str, int]:
    """For clean search: a relevant duplicate counts as its canonical copy, and
    relevant ghosts are dropped (we do not want them returned)."""
    out: dict[str, int] = {}
    for job_id, grade in rel.items():
        canon = engine.canonical.get(job_id, job_id)
        if engine.ghost[canon].flagged:
            continue
        out[canon] = max(out.get(canon, 0), grade)
    return out


def evaluate(engine: JobSearch, queries: list[dict], k: int = 5, clean: bool = False) -> dict:
    results = {}
    for mode in MODES:
        per_query = []
        for q in queries:
            ranked = [h.job.id for h in engine.search(q["query"], mode=mode, k=k, clean=clean)]
            rel = collapse_labels(q["relevant"], engine) if clean else q["relevant"]
            per_query.append(
                {
                    "id": q["id"],
                    "query": q["query"],
                    "retrieved": ranked,
                    "missed": sorted(d for d, g in rel.items() if g > 0 and d not in ranked),
                    "recall": recall_at_k(ranked, rel, k),
                    "ndcg": ndcg_at_k(ranked, rel, k),
                    "mrr": mrr(ranked, rel, k),
                }
            )
        n = len(per_query)
        results[mode] = {
            "recall": sum(p["recall"] for p in per_query) / n,
            "ndcg": sum(p["ndcg"] for p in per_query) / n,
            "mrr": sum(p["mrr"] for p in per_query) / n,
            "per_query": per_query,
        }
    return results


def format_table(results: dict, k: int) -> str:
    lines = [
        f"| mode | recall@{k} | NDCG@{k} | MRR |",
        "|---|---|---|---|",
    ]
    for mode in MODES:
        r = results[mode]
        lines.append(f"| {mode} | {r['recall']:.3f} | {r['ndcg']:.3f} | {r['mrr']:.3f} |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare bm25 / vector / hybrid search.")
    p.add_argument("--jobs", default=str(ROOT / "data" / "jobs_sample.csv"))
    p.add_argument("--queries", default=str(ROOT / "data" / "eval_queries.json"))
    p.add_argument("--embedder", default="st", choices=["st", "hash"])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--clean", action="store_true", help="drop duplicates and ghost listings")
    p.add_argument("--worst", type=int, default=3, help="failure cases to print per mode")
    args = p.parse_args()

    engine = JobSearch(load_jobs(args.jobs), get_embedder(args.embedder))
    queries = load_eval(args.queries)
    results = evaluate(engine, queries, k=args.k, clean=args.clean)

    print(f"Embedder: {args.embedder}   Queries: {len(queries)}   k={args.k}   clean={args.clean}\n")
    print(format_table(results, args.k))

    print("\nWorst queries per mode (by NDCG) - read these, they are your failure analysis:")
    for mode in MODES:
        worst = sorted(results[mode]["per_query"], key=lambda r: r["ndcg"])[: args.worst]
        print(f"\n[{mode}]")
        for r in worst:
            print(f"  {r['id']} ndcg={r['ndcg']:.2f} '{r['query']}'")
            print(f"      got: {r['retrieved']}   missed relevant: {r['missed']}")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    tag = args.embedder + ("_clean" if args.clean else "")
    out = out_dir / f"eval_{tag}.json"
    out.write_text(json.dumps(results, indent=2))
    (out_dir / f"eval_{tag}.md").write_text(format_table(results, args.k) + "\n")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
