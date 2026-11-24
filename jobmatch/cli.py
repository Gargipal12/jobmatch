"""Try a query from the terminal:

    python -m jobmatch.cli "python backend fastapi" --mode hybrid --embedder st
"""
import argparse
from pathlib import Path

from .data import load_jobs
from .embedders import get_embedder
from .search import MODES, JobSearch

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description="Search the jobs index.")
    p.add_argument("query")
    p.add_argument("--mode", default="hybrid", choices=MODES)
    p.add_argument("--embedder", default="st", choices=["st", "hash"])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--clean", action="store_true", help="drop duplicates and ghost listings")
    p.add_argument("--jobs", default=str(ROOT / "data" / "jobs_sample.csv"))
    args = p.parse_args()

    engine = JobSearch(load_jobs(args.jobs), get_embedder(args.embedder))
    for hit in engine.search(args.query, mode=args.mode, k=args.k, clean=args.clean):
        j = hit.job
        print(f"{hit.rank}. [{j.id}] {j.title} - {j.company} ({j.location}) score={hit.score:.3f}")


if __name__ == "__main__":
    main()
