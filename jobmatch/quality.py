"""Measure the dedup and ghost detectors against hand labels.

    python -m jobmatch.quality

Precision = of the things we flagged, how many were truly bad?
Recall    = of the truly bad things, how many did we flag?
Both matter: low precision hides real jobs from users, low recall lets junk through.
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

from .data import load_jobs
from .dedup import find_duplicate_groups
from .ghost import score_job

ROOT = Path(__file__).resolve().parent.parent


def prf(predicted: set, truth: set) -> tuple[float, float]:
    tp = len(predicted & truth)
    precision = tp / len(predicted) if predicted else 1.0
    recall = tp / len(truth) if truth else 1.0
    return precision, recall


def duplicate_pairs(groups) -> set[frozenset]:
    return {frozenset(pair) for g in groups for pair in combinations(sorted(g), 2)}


def run(jobs, labels, threshold: float = 0.6) -> dict:
    ghost_pred = {j.id for j in jobs if score_job(j).flagged}
    ghost_truth = set(labels["ghost"])
    pred_pairs = duplicate_pairs([[j.id for j in g] for g in find_duplicate_groups(jobs, threshold)])
    true_pairs = duplicate_pairs(labels["duplicate_groups"])
    gp, gr = prf(ghost_pred, ghost_truth)
    dp, dr = prf(pred_pairs, true_pairs)
    return {
        "ghost": {"precision": gp, "recall": gr, "flagged": sorted(ghost_pred),
                  "false_positives": sorted(ghost_pred - ghost_truth),
                  "missed": sorted(ghost_truth - ghost_pred)},
        "duplicates": {"precision": dp, "recall": dr,
                       "false_pairs": sorted(sorted(p) for p in pred_pairs - true_pairs),
                       "missed_pairs": sorted(sorted(p) for p in true_pairs - pred_pairs)},
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate dedup and ghost detection.")
    p.add_argument("--jobs", default=str(ROOT / "data" / "jobs_sample.csv"))
    p.add_argument("--labels", default=str(ROOT / "data" / "quality_labels.json"))
    p.add_argument("--threshold", type=float, default=0.6)
    args = p.parse_args()

    jobs = load_jobs(args.jobs)
    labels = json.loads(Path(args.labels).read_text())
    r = run(jobs, labels, args.threshold)

    g, d = r["ghost"], r["duplicates"]
    print(f"Ghost detection : precision={g['precision']:.2f} recall={g['recall']:.2f}  flagged={g['flagged']}")
    print(f"  false positives: {g['false_positives']}   missed: {g['missed']}")
    print(f"Duplicate pairs : precision={d['precision']:.2f} recall={d['recall']:.2f}")
    print(f"  false pairs: {d['false_pairs']}   missed pairs: {d['missed_pairs']}")
    print("\nWhy each flagged job was flagged:")
    for j in jobs:
        res = score_job(j)
        if res.flagged:
            print(f"  [{j.id}] {j.title} (score {res.score:.2f}): " + "; ".join(res.reasons))


if __name__ == "__main__":
    main()
