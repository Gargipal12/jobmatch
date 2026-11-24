"""Ghost / fake listing detection with simple, explainable rules.

A "ghost" posting is one that is not a real, open role: stale listings nobody
removed, "talent pool" collection posts, or outright scams. Each rule adds to a
score and records a human-readable reason, so every flag can be explained.

These rules are a starting point written by hand. Their precision on real
data is unknown until you check a sample yourself (see quality.py).
"""
from dataclasses import dataclass

from .data import Job

STALE_DAYS = 90
VERY_STALE_DAYS = 150
FLAG_THRESHOLD = 0.5

VAGUE_PHRASES = (
    "multiple openings",
    "multiple positions",
    "talent pool",
    "always hiring",
    "competitive salary",
    "freshers and experienced",
)
SCAM_PHRASES = ("registration fee", "no experience needed", "per day", "!!!")
VAGUE_LOCATIONS = ("anywhere", "multiple")


@dataclass
class GhostResult:
    score: float
    reasons: list[str]

    @property
    def flagged(self) -> bool:
        return self.score >= FLAG_THRESHOLD


def score_job(job: Job) -> GhostResult:
    text = f"{job.title} {job.description}".lower()
    score, reasons = 0.0, []

    if job.posted_days_ago > VERY_STALE_DAYS:
        score += 0.5
        reasons.append(f"posted {job.posted_days_ago} days ago (very stale)")
    elif job.posted_days_ago > STALE_DAYS:
        score += 0.35
        reasons.append(f"posted {job.posted_days_ago} days ago (stale)")

    vague = [p for p in VAGUE_PHRASES if p in text]
    if vague:
        score += min(0.2 * len(vague), 0.5)
        reasons.append(f"vague hiring language: {', '.join(vague)}")

    scam = [p for p in SCAM_PHRASES if p in text]
    if scam:
        score += 0.6
        reasons.append(f"scam-style language: {', '.join(scam)}")

    if any(v in job.location.lower() for v in VAGUE_LOCATIONS):
        score += 0.15
        reasons.append(f"vague location: {job.location!r}")

    return GhostResult(score=min(score, 1.0), reasons=reasons)
