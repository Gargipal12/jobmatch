"""Search over jobs: keyword (BM25), vector (embeddings) and hybrid (both).

With clean=True, results skip duplicate copies (keeping one canonical listing)
and listings flagged as likely ghosts.
"""
from dataclasses import dataclass

import numpy as np

from .bm25 import BM25
from .data import Job
from .dedup import canonical_map
from .ghost import GhostResult, score_job

MODES = ("bm25", "vector", "hybrid")


@dataclass
class Hit:
    job: Job
    score: float
    rank: int  # 1-based


class JobSearch:
    def __init__(self, jobs: list[Job], embedder, rrf_k: int = 60, dup_threshold: float = 0.6):
        self.jobs = jobs
        self.rrf_k = rrf_k
        texts = [j.text for j in jobs]
        self.embedder = embedder
        self.bm25 = BM25(texts)
        # One vector per job, computed once up front.
        self.doc_vecs = embedder.encode(texts)

        # Index-time quality signals.
        self.canonical: dict[str, str] = canonical_map(jobs, dup_threshold)
        self.ghost: dict[str, GhostResult] = {j.id: score_job(j) for j in jobs}
        self.excluded: set[str] = {
            j.id
            for j in jobs
            if self.canonical[j.id] != j.id or self.ghost[j.id].flagged
        }

    # Each helper returns a ranked list of (job_index, score), best first.
    def _bm25_ranking(self, query: str) -> list[tuple[int, float]]:
        scores = self.bm25.scores(query)
        order = np.argsort(-scores)
        # A job that shares no words with the query is not a keyword match.
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def _vector_ranking(self, query: str) -> list[tuple[int, float]]:
        q = self.embedder.encode([query])[0]
        sims = self.doc_vecs @ q  # cosine similarity (vectors are normalised)
        order = np.argsort(-sims)
        return [(int(i), float(sims[i])) for i in order]

    def _hybrid_ranking(self, query: str) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion: score = sum over rankers of 1 / (k + rank).

        It combines rankings by position, so we never have to compare BM25
        scores with cosine similarities (they live on different scales).
        """
        fused: dict[int, float] = {}
        for ranking in (self._bm25_ranking(query), self._vector_ranking(query)):
            for rank, (idx, _) in enumerate(ranking, start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)
        return sorted(fused.items(), key=lambda kv: -kv[1])

    def search(self, query: str, mode: str = "hybrid", k: int = 5, clean: bool = False) -> list[Hit]:
        if mode == "bm25":
            ranking = self._bm25_ranking(query)
        elif mode == "vector":
            ranking = self._vector_ranking(query)
        elif mode == "hybrid":
            ranking = self._hybrid_ranking(query)
        else:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if clean:
            ranking = [(i, s) for i, s in ranking if self.jobs[i].id not in self.excluded]
        return [
            Hit(job=self.jobs[idx], score=score, rank=r)
            for r, (idx, score) in enumerate(ranking[:k], start=1)
        ]
