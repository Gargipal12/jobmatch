"""The full pipeline: résumé -> profile -> search -> filter -> LLM rerank.

    python -m jobmatch.recommend --resume resume.txt
    python -m jobmatch.recommend --resume resume.txt --message "no sales jobs, remote only"
    python -m jobmatch.recommend --resume resume.txt --hide j08

Every LLM step can fail, so each one has a fallback and adds a note explaining
what happened. The user always gets results, just less personalised.
"""
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .data import load_jobs
from .embedders import get_embedder
from .llm import LLM, GeminiClient, LLMError
from .memory import PreferenceStore, allowed, update_from_message
from .rerank import Recommendation, rerank
from .resume import ResumeProfile, parse_resume, profile_to_query
from .search import JobSearch

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RecommendationResult:
    recommendations: list[Recommendation]
    query: str
    profile: Optional[ResumeProfile] = None
    notes: list[str] = field(default_factory=list)


def recommend(
    resume_text: str,
    user_id: str,
    engine: JobSearch,
    llm: LLM,
    store: PreferenceStore,
    k: int = 5,
    n_candidates: int = 20,
    mode: str = "hybrid",
    use_rerank: bool = True,
) -> RecommendationResult:
    notes: list[str] = []

    # 1. Résumé -> structured profile (fallback: search with the raw résumé text).
    profile: Optional[ResumeProfile] = None
    query = ""
    try:
        profile = parse_resume(resume_text, llm)
        query = profile_to_query(profile)
    except LLMError as e:
        notes.append(f"resume parsing failed ({e}); searched with raw resume text")
    if not query.strip():
        query = resume_text[:500]

    # 2. Retrieve (duplicates and ghosts already removed), then apply the user's saved preferences.
    prefs = store.get(user_id)
    hits = engine.search(query, mode=mode, k=n_candidates * 3, clean=True)
    jobs = [h.job for h in hits if allowed(h.job, prefs)][:n_candidates]

    # 3. LLM rerank (fallback: keep retrieval order).
    recs: list[Recommendation]
    if use_rerank and jobs:
        candidate = profile.model_dump_json() if profile else resume_text[:1000]
        try:
            recs = rerank(candidate, prefs.summary(), jobs, llm, k)
        except LLMError as e:
            notes.append(f"reranking failed ({e}); showing retrieval order")
            recs = [Recommendation(j, r, None, "") for r, j in enumerate(jobs[:k], start=1)]
    else:
        recs = [Recommendation(j, r, None, "") for r, j in enumerate(jobs[:k], start=1)]

    return RecommendationResult(recommendations=recs, query=query, profile=profile, notes=notes)


def main() -> None:
    p = argparse.ArgumentParser(description="Recommend jobs for a resume.")
    p.add_argument("--resume", required=True, help="path to a text file with your resume")
    p.add_argument("--user", default="me")
    p.add_argument("--message", help='tell it preferences, e.g. "no sales jobs, remote only"')
    p.add_argument("--hide", nargs="*", default=[], help="job ids to never show this user again")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--candidates", type=int, default=20)
    p.add_argument("--embedder", default="st", choices=["st", "hash"])
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--model", help="Gemini model name (default: gemini-2.5-flash or $GEMINI_MODEL)")
    p.add_argument("--jobs", default=str(ROOT / "data" / "jobs_sample.csv"))
    p.add_argument("--db", default="prefs.db")
    args = p.parse_args()

    try:
        llm = GeminiClient(model=args.model)
    except LLMError as e:
        raise SystemExit(str(e))

    engine = JobSearch(load_jobs(args.jobs), get_embedder(args.embedder))
    store = PreferenceStore(args.db)
    for job_id in args.hide:
        store.hide_job(args.user, job_id)
    if args.message:
        try:
            update_from_message(args.user, args.message, llm, store)
        except LLMError as e:
            print(f"Could not read preferences from message: {e}")
    print(f"Saved preferences for '{args.user}': {store.get(args.user).summary()}\n")

    resume_text = Path(args.resume).read_text(encoding="utf-8")
    result = recommend(
        resume_text, args.user, engine, llm, store,
        k=args.k, n_candidates=args.candidates, use_rerank=not args.no_rerank,
    )
    for note in result.notes:
        print(f"NOTE: {note}")
    print(f"Search query: {result.query}\n")
    for i, r in enumerate(result.recommendations, start=1):
        j = r.job
        fit = f"fit {r.llm_score}/3" if r.llm_score is not None else "not reranked"
        print(f"{i}. [{j.id}] {j.title} - {j.company} ({j.location})  {fit}")
        if r.reason:
            print(f"     {r.reason}")


if __name__ == "__main__":
    main()
