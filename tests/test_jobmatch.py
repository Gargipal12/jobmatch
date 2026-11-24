from pathlib import Path

from jobmatch.bm25 import BM25
from jobmatch.data import load_eval, load_jobs
from jobmatch.embedders import HashingEmbedder
from jobmatch.evaluate import mrr, ndcg_at_k, recall_at_k
from jobmatch.search import JobSearch

DATA = Path(__file__).resolve().parent.parent / "data"


def test_bm25_puts_exact_match_first():
    docs = ["python backend developer", "react frontend developer", "sales manager"]
    scores = BM25(docs).scores("python backend")
    assert scores.argmax() == 0
    assert scores[2] == 0


def test_metrics_perfect_and_empty():
    rel = {"a": 2, "b": 1}
    assert recall_at_k(["a", "b", "x"], rel, 3) == 1.0
    assert ndcg_at_k(["a", "b", "x"], rel, 3) == 1.0
    assert recall_at_k(["x", "y", "z"], rel, 3) == 0.0
    assert mrr(["x", "a"], rel, 5) == 0.5


def test_ndcg_rewards_better_order():
    rel = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b"], rel, 2) > ndcg_at_k(["b", "a"], rel, 2)


def test_embedder_vectors_are_unit_length():
    vecs = HashingEmbedder().encode(["python developer", "sales"])
    assert vecs.shape[0] == 2
    assert abs((vecs[0] ** 2).sum() - 1.0) < 1e-5


def test_all_modes_return_k_results_on_sample_data():
    jobs = load_jobs(DATA / "jobs_sample.csv")
    engine = JobSearch(jobs, HashingEmbedder())
    for mode in ("vector", "hybrid"):
        assert len(engine.search("python fastapi backend", mode=mode, k=5)) == 5
    assert len(engine.search("python fastapi backend", mode="bm25", k=5)) >= 1


def test_every_labeled_job_id_exists():
    ids = {j.id for j in load_jobs(DATA / "jobs_sample.csv")}
    for q in load_eval(DATA / "eval_queries.json"):
        assert set(q["relevant"]) <= ids
