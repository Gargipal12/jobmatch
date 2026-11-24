from pathlib import Path

from jobmatch.data import Job, load_jobs
from jobmatch.dedup import canonical_map, choose_canonical, find_duplicate_groups, jaccard, shingles
from jobmatch.embedders import HashingEmbedder
from jobmatch.ghost import score_job
from jobmatch.search import JobSearch

DATA = Path(__file__).resolve().parent.parent / "data"


def _job(id, company="Co", days=5, loc="Remote", desc="Build APIs in Python.", title="Dev"):
    return Job(id, title, company, loc, days, desc)


def test_jaccard_identical_and_disjoint():
    a = shingles("build apis in python with fastapi today")
    assert jaccard(a, a) == 1.0
    assert jaccard(a, shingles("completely different words go right here now")) == 0.0


def test_one_changed_word_is_still_a_duplicate():
    jobs = [
        _job("a", desc="Build and deploy models for our platform. Work with PyTorch, pipelines and tests to improve ranking quality."),
        _job("b", desc="Build and deploy models for our platform. Work with PyTorch, pipelines and experiments to improve ranking quality."),
        _job("c", desc="Plan social media campaigns and report on engagement for the brand."),
    ]
    groups = find_duplicate_groups(jobs)
    assert [sorted(j.id for j in g) for g in groups] == [["a", "b"]]


def test_canonical_prefers_direct_employer_then_fresh():
    agency = _job("agency", company="Acme via StaffPro", days=2)
    direct_old = _job("direct", company="Acme", days=30)
    assert choose_canonical([agency, direct_old]).id == "direct"
    assert choose_canonical([_job("x", days=9), _job("y", days=3)]).id == "y"


def test_sample_data_canonicals():
    cm = canonical_map(load_jobs(DATA / "jobs_sample.csv"))
    assert cm["j02"] == "j01" and cm["j34"] == "j03"
    assert cm["j05"] == "j05"


def test_ghost_flags_exactly_the_known_bad_listings():
    flagged = {j.id for j in load_jobs(DATA / "jobs_sample.csv") if score_job(j).flagged}
    assert flagged == {"j30", "j31", "j32"}


def test_ghost_reasons_are_explainable():
    res = score_job(_job("s", days=200, desc="Always hiring. Join our talent pool."))
    assert res.flagged and any("stale" in r for r in res.reasons)


def test_clean_search_hides_duplicates_and_ghosts():
    engine = JobSearch(load_jobs(DATA / "jobs_sample.csv"), HashingEmbedder())
    plain = {h.job.id for h in engine.search("python backend fastapi", mode="hybrid", k=40)}
    clean = {h.job.id for h in engine.search("python backend fastapi", mode="hybrid", k=40, clean=True)}
    assert {"j34", "j30", "j31", "j32"} <= plain
    assert not ({"j34", "j02", "j30", "j31", "j32"} & clean)
