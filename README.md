# jobmatch

A small job search engine that compares three ways of finding relevant jobs, and measures which one works best.

| Mode | How it finds jobs | Good at | Weak at |
|---|---|---|---|
| `bm25` | Counts matching words, weighted by how rare each word is | Exact terms like "FastAPI", "Kubernetes" | Different words, same meaning |
| `vector` | Embeds query and jobs, ranks by cosine similarity | Meaning, e.g. "ML engineer" ~ "machine learning developer" | Exact tool names, rare terms |
| `hybrid` | Fuses both rankings with Reciprocal Rank Fusion | Usually the best of both | Needs both parts to work |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest                                                  # sanity tests
python -m jobmatch.cli "python backend fastapi" --mode hybrid
python -m jobmatch.evaluate --embedder st               # the comparison table
```

`--embedder st` uses a real neural model (all-MiniLM-L6-v2, downloaded on first run, no API key).
`--embedder hash` is an offline fallback that only matches spelling, not meaning. It exists to test the pipeline. Do not report its numbers as semantic search results.

## Layout

```
jobmatch/bm25.py        keyword ranking, written from scratch
jobmatch/embedders.py   text -> vectors (neural model or offline fallback)
jobmatch/search.py      bm25 / vector / hybrid search
jobmatch/evaluate.py    recall@k, NDCG@k, MRR + failure analysis (--clean to drop dupes/ghosts)
jobmatch/dedup.py       near-duplicate detection (word-shingle Jaccard + union-find)
jobmatch/ghost.py       explainable ghost/scam listing score
jobmatch/quality.py     precision/recall of dedup and ghost detection vs hand labels
data/jobs_sample.csv    40 synthetic jobs (with near-duplicates and ghost-like postings)
data/eval_queries.json  10 starter queries with graded labels
data/quality_labels.json  hand labels for duplicates and ghosts
```

## Cleaning the index

```bash
python -m jobmatch.quality                      # how good are the detectors?
python -m jobmatch.evaluate --embedder st --clean   # search quality with dupes/ghosts removed
python -m jobmatch.cli "python backend" --clean
```

- **Duplicates:** two listings are duplicates if their word 3-gram sets overlap by 60% or more. We keep one copy, preferring a direct employer over an agency, then the freshest.
- **Ghosts:** each rule (stale posting, vague "talent pool" language, scam phrases, vague location) adds to a score and records a reason. Score 0.5 or more is flagged.
- **Caveat:** the thresholds were tuned on the 40-job sample, so perfect scores there prove little. Measure on a hand-checked sample of real data and report those numbers.

## Recommend jobs for a resume (needs a Gemini API key)

```bash
# PowerShell:  $env:GEMINI_API_KEY="your-key"      Mac/Linux:  export GEMINI_API_KEY="your-key"
python -m jobmatch.recommend --resume my_resume.txt
python -m jobmatch.recommend --resume my_resume.txt --message "no sales jobs, remote only"
python -m jobmatch.recommend --resume my_resume.txt --hide j08
```

Pipeline: resume -> structured profile (LLM) -> hybrid search with dupes/ghosts removed -> your saved preferences -> LLM rerank. Each LLM step validates its JSON, retries on bad output, and falls back gracefully (see `jobmatch/llm.py`, `jobmatch/recommend.py`). Preferences live in SQLite (`prefs.db`).

## Results

Run `python -m jobmatch.evaluate --embedder st` and paste the table here.

| mode | recall@5 | NDCG@5 | MRR |
|---|---|---|---|
| bm25 | | | |
| vector | | | |
| hybrid | | | |

## Failure analysis

Write 3-5 lines on queries where each mode failed and why. The evaluator prints the worst cases for you.

## Next steps

- Replace the sample data with a real jobs dataset and hand-label about 50 query-job pairs
- Re-check dedup and ghost detection on real data (MinHash/LSH if the index gets large)
- Add an LLM reranker and résumé parsing with structured output
- Add per-user memory of preferences
- Dockerize, deploy, put it behind a Telegram bot
