# jobmatch

A resume-to-role job search and recommendation engine that compares keyword, dense vector, and hybrid retrieval, cleans noisy listings, personalizes results with saved preferences, and reranks candidates using Gemini with resilient fallbacks.

**Live Demo:** `<your-streamlit-app-url>` · **Repository:** `https://github.com/Gargipal12/jobmatch`

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
streamlit run app.py                                    # launch local web demo
```

`--embedder st` uses a real neural model (`all-MiniLM-L6-v2`, downloaded on first run, no API key).
`--embedder hash` is an offline fallback that only matches spelling, not meaning. It exists to test the pipeline. Do not report its numbers as semantic search results.

## Layout

```text
app.py                  Streamlit web interface for resume matching and eval summary
jobmatch/bm25.py        keyword ranking, written from scratch
jobmatch/embedders.py   text -> vectors (neural model or offline fallback)
jobmatch/search.py      bm25 / vector / hybrid search
jobmatch/evaluate.py    recall@k, NDCG@k, MRR + failure analysis (--clean to drop dupes/ghosts)
jobmatch/dedup.py       near-duplicate detection (word-shingle Jaccard + union-find)
jobmatch/ghost.py       explainable ghost/scam listing score
jobmatch/quality.py     precision/recall of dedup and ghost detection vs hand labels
jobmatch/memory.py      SQLite-backed user preferences, hidden jobs, and preference updates
jobmatch/resume.py      structured resume parsing and search query construction
jobmatch/rerank.py      LLM candidate reranking with fit scores and explanations
jobmatch/recommend.py   end-to-end recommendation pipeline with fallbacks
jobmatch/llm.py         Gemini client with JSON validation, retries, and error handling
data/jobs_sample.csv    40 synthetic jobs (with near-duplicates and ghost-like postings)
data/eval_queries.json  10 queries with 50 graded query-job pairs (2 = strong, 1 = partial)
data/quality_labels.json  hand labels for duplicates and ghosts
results/eval_st.json    saved evaluation output for --embedder st
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

Pipeline: resume -> structured profile (LLM) -> hybrid search with dupes/ghosts removed -> your saved preferences -> LLM rerank. Each LLM step validates its JSON, retries on bad output or transient errors, and falls back gracefully (see `jobmatch/llm.py`, `jobmatch/recommend.py`). Preferences live in SQLite (`prefs.db`).

## Results

Evaluated on 10 queries (50 labeled query-job pairs) with `k=5`, `clean=False`, and `--embedder st` (saved to `results/eval_st.json`):

| mode | recall@5 | NDCG@5 | MRR |
|---|---|---|---|
| bm25 | 0.560 | 0.806 | 1.000 |
| vector | 0.620 | 0.825 | 1.000 |
| hybrid | 0.600 | 0.834 | 1.000 |

## Failure analysis

- **Computer vision (`q6`, NDCG `0.61` across all modes):** Retrievers surface `j25` cleanly, but miss broader partial matches (`j01`, `j02`, `j09`, `j16`) when vision isn't explicitly prominent in the job snippet.
- **React frontend (`q10`, NDCG `0.73` across all modes):** All modes match `j05` and `j18`, while `vector` and `hybrid` pull backend/Python roles (`j04`, `j03`, `j34`) and miss adjacent UI/mobile roles (`j06`, `j15`, `j16`).
- **Mode-specific misses:** `bm25` struggles on `q5` (`recommendation systems personalisation`, NDCG `0.76`) when wording diverges; `vector` misses `j09` on `q9` (`retrieval augmented generation vector database`, NDCG `0.77`); `hybrid` scores `0.79` NDCG on `q3` (`semantic search and ranking engineer`) while missing `j07` and `j08`.

## Next steps

- [x] Expand evaluation set to 50 labeled query-job pairs and benchmark `bm25`, `vector`, and `hybrid`
- [x] Add structured resume parsing, SQLite preference memory, and Gemini reranking with retry/fallback logic
- [x] Build a Streamlit web demo (`app.py`) and deploy it
- [ ] Replace the sample CSV with a larger real-world jobs dataset and re-verify dedup/ghost thresholds
- [ ] Add MinHash/LSH indexing if the job catalog grows large, and explore a Telegram bot interface
