import json
from pathlib import Path
import streamlit as st

from jobmatch.data import load_jobs
from jobmatch.embedders import get_embedder
from jobmatch.llm import GeminiClient, LLMError
from jobmatch.memory import PreferenceStore
from jobmatch.recommend import recommend
from jobmatch.search import JobSearch

st.set_page_config(page_title="JobMatch — AI Job Recommender", layout="wide")
st.title("JobMatch — Resume-to-Role Recommender")
st.caption(
    "Hybrid retrieval (BM25 + embeddings) with Gemini structured parsing & reranking "
    "(automatic fallback when LLM is busy)."
)

DEFAULT_RESUME = """SUMMARY
Third-year computer science student interested in backend development and applied AI.

SKILLS
Python, SQL, FastAPI, Flask, REST APIs, Git, Docker, scikit-learn, PostgreSQL

PROJECTS
Campus Event Finder: Built a Python/Flask web app with search and filtering.
"""


@st.cache_resource
def get_engine():
    jobs = load_jobs(Path("data/jobs_sample.csv"))
    embedder = get_embedder("st")
    return JobSearch(jobs, embedder)


engine = get_engine()
store = PreferenceStore(Path("prefs.db"))

with st.sidebar:
    st.header("Settings")
    k = st.slider("Top K recommendations", min_value=1, max_value=10, value=5)
    use_rerank = st.checkbox("Use Gemini reranking", value=True)
    model_name = st.text_input("Gemini model", value="gemini-3.8-flash")

    eval_path = Path("results/eval_st.json")
    if eval_path.exists():
        st.divider()
        st.subheader("Offline Eval (k=5)")
        eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
        for row in eval_data.get("results", []):
            st.write(
                f"**{row['mode']}**: Recall={row['recall@5']:.3f} | "
                f"NDCG={row['ndcg@5']:.3f} | MRR={row['mrr']:.3f}"
            )

resume_text = st.text_area(
    "Paste your resume or target role summary", value=DEFAULT_RESUME, height=220
)

if st.button("Find Matching Jobs", type="primary"):
    if not resume_text.strip():
        st.warning("Please enter some resume or role text first.")
    else:
        llm = None
        if use_rerank:
            try:
                llm = GeminiClient(model=model_name.strip() or None)
            except LLMError as exc:
                st.info(f"LLM unavailable ({exc}); showing retrieval results.")

        with st.spinner("Searching and ranking matching roles..."):
            result = recommend(
                resume_text=resume_text,
                user_id="demo_user",
                engine=engine,
                llm=llm,
                store=store,
                k=k,
                use_rerank=use_rerank,
            )

        for note in result.notes:
            st.warning(note)

        st.caption(f"Search query: {result.query}")
        for idx, rec in enumerate(result.recommendations, start=1):
            job = rec.job
            score_label = (
                f"Fit {rec.llm_score}/3" if rec.llm_score is not None else "Retrieval order"
            )
            st.markdown(
                f"### {idx}. [{job.id}] {job.title} — {job.company} ({job.location}) · *{score_label}*"
            )
            if rec.reason:
                st.write(rec.reason)