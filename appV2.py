import os

import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util


# ------------ 1. Load data and build indices (cached) ------------

@st.cache_resource
def load_corpus_and_indices(corpus_path="data/interim/corpus_v1.csv"):
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found at: {corpus_path}")

    df = pd.read_csv(corpus_path)

    # Basic safety checks
    if "abstract" not in df.columns:
        raise ValueError("Corpus must have an 'abstract' column.")
    if "title" not in df.columns:
        df["title"] = "Untitled"
    # Use 'region' instead of 'location'
    if "region" not in df.columns:
        df["region"] = "Unknown"
    if "year" not in df.columns:
        df["year"] = None

    df["abstract"] = df["abstract"].astype(str)
    df["title"] = df["title"].astype(str)
    df["region"] = df["region"].astype(str)

    # --- TF-IDF ---
    texts = df["abstract"].tolist()
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    # --- BM25 ---
    tokenized_corpus = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # --- Sentence Transformers ---
    model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = model.encode(texts, convert_to_tensor=True)

    return df, vectorizer, tfidf_matrix, bm25, model, doc_embeddings


# ------------ 2. Search helpers ------------

def simple_search(query, df, vectorizer, tfidf_matrix, max_results=20):
    query = query.strip()
    if not query:
        return []

    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()

    # Sort all docs by TF-IDF score
    indices = np.argsort(scores)[::-1]

    results = []
    for idx in indices[:max_results]:
        results.append({
            "index": int(idx),
            "title": df.iloc[idx]["title"],
            "abstract": df.iloc[idx]["abstract"],
            "region": df.iloc[idx].get("region", "Unknown"),
            "year": df.iloc[idx].get("year", None),
            "tfidf_score": float(scores[idx])
        })
    return results


def advanced_search(
    query,
    df,
    vectorizer,
    tfidf_matrix,
    bm25,
    semantic_model,
    doc_embeddings,
    tfidf_threshold=0.05,
    max_results=None
):
    query = query.strip()
    if not query:
        return []

    # TF-IDF
    q_vec = vectorizer.transform([query])
    tfidf_scores = cosine_similarity(q_vec, tfidf_matrix).flatten()

    # BM25
    q_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(q_tokens)

    # Semantic
    q_emb = semantic_model.encode(query, convert_to_tensor=True)
    semantic_scores_tensor = util.cos_sim(q_emb, doc_embeddings).squeeze(0)
    semantic_scores = semantic_scores_tensor.cpu().numpy()

    results = []
    for idx in range(len(df)):
        tfidf_score = float(tfidf_scores[idx])
        if tfidf_score < tfidf_threshold:
            continue

        results.append({
            "index": int(idx),
            "title": df.iloc[idx]["title"],
            "abstract": df.iloc[idx]["abstract"],
            "region": df.iloc[idx].get("region", "Unknown"),
            "year": df.iloc[idx].get("year", None),
            "tfidf_score": tfidf_score,
            "bm25_score": float(bm25_scores[idx]),
            "semantic_score": float(semantic_scores[idx])
        })

    # Sort by TF-IDF
    results.sort(key=lambda x: x["tfidf_score"], reverse=True)

    if max_results is not None:
        results = results[:max_results]

    return results


# ------------ 3. Visualization helpers ------------

def make_country_map(df, results):
    # Build DataFrame of matched documents and their regions
    if not results:
        return None

    indices = [r["index"] for r in results]
    subset = df.iloc[indices].copy()

    if "region" not in subset.columns:
        return None

    # Count per region
    counts = subset.groupby("region").size().reset_index(name="count")

    # Filter unknowns
    counts = counts[counts["region"].str.lower() != "unknown"]
    if counts.empty:
        return None

    # We treat 'region' as country names for mapping
    fig = px.choropleth(
        counts,
        locations="region",
        locationmode="country names",
        color="count",
        color_continuous_scale="Blues",
        title="Research density by region (for this query)"
    )
    return fig


def make_timeline(df, results):
    if not results:
        return None

    indices = [r["index"] for r in results]
    subset = df.iloc[indices].copy()

    if "year" not in subset.columns:
        return None

    # Drop missing years
    subset = subset.dropna(subset=["year"])
    if subset.empty:
        return None

    subset["year"] = subset["year"].astype(int)
    counts = subset.groupby("year").size().reset_index(name="count")

    fig = px.line(
        counts,
        x="year",
        y="count",
        markers=True,
        title="Matched documents over time"
    )
    return fig


# ------------ 4. Streamlit App ------------

def init_session_state():
    if "query_log" not in st.session_state:
        st.session_state["query_log"] = []


def main():
    st.set_page_config(
        page_title="ArchLearn Mini IR Lab",
        layout="wide"
    )
    init_session_state()

    st.title("ArchLearn Mini – IR Lab for Creative Learning Research")

    with st.sidebar:
        st.header("Search Settings")

        mode = st.radio(
            "Mode",
            options=["Simple Search", "Advanced Lab"],
            index=0
        )

        max_results = st.slider(
            "Max results to display",
            min_value=10,
            max_value=100,
            value=20,
            step=10
        )

        if mode == "Advanced Lab":
            tfidf_threshold = st.slider(
                "TF-IDF threshold (filter)",
                min_value=0.0,
                max_value=0.2,
                value=0.05,
                step=0.01
            )
        else:
            tfidf_threshold = 0.0  # Not used in simple mode

    # Load data and indices
    df, vectorizer, tfidf_matrix, bm25, semantic_model, doc_embeddings = load_corpus_and_indices()

    # Main search UI
    query = st.text_input("Enter your search query")
    run_button = st.button("Search")

    results = []
    if run_button and query.strip():
        if mode == "Simple Search":
            results = simple_search(
                query,
                df,
                vectorizer,
                tfidf_matrix,
                max_results=max_results
            )
        else:
            results = advanced_search(
                query,
                df,
                vectorizer,
                tfidf_matrix,
                bm25,
                semantic_model,
                doc_embeddings,
                tfidf_threshold=tfidf_threshold,
                max_results=max_results
            )

        # Log query
        st.session_state["query_log"].append({
            "query": query,
            "mode": mode,
            "n_results": len(results)
        })

    # Layout for results + analytics
    col_results, col_analytics = st.columns([2, 1])

    with col_results:
        st.subheader("Search Results")

        if not results and run_button:
            st.info("No results found for this query.")
        elif results:
            st.write(f"Showing {len(results)} result(s).")

            for r in results:
                with st.expander(f"{r['title']}"):
                    meta = []
                    if r.get("year") is not None and not pd.isna(r.get("year")):
                        meta.append(f"Year: {int(r['year'])}")
                    if r.get("region"):
                        meta.append(f"Region: {r['region']}")
                    if meta:
                        st.write(" | ".join(meta))

                    # Scores
                    if mode == "Simple Search":
                        st.write(f"**TF-IDF score:** {r['tfidf_score']:.4f}")
                    else:
                        st.write(
                            f"**TF-IDF:** {r['tfidf_score']:.4f} | "
                            f"**BM25:** {r['bm25_score']:.4f} | "
                            f"**Semantic:** {r['semantic_score']:.4f}"
                        )

                    st.write("**Abstract:**")
                    st.write(r["abstract"])

    with col_analytics:
        st.subheader("Analytics")

        if results:
            # Map
            st.markdown("**Research Density Map**")
            map_fig = make_country_map(df, results)
            if map_fig is not None:
                st.plotly_chart(map_fig, use_container_width=True)
            else:
                st.caption("No region data for this query.")

            # Timeline
            st.markdown("**Timeline**")
            timeline_fig = make_timeline(df, results)
            if timeline_fig is not None:
                st.plotly_chart(timeline_fig, use_container_width=True)
            else:
                st.caption("No year data for this query.")

        # Query log
        st.subheader("Query Log")
        if st.session_state["query_log"]:
            log_df = pd.DataFrame(st.session_state["query_log"])
            st.dataframe(log_df)
        else:
            st.caption("No queries yet.")


if __name__ == "__main__":
    main()
