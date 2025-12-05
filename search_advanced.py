import os
from typing import List, Dict

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util


# ---------- Helper functions ----------

def load_corpus(csv_path: str) -> pd.DataFrame:
    """
    Load the cleaned corpus from CSV.
    """
    print(f"Loading corpus from: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Corpus file not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    if "abstract" not in df.columns:
        raise ValueError("Corpus must have an 'abstract' column.")

    if "title" not in df.columns:
        df["title"] = "Untitled"

    df["abstract"] = df["abstract"].astype(str)
    df["title"] = df["title"].astype(str)

    print(f"Loaded {len(df)} documents.")
    return df


def build_tfidf_index(df: pd.DataFrame):
    """
    Build a TF-IDF index on the abstract texts.
    """
    print("Building TF-IDF index...")
    texts = df["abstract"].tolist()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    print("TF-IDF index built.")
    return vectorizer, tfidf_matrix


def build_bm25_index(df: pd.DataFrame):
    """
    Build a BM25 index using simple tokenization (lowercase + split).
    """
    print("Building BM25 index...")
    texts = df["abstract"].astype(str).tolist()
    tokenized_corpus = [text.lower().split() for text in texts]

    bm25 = BM25Okapi(tokenized_corpus)
    print("BM25 index built.")
    return bm25, tokenized_corpus


def build_semantic_index(df: pd.DataFrame):
    """
    Build SentenceTransformer embeddings for semantic similarity.
    """
    print("Loading SentenceTransformer model (this loads once)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")  # small, good general model

    texts = df["abstract"].astype(str).tolist()
    print("Encoding documents into semantic embeddings...")
    doc_embeddings = model.encode(texts, convert_to_tensor=True)

    print("Semantic index built.")
    return model, doc_embeddings


def search_all(
    query: str,
    df: pd.DataFrame,
    vectorizer,
    tfidf_matrix,
    bm25: BM25Okapi,
    semantic_model: SentenceTransformer,
    doc_embeddings,
    tfidf_threshold: float = 0.05
) -> List[Dict]:

    """
    Search the corpus with TF-IDF, BM25, and Sentence Transformers.
    Return ALL documents whose TF-IDF cosine similarity >= tfidf_threshold.
    """
    query = query.strip()
    if not query:
        return []

    # --- TF-IDF similarity ---
    query_vec_tfidf = vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_vec_tfidf, tfidf_matrix).flatten()

    # --- BM25 scores ---
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)

    # --- Semantic similarity ---
    query_emb = semantic_model.encode(query, convert_to_tensor=True)
    semantic_scores_tensor = util.cos_sim(query_emb, doc_embeddings).squeeze(0)
    semantic_scores = semantic_scores_tensor.cpu().numpy()

    # Prepare combined results
    results = []
    for idx in range(len(df)):
        tfidf_score = float(tfidf_scores[idx])

        # Filter by tfidf_threshold
        if tfidf_score < tfidf_threshold:
            continue

        result = {
            "index": idx,
            "title": df.iloc[idx]["title"],
            "abstract": df.iloc[idx]["abstract"],
            "tfidf_score": tfidf_score,
            "bm25_score": float(bm25_scores[idx]),
            "semantic_score": float(semantic_scores[idx])
        }
        results.append(result)

    # Sort primarily by TF-IDF score (descending)
    results.sort(key=lambda x: x["tfidf_score"], reverse=True)

    return results


# ---------- Command-line interface ----------

def main():
    # 1. Load corpus
    csv_path = os.path.join("data", "interim", "corpus_v1.csv")
    df = load_corpus(csv_path)

    # 2. Build all three indices
    vectorizer, tfidf_matrix = build_tfidf_index(df)
    bm25, _ = build_bm25_index(df)
    semantic_model, doc_embeddings = build_semantic_index(df)

    # 3. Interactive loop
    print("\nType a query to search the abstracts.")
    print("You'll see TF-IDF, BM25, and semantic scores for all results with TF-IDF >= 0.01.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Query> ")

        if query.lower().strip() in ["exit", "quit", ":q"]:
            print("Goodbye.")
            break

        results = search_all(
            query,
            df,
            vectorizer,
            tfidf_matrix,
            bm25,
            semantic_model,
            doc_embeddings,
            tfidf_threshold=0.01
        )

        if not results:
            print("No results found with TF-IDF score >= 0.05.")
            continue

        total = len(results)
        print(f"\nFound {total} results (TF-IDF >= 0.05).\n")

        # Show in batches of 20
        batch_size = 20
        start = 0

        while start < total:
            end = min(start + batch_size, total)
            print(f"Showing results {start + 1} to {end} of {total}:\n")

            for i in range(start, end):
                r = results[i]
                print(f"{i + 1}.")
                print(f"   Title         : {r['title']}")
                print(f"   TF-IDF score  : {r['tfidf_score']:.4f}")
                print(f"   BM25 score    : {r['bm25_score']:.4f}")
                print(f"   Semantic score: {r['semantic_score']:.4f}")
                print(f"   Abstract      : {r['abstract'][:400]}...")
                print("-" * 80)
            print()

            # If we've shown everything, break
            if end >= total:
                break

            # Ask user if they want to see more
            choice = input("Show more results? (y/n): ").strip().lower()
            if choice not in ["y", "yes"]:
                break

            # Move to next batch
            start = end

        print()


if __name__ == "__main__":
    main()
