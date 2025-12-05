import os
from typing import List, Dict

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


def build_index(df: pd.DataFrame):
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
    print("Index built.")
    return vectorizer, tfidf_matrix


def search(query: str, vectorizer, tfidf_matrix, df: pd.DataFrame, top_k: int = 5) -> List[Dict]:
    """
    Search for the query and return top_k results.
    """
    query = query.strip()
    if not query:
        return []

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Get indices of documents sorted by relevance
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "score": float(scores[idx]),
            "title": df.iloc[idx]["title"],
            "abstract": df.iloc[idx]["abstract"]
        })
    return results


def main():
    # 1. Load corpus
    csv_path = os.path.join("data", "interim", "corpus_v1.csv")
    df = load_corpus(csv_path)

    # 2. Build index
    vectorizer, tfidf_matrix = build_index(df)

    # 3. Loop for user queries
    print("\nType a query to search the abstracts. Type 'exit' to quit.\n")

    while True:
        query = input("Query> ")

        if query.lower().strip() in ["exit", "quit", ":q"]:
            print("Goodbye.")
            break

        results = search(query, vectorizer, tfidf_matrix, df, top_k=5)

        if not results:
            print("No results. Try a different query.")
            continue

        print(f"\nTop {len(results)} results:\n")
        for i, r in enumerate(results, start=1):
            print(f"{i}. score={r['score']:.3f}")
            print(f"   Title   : {r['title']}")
            print(f"   Abstract: {r['abstract'][:400]}...")
            print("-" * 80)
        print()


if __name__ == "__main__":
    main()
