# ArchLearn Mini
A small-scale information retrieval and text analytics project for exploring creative learning in architecture and art education.

Goal:
To collect around 100 academic abstracts from open sources like Semantic Scholar and build a tool to search, analyze, and visualize themes about creativity, design learning, and pedagogy.

Scope:
- Abstracts only (no full papers)
- English language
- From 2015 to 2025
- Focused on architecture and art education

Technical plan:
- Retrieval model: keyword-based and simple semantic (TF-IDF + optional Sentence-BERT)
- Visualization plan:
  - Word cloud of top keywords
  - Bar chart of most frequent terms per decade
  - Line chart showing rise/fall of key topics over time
  - Search interface for exploring abstracts
  - Location map