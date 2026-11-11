"""Find the news that most likely explains a flagged move.

Two backends so you can compare them:
    tfidf       word-overlap baseline, runs anywhere, no model download
    embeddings  sentence-transformers (BGE by default), catches paraphrases
                ("yields jump" vs "bond selloff")

Search is restricted to news published from the previous business day up to the move
date. That candidate set is small (tens to a few hundred headlines), so scores are
computed exactly with a dot product. The FAISS index is used for the free-text search
over the whole archive (`search_all`), where the set is large.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from marketpulse.config import Config


class NewsRetriever:
    def __init__(self, backend: str = "tfidf", embedding_model: str = "BAAI/bge-small-en-v1.5"):
        if backend not in ("tfidf", "embeddings"):
            raise ValueError("backend must be 'tfidf' or 'embeddings'")
        self.backend = backend
        self.embedding_model = embedding_model

    def fit(self, news: pd.DataFrame) -> "NewsRetriever":
        self.news = news.reset_index(drop=True)
        texts = self.news["text"].astype(str).tolist()
        if self.backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english",
                                              sublinear_tf=True, min_df=1)
            self.matrix = self.vectorizer.fit_transform(texts)  # rows are L2-normalised
        else:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.embedding_model)
            self.matrix = self.model.encode(texts, normalize_embeddings=True, batch_size=64,
                                            show_progress_bar=True).astype("float32")
            self._build_faiss()
        return self

    def _build_faiss(self) -> None:
        try:
            import faiss
        except ImportError:
            self.faiss_index = None
            return
        self.faiss_index = faiss.IndexFlatIP(self.matrix.shape[1])
        self.faiss_index.add(self.matrix)

    def _encode_query(self, text: str):
        if self.backend == "tfidf":
            return self.vectorizer.transform([text])
        return self.model.encode([text], normalize_embeddings=True).astype("float32")

    def _scores(self, rows: np.ndarray, q) -> np.ndarray:
        m = self.matrix[rows]
        if self.backend == "tfidf":
            return (m @ q.T).toarray().ravel()
        return m @ q[0]

    def query(self, text: str, date, window_days: int = 1, k: int = 5) -> pd.DataFrame:
        date = pd.Timestamp(date).normalize()
        start = date - pd.tseries.offsets.BDay(window_days)
        rows = np.where((self.news["date"] >= start) & (self.news["date"] <= date))[0]
        if len(rows) == 0:
            return self.news.iloc[[]].assign(score=[])
        scores = self._scores(rows, self._encode_query(text))
        top = np.argsort(-scores)[:k]
        out = self.news.iloc[rows[top]].copy()
        out["score"] = scores[top]
        return out.reset_index(drop=True)

    def search_all(self, text: str, k: int = 10) -> pd.DataFrame:
        q = self._encode_query(text)
        if self.backend == "embeddings" and getattr(self, "faiss_index", None) is not None:
            scores, idx = self.faiss_index.search(q, k)
            out = self.news.iloc[idx[0]].copy()
            out["score"] = scores[0]
            return out.reset_index(drop=True)
        scores = self._scores(np.arange(len(self.news)), q)
        top = np.argsort(-scores)[:k]
        out = self.news.iloc[top].copy()
        out["score"] = scores[top]
        return out.reset_index(drop=True)


def move_query(inst_id: str, change: float, cfg: Config) -> str:
    """Turn a flagged move into a search query, e.g. '10-year Treasury yield jumps rises'."""
    inst = cfg.instrument(inst_id)
    if inst.kind == "rate":
        verbs = "jumps rises surges" if change > 0 else "drops falls slides"
    else:
        verbs = "rallies rises gains surges" if change > 0 else "slides falls drops tumbles"
    return f"{' '.join(inst.aliases)} {verbs} sharply"


def evaluate_retrieval(retriever: NewsRetriever, labels: pd.DataFrame, changes: pd.DataFrame,
                       cfg: Config, k_values=(1, 3, 5), window_days: int = 1) -> pd.DataFrame:
    """Recall@k and MRR on hand-labelled rows.

    labels columns: date, instrument, relevant_url (the article that explains the move).
    A row can appear several times if several articles are relevant.
    """
    max_k = max(k_values)
    grouped = labels.groupby(["date", "instrument"])["relevant_url"].apply(set)
    hits = {k: [] for k in k_values}
    rr = []
    for (d, inst), relevant in grouped.items():
        d = pd.Timestamp(d)
        if d not in changes.index:
            continue
        res = retriever.query(move_query(inst, changes.at[d, inst], cfg), d, window_days, max_k)
        urls = res["url"].tolist()
        for k in k_values:
            hits[k].append(any(u in relevant for u in urls[:k]))
        rank = next((i + 1 for i, u in enumerate(urls) if u in relevant), None)
        rr.append(1.0 / rank if rank else 0.0)
    row = {f"recall@{k}": float(np.mean(v)) if v else float("nan") for k, v in hits.items()}
    row["mrr"] = float(np.mean(rr)) if rr else float("nan")
    row["n_queries"] = len(rr)
    return pd.DataFrame([row], index=[retriever.backend])
