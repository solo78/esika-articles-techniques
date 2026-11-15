"""Recherche dans la vue non structurée : BM25 (plein texte) et LSA (proxy d'embeddings denses).

La LSA (TF-IDF puis SVD tronquée) remplace ici un modèle d'embeddings type multilingual-e5-large,
non téléchargeable dans l'environnement d'exécution. Le point étudié (top-k fixe) ne dépend pas
de la qualité fine des embeddings : même une recherche parfaite renvoie k documents.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

TOKEN = r"(?u)\b\w[\w']+\b"


class BM25:
    def __init__(self, texts, k1=1.5, b=.75):
        self.cv = CountVectorizer(token_pattern=TOKEN, lowercase=True)
        X = self.cv.fit_transform(texts).tocsr().astype(float)
        n = X.shape[0]
        dl = np.asarray(X.sum(1)).ravel()
        df = np.asarray((X > 0).sum(0)).ravel()
        self.idf = np.log(1 + (n - df + .5) / (df + .5))
        X = X.tocoo()
        denom = X.data + k1 * (1 - b + b * dl[X.row] / dl.mean())
        data = X.data * (k1 + 1) / denom * self.idf[X.col]
        self.W = sparse.csr_matrix((data, (X.row, X.col)), shape=X.shape).tocsc()

    def search(self, query: str, k: int) -> np.ndarray:
        q = (self.cv.transform([query]) > 0).astype(float)
        s = np.asarray((self.W @ q.T).todense()).ravel()
        return np.argsort(-s, kind="stable")[:k]


class LSA:
    def __init__(self, texts, dim=150, seed=0):
        self.tv = TfidfVectorizer(token_pattern=TOKEN, sublinear_tf=True, min_df=2)
        X = self.tv.fit_transform(texts)
        self.svd = TruncatedSVD(dim, random_state=seed)
        self.E = normalize(self.svd.fit_transform(X))

    def embed(self, texts):
        return normalize(self.svd.transform(self.tv.transform(texts)))

    def search(self, query: str, k: int) -> np.ndarray:
        s = self.E @ self.embed([query])[0]
        return np.argsort(-s, kind="stable")[:k]

    def similar_to(self, i: int, k: int) -> np.ndarray:
        s = self.E @ self.E[i]
        s[i] = -np.inf
        return np.argsort(-s, kind="stable")[:k]
