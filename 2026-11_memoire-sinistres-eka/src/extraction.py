"""Vue structurée : induction du schéma et extraction simulée avec erreurs.

Le modèle d'erreur reprend les ordres de grandeur publiés par Loffredo et al. (2026) :
environ 92 % de champs corrects, 6 % d'incohérences mineures, 2 % de manquants.
Les tirages aléatoires sont communs à tous les niveaux de précision (couplage monotone),
pour que les comparaisons entre niveaux soient appariées.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .corpus import BRANCH_NAMES, PAYS, F_NAMES

CAT_FIELDS = {"branche": len(BRANCH_NAMES), "pays": len(PAYS), "cause": 3}


def induce_schema(df: pd.DataFrame, rng, n_sample: int = 200, min_count: int = 5) -> list[int]:
    """Facteurs retenus dans le schéma : ceux vus au moins `min_count` fois dans l'échantillon d'induction."""
    idx = rng.choice(len(df), n_sample, replace=False)
    counts = df.iloc[idx][[f"f{j}" for j in range(len(F_NAMES))]].sum().to_numpy()
    return [j for j in range(len(F_NAMES)) if counts[j] >= min_count]


class ExtractionDraws:
    """Tirages uniformes figés une fois par simulation."""

    def __init__(self, df: pd.DataFrame, rng):
        n = len(df)
        self.U = {k: rng.random(n) for k in list(CAT_FIELDS) + ["litige", "montant"]}
        self.V = {k: rng.random(n) for k in CAT_FIELDS}
        self.UF = rng.random((n, len(F_NAMES)))
        self.noise = rng.normal(0, .3, n)


def extract(df: pd.DataFrame, draws: ExtractionDraws, schema: list[int], p: float,
            f_recall: float | None = None, f_fp: float | None = None) -> pd.DataFrame:
    """p : précision des champs de base ; f_recall / f_fp : rappel et faux positifs des facteurs
    (par défaut p et (1-p)/20, hypothèse initiale ; le scénario calibré utilise les valeurs mesurées en E2)."""
    f_recall = p if f_recall is None else f_recall
    f_fp = (1 - p) / 20 if f_fp is None else f_fp
    out = pd.DataFrame({"id": df["id"]})
    minor = p + (1 - p) * .75
    for k, ncat in CAT_FIELDS.items():
        u, v, true = draws.U[k], draws.V[k], df[k].to_numpy()
        wrong = (true + 1 + np.floor(v * (ncat - 1)).astype(int)) % ncat  # autre catégorie
        val = np.where(u < p, true, np.where(u < minor, wrong, -1))
        out[k] = val
    u, true = draws.U["litige"], df["litige"].to_numpy().astype(float)
    out["litige"] = np.where(u < p, true, np.where(u < minor, 1 - true, np.nan))
    u, true = draws.U["montant"], df["montant"].to_numpy()
    out["montant"] = np.where(u < p, true, np.where(u < minor, true * np.exp(draws.noise), np.nan))
    for j in range(len(F_NAMES)):
        if j in schema:
            t = df[f"f{j}"].to_numpy()
            uf = draws.UF[:, j]
            out[f"f{j}"] = np.where(t, uf < f_recall, uf < f_fp)
        else:
            out[f"f{j}"] = np.nan  # champ absent du schéma
    return out
