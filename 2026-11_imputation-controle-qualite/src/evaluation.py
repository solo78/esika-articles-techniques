"""Mesures de détection et impact sur l'engagement actuariel."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .controles import matrice

CHAMPS = ["age", "anciennete", "salaire", "cotisation", "nb_mois"]


def metriques(decision: np.ndarray, verite: np.ndarray) -> dict:
    tp = int(np.sum(decision & verite))
    fp = int(np.sum(decision & ~verite))
    fn = int(np.sum(~decision & verite))
    tn = int(np.sum(~decision & ~verite))
    precision = tp / (tp + fp) if tp + fp else np.nan
    rappel = tp / (tp + fn) if tp + fn else np.nan
    f1 = 0.0 if (np.isnan(precision) or np.isnan(rappel) or precision + rappel == 0) else 2 * precision * rappel / (precision + rappel)
    return {"vp": tp, "fp": fp, "fn": fn, "vn": tn, "n_alertes": tp + fp,
            "precision": precision, "rappel": rappel, "f1": f1,
            "taux_fausses_alertes": fp / (fp + tn),       # FP / lignes saines
            "part_alertes_fausses": fp / (tp + fp) if tp + fp else np.nan}  # 1 - précision


def precision_moyenne(score: np.ndarray, verite: np.ndarray) -> float:
    return float(average_precision_score(verite, score))


def engagement(df: pd.DataFrame, cfg: dict) -> float:
    """Valeur actuelle des droits acquis (méthode des unités de crédit simplifiée)."""
    e = cfg["engagement"]
    age = df["age"].to_numpy(dtype=float)
    anc = np.clip(df["anciennete"].to_numpy(dtype=float), 0, None)
    sal = df["salaire"].to_numpy(dtype=float)
    rente = e["taux_annuite"] * anc * sal * 12.0
    duree = np.clip(e["age_retraite"] - age, 0, None)
    v = (1 + e["taux_actualisation"]) ** (-duree)
    a, b = e["gompertz_a"], e["gompertz_b"]
    age_c = np.minimum(age, e["age_retraite"])
    survie = np.exp(-a / b * (np.exp(b * e["age_retraite"]) - np.exp(b * age_c)))
    return float(np.sum(rente * v * survie * e["facteur_rente_60"]))


class ImputeurEngagement:
    """Imputation multiple des champs manquants avant le calcul de l'engagement.

    Les imputeurs sont ceux de la détection (appris une fois par simulation sur les lignes
    non signalées par C1), appliqués à chaque version corrigée du fichier : seule la correction diffère d'un
    traitement à l'autre. L'engagement est la moyenne sur les m tirages : imputer le log du
    salaire par sa seule espérance puis repasser à l'exponentielle sous-estimerait le salaire
    (inégalité de Jensen)."""

    def __init__(self, imputeurs: list, colonnes):
        # On réutilise les imputeurs déjà appris pour la détection (même apprentissage).
        self.imps = imputeurs
        self.colonnes = colonnes

    def completer(self, df: pd.DataFrame) -> list:
        X = matrice(df)[self.colonnes]
        lignes = X[["anciennete", "log_salaire"]].isna().any(axis=1).to_numpy()
        out = []
        for imp in self.imps:
            # Copie de l'imputeur : chaque traitement reçoit exactement les mêmes tirages
            # (nombres aléatoires communs), sinon l'état aléatoire avance à chaque appel.
            imp = copy.deepcopy(imp)
            d = df.copy()
            if lignes.any():
                Xi = imp.transform(X[lignes])
                d.loc[lignes, "anciennete"] = d.loc[lignes, "anciennete"].fillna(Xi["anciennete"])
                d.loc[lignes, "salaire"] = d.loc[lignes, "salaire"].fillna(np.exp(Xi["log_salaire"]))
            out.append(d)
        return out


def corriger(df_obs: pd.DataFrame, vrai: pd.DataFrame, champs_errones: dict, revue: np.ndarray) -> pd.DataFrame:
    """Les lignes envoyées en revue sont corrigées : une valeur erronée observée est remplacée
    par la vraie valeur ; une valeur manquante reste manquante ; une ligne saine est inchangée."""
    out = df_obs.copy()
    for c, flag in champs_errones.items():
        m = revue & flag & out[c].notna().to_numpy()
        out.loc[m, c] = vrai.loc[m, c]
    return out
