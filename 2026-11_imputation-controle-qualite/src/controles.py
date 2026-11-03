"""Couches de contrôle qualité, de la plus simple à la plus élaborée.

C0 : règles métier (bornes, cohérence cotisation = taux x salaire x mois)
C1 : C0 + statistiques robustes (barrières de Tukey par catégorie)
C2 : C1 + Isolation Forest sur données complétées par la médiane (remplissage naïf)
C3 : C1 + Isolation Forest sur données imputées (imputation multiple)
C4 : C1 + « predict-then-check » : prédire chaque champ à partir des autres
     (après imputation multiple) et signaler les résidus trop grands au regard
     de l'incertitude totale (modèle + imputation).

Les modèles (imputation, forêt, régressions) sont appris sur les lignes non
signalées par C1 : on n'apprend pas sur ce qui est déjà reconnu comme suspect.
Aucune grandeur vraie n'entre dans un score ou une décision.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import HuberRegressor

warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def matrice(df: pd.DataFrame) -> pd.DataFrame:
    """Variables utilisées par les modèles : log des montants, indicatrices de catégorie."""
    X = pd.DataFrame({
        "age": df["age"],
        "anciennete": df["anciennete"],
        "log_salaire": np.log(df["salaire"]),
        "log_cotisation_mensuelle": np.log(df["cotisation"] / df["nb_mois"]),
        "nb_mois": df["nb_mois"],
        "cat_maitrise": (df["categorie"] == "maitrise").astype(float),
        "cat_cadre": (df["categorie"] == "cadre").astype(float),
    })
    return X.loc[:, X.notna().any()]  # retire une colonne entièrement absente


# --- C0 : règles métier -------------------------------------------------------
def regles(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    r = cfg["regles"]
    tx = cfg["population"]["taux_cotisation"]
    age, anc, sal, cot, mois = (df[c].to_numpy(dtype=float) for c in
                                ["age", "anciennete", "salaire", "cotisation", "nb_mois"])
    with np.errstate(invalid="ignore", divide="ignore"):
        hors = (age < r["age"][0]) | (age > r["age"][1])
        hors |= (anc < 0) | (anc > age - r["age_minimum_embauche"])
        hors |= (sal < r["salaire"][0]) | (sal > r["salaire"][1])
        hors |= (mois < 1) | (mois > 12)
        ratio = cot / (tx * sal * mois)
        hors |= np.abs(ratio - 1.0) > r["tolerance_cotisation"]
    return np.nan_to_num(hors, nan=0).astype(bool)


# --- C1 : statistiques robustes ------------------------------------------------
def tukey(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    k = cfg["statistiques"]["tukey_k"]
    with np.errstate(invalid="ignore", divide="ignore"):
        series = {
            "log_salaire": np.log(df["salaire"]),
            "log_cotisation_mensuelle": np.log(df["cotisation"] / df["nb_mois"]),
            "anciennete": df["anciennete"],
        }
    flag = np.zeros(len(df), dtype=bool)
    for cat in df["categorie"].unique():
        m = (df["categorie"] == cat).to_numpy()
        for s in series.values():
            v = s.to_numpy(dtype=float)[m]
            if np.all(np.isnan(v)):
                continue
            q1, q3 = np.nanpercentile(v, [25, 75])
            iqr = q3 - q1
            out = (v < q1 - k * iqr) | (v > q3 + k * iqr)
            flag[np.where(m)[0][np.nan_to_num(out, nan=0).astype(bool)]] = True
    return flag


# --- C2 / C3 : Isolation Forest -------------------------------------------------
def _iforest_scores(X_fit: np.ndarray, X_all: np.ndarray, cfg: dict, graine: int):
    mod = IsolationForest(n_estimators=cfg["isolation_forest"]["n_estimators"],
                          contamination=cfg["isolation_forest"]["budget_revue"], random_state=graine)
    mod.fit(X_fit)
    return mod.score_samples(X_all), mod.offset_


def iforest_naif(X: pd.DataFrame, apprentissage: np.ndarray, cfg: dict, graine: int):
    med = X[apprentissage].median()
    Xf = X.fillna(med).to_numpy()
    s, _ = _iforest_scores(Xf[apprentissage], Xf, cfg, graine)
    seuil = np.quantile(s[apprentissage], cfg["isolation_forest"]["budget_revue"])
    return -s, s < seuil  # score d'anomalie (plus grand = plus suspect), décision


class ImputeurPassif:
    """Imputation par équations chaînées (MICE) avec traitement « passif » de l'identité
    cotisation = taux x salaire x mois.

    Salaire et cotisation mensuelle sont quasi colinéaires. Si on les laisse tous deux dans
    MICE, une ligne où les deux manquent ne peut être imputée qu'à partir de l'autre, elle-même
    manquante : la chaîne reste bloquée près de la moyenne. On ne garde donc qu'une variable
    dans MICE (le salaire) : un salaire manquant est d'abord déduit de la cotisation quand elle
    est observée, et la cotisation manquante est ensuite déduite du salaire imputé."""

    COT = "log_cotisation_mensuelle"
    SAL = "log_salaire"

    def __init__(self, taux: float, **kw):
        self.log_taux = float(np.log(taux))
        self.imp = IterativeImputer(**kw)

    def _reduire(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.COT not in X.columns:
            return X
        R = X.drop(columns=[self.COT])
        deduit = R[self.SAL].isna() & X[self.COT].notna()
        R.loc[deduit, self.SAL] = X.loc[deduit, self.COT] - self.log_taux
        return R

    def fit(self, X: pd.DataFrame):
        self.imp.fit(self._reduire(X))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        R = self._reduire(X)
        Z = pd.DataFrame(self.imp.transform(R), columns=R.columns, index=X.index)
        out = X.copy()
        for c in R.columns:
            out[c] = out[c].fillna(Z[c])
        if self.COT in X.columns:
            out[self.COT] = out[self.COT].fillna(out[self.SAL] + self.log_taux)
        return out


def imputations_multiples(X: pd.DataFrame, apprentissage: np.ndarray, cfg: dict, graine: int):
    imp_cfg = cfg["imputation"]
    out, modeles = [], []
    for j in range(imp_cfg["n_imputations"]):
        imp = ImputeurPassif(cfg["population"]["taux_cotisation"], sample_posterior=True,
                             max_iter=imp_cfg["max_iter"], random_state=graine + j)
        imp.fit(X[apprentissage])
        modeles.append(imp)
        out.append(imp.transform(X))
    return out, modeles


def iforest_impute(imputes: list, apprentissage: np.ndarray, cfg: dict, graine: int):
    """Scores moyennés sur les imputations ; le seuil est le quantile « budget » du score moyen
    sur les lignes d'apprentissage, pour que C2 et C3 revoient le même volume de dossiers."""
    scores = []
    for j, Xi in enumerate(imputes):
        s, _ = _iforest_scores(Xi.to_numpy()[apprentissage], Xi.to_numpy(), cfg, graine + 100 + j)
        scores.append(s)
    s = np.mean(scores, axis=0)
    seuil = np.quantile(s[apprentissage], cfg["isolation_forest"]["budget_revue"])
    return -s, s < seuil


# --- C4 : predict-then-check ----------------------------------------------------
def predict_then_check(X_obs: pd.DataFrame, imputes: list, apprentissage: np.ndarray, cfg: dict,
                       exclusions: dict | None = None):
    """Renvoie le z maximal par ligne. z = |x - mu| / sqrt(sigma_modele^2 + sigma_imputation^2),
    calculé seulement pour les valeurs observées.

    ``exclusions`` retire certains prédicteurs pour une cible donnée. Version « structurelle » :
    le salaire n'est pas prédit par la cotisation (et inversement), car leur cohérence comptable
    est déjà vérifiée par une règle ; on confronte chacun à la structure (âge, ancienneté,
    catégorie). Version « générique » : chaque champ est prédit par tous les autres."""
    ptc = cfg["predict_then_check"]
    eps = ptc["huber_epsilon"]
    exclusions = exclusions or {}
    n = len(X_obs)
    zmax = np.zeros(n)
    for cible in [c for c in ptc["cibles"] if c in X_obs.columns]:
        observe = X_obs[cible].notna().to_numpy()
        a_retirer = [cible] + [c for c in exclusions.get(cible, []) if c in X_obs.columns]
        preds, sig = [], []
        for Xi in imputes:
            autres = Xi.drop(columns=a_retirer).to_numpy()
            mu, sd = autres[apprentissage].mean(0), autres[apprentissage].std(0) + 1e-9
            A = (autres - mu) / sd
            fit_rows = apprentissage & observe
            y = X_obs[cible].to_numpy()
            mod = HuberRegressor(epsilon=eps, max_iter=500).fit(A[fit_rows], y[fit_rows])
            p = mod.predict(A)
            r = y[fit_rows] - p[fit_rows]
            sig.append(1.4826 * np.median(np.abs(r - np.median(r))))
            preds.append(p)
        preds = np.array(preds)
        mu_pred = preds.mean(0)
        sigma_imp = preds.std(0)
        sigma_mod = float(np.mean(sig))
        x = X_obs[cible].to_numpy()
        with np.errstate(invalid="ignore"):
            z = np.abs(x - mu_pred) / np.sqrt(sigma_mod ** 2 + sigma_imp ** 2)
        zmax = np.maximum(zmax, np.nan_to_num(z, nan=0.0))
    return zmax


def appliquer_chaine(df: pd.DataFrame, cfg: dict, graine: int, complet: bool = False) -> dict:
    """Applique toutes les couches. Renvoie les décisions et les scores par ligne."""
    c0 = regles(df, cfg)
    c1 = c0 | tukey(df, cfg)
    X = matrice(df)
    apprentissage = ~c1
    score_if_naif, dec_if_naif = iforest_naif(X, apprentissage, cfg, graine)
    if complet:
        imputes, modeles = [X], []  # aucune valeur manquante : pas d'imputation
        score_if_imp, dec_if_imp = score_if_naif, dec_if_naif
    else:
        imputes, modeles = imputations_multiples(X, apprentissage, cfg, graine)
        score_if_imp, dec_if_imp = iforest_impute(imputes, apprentissage, cfg, graine)
    z = predict_then_check(X, imputes, apprentissage, cfg, cfg["predict_then_check"]["exclusions"])
    z_gen = predict_then_check(X, imputes, apprentissage, cfg, {})
    k = cfg["predict_then_check"]["seuil_z"]
    return {
        "C0": c0, "C1": c1,
        "C2": c1 | dec_if_naif, "C3": c1 | dec_if_imp, "C4": c1 | (z > k),
        "C4_generique": c1 | (z_gen > k),
        "score_if_naif": score_if_naif, "score_if_impute": score_if_imp, "z_ptc": z, "z_ptc_generique": z_gen,
        "apprentissage": apprentissage, "imputeurs": modeles, "colonnes": X.columns,
    }
