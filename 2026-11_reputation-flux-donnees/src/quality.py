"""Couche 2 : évaluation atomique de chaque bordereau reçu.

Chaque bordereau mensuel reçoit une note entre 0 et 1 sur sept dimensions de qualité,
reprises de la grille du survey (Wu & Konstantinidis, 2025, section VII-A) et limitées
à celles que l'on peut mesurer automatiquement sur un bordereau de sinistres.

Chaque dimension compare un taux de défaut à une tolérance contractuelle :
note = max(0, 1 - taux observé / tolérance). Une note de 1 signifie « aucun défaut »,
une note de 0 signifie « tolérance atteinte ou dépassée ».
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .simulation import DEADLINE_DAYS, N_MONTHS

DIMENSIONS = ["completude", "validite", "unicite", "exactitude", "fraicheur", "coherence", "tracabilite"]

LABELS = {
    "completude": "Complétude",
    "validite": "Validité",
    "unicite": "Unicité",
    "exactitude": "Exactitude (rapprochement comptable)",
    "fraicheur": "Fraîcheur",
    "coherence": "Cohérence",
    "tracabilite": "Traçabilité",
}

# Poids pour l'usage « provisionnement » : les défauts qu'un nettoyage automatique corrige
# (doublons exacts, identifiants manquants) pèsent moins que ceux qui faussent le triangle
# sans être visibles ligne à ligne (lignes manquantes, survenances mal codées).
WEIGHTS_RESERVING = {
    "exactitude": 0.25,
    "completude": 0.20,
    "coherence": 0.20,
    "fraicheur": 0.15,
    "validite": 0.08,
    "unicite": 0.07,
    "tracabilite": 0.05,
}

TOLERANCE = {
    "completude": 0.10,   # part de lignes sans date de survenance
    "validite": 0.05,     # part de lignes invalides (montant <= 0, survenance après paiement)
    "unicite": 0.10,      # part de lignes en double
    "exactitude": 0.10,   # écart relatif entre le bordereau et le flux comptable
    "tracabilite": 0.10,  # part de lignes sans identifiant de sinistre
    "lag0": 0.15,         # hausse de la part des paiements « survenus le mois même »
}
LAG0_DEADZONE = 0.06          # variation jugée normale (fluctuation d'échantillonnage)
CRITICAL_GAP = 0.30           # écart comptable déclenchant une alerte immédiate
SCALE_OK, SCALE_CRIT = np.log(2.0), np.log(5.0)
MIN_HISTORY = 6               # nombre de bordereaux validés avant d'activer les contrôles de cohérence


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def raw_metrics(rows: dict, cash: np.ndarray) -> pd.DataFrame:
    """Indicateurs bruts par mois de transmission."""
    m = rows["pay"]
    L = N_MONTHS + 1
    n = np.bincount(m, minlength=L).astype(float)
    acc_nan = np.isnan(rows["acc"])
    miss_acc = np.bincount(m, acc_nan, minlength=L)
    miss_id = np.bincount(m, rows["claim"] < 0, minlength=L)
    acc_filled = np.where(acc_nan, -1e9, rows["acc"])
    invalid = (rows["amount"] <= 0) | (~acc_nan & (acc_filled > m))
    n_invalid = np.bincount(m, invalid, minlength=L)
    # doublons exacts : même ligne transmise plusieurs fois
    _, first_idx = np.unique(rows["uid"], return_index=True)
    n_unique = np.bincount(m[first_idx], minlength=L)
    total = np.bincount(m, rows["amount"], minlength=L)
    # le rapprochement comptable se fait après suppression des doublons exacts :
    # un même défaut n'est ainsi compté qu'une fois (en unicité, pas en exactitude)
    total_dedup = np.bincount(m[first_idx], rows["amount"][first_idx], minlength=L)
    lag0 = np.bincount(m, ~acc_nan & (acc_filled == m), minlength=L)
    n_acc = n - miss_acc

    with np.errstate(divide="ignore", invalid="ignore"):
        df = pd.DataFrame({
            "n_lines": n,
            "miss_acc_rate": np.where(n > 0, miss_acc / n, 0.0),
            "miss_id_rate": np.where(n > 0, miss_id / n, 0.0),
            "invalid_rate": np.where(n > 0, n_invalid / n, 0.0),
            "dup_rate": np.where(n > 0, 1 - n_unique / n, 0.0),
            "total": total,
            "cash": cash,
            "gap": np.where(cash > 0, np.abs(total_dedup - cash) / cash, 0.0),
            "lag0_share": np.where(n_acc > 0, lag0 / n_acc, np.nan),
        })
    df.index.name = "month"
    return df.iloc[1:]


def evaluate(rows: dict, cash: np.ndarray, delay: np.ndarray, active: np.ndarray,
             weights: dict[str, float] = WEIGHTS_RESERVING) -> pd.DataFrame:
    """Notes par dimension, note atomique globale et drapeau d'incident critique, mois par mois."""
    df = raw_metrics(rows, cash)
    df["delay_days"] = delay[1:]
    df["active"] = active[1:]

    df["completude"] = _clip01(1 - df["miss_acc_rate"] / TOLERANCE["completude"])
    df["validite"] = _clip01(1 - df["invalid_rate"] / TOLERANCE["validite"])
    df["unicite"] = _clip01(1 - df["dup_rate"] / TOLERANCE["unicite"])
    df["exactitude"] = _clip01(1 - df["gap"] / TOLERANCE["exactitude"])
    df["tracabilite"] = _clip01(1 - df["miss_id_rate"] / TOLERANCE["tracabilite"])
    df["fraicheur"] = _clip01(1 - (df["delay_days"] - DEADLINE_DAYS).clip(lower=0) / 20)

    # Cohérence : comparaison à l'historique VALIDÉ du partenaire. Un bordereau suspect
    # n'entre jamais dans la référence, ce qui évite qu'une dérive devienne « la nouvelle norme ».
    coh, lag_scores, scale_scores, critical = [], [], [], []
    ref_lag, ref_tot = [], []
    for month, r in df.iterrows():
        if not r["active"] or r["n_lines"] == 0:
            coh.append(np.nan); lag_scores.append(np.nan); scale_scores.append(np.nan)
            critical.append(False)
            continue
        s_lag = s_scale = 1.0
        crit_scale = False
        if len(ref_tot) >= MIN_HISTORY:
            base_lag = np.nanmedian(ref_lag[-12:])
            excess = max(r["lag0_share"] - base_lag - LAG0_DEADZONE, 0)
            s_lag = float(_clip01(1 - excess / TOLERANCE["lag0"]))
            base_tot = np.median(ref_tot[-6:])
            lr = abs(np.log(max(r["total"], 1e-9) / base_tot))
            s_scale = 1.0 if lr <= SCALE_OK else float(_clip01(1 - (lr - SCALE_OK) / (SCALE_CRIT - SCALE_OK)))
            crit_scale = lr >= SCALE_CRIT
        s_coh = min(s_lag, s_scale)
        is_crit = bool(r["gap"] > CRITICAL_GAP or crit_scale)
        if s_coh >= 0.8 and not is_crit:
            ref_lag.append(r["lag0_share"]); ref_tot.append(r["total"])
        coh.append(s_coh); lag_scores.append(s_lag); scale_scores.append(s_scale); critical.append(is_crit)

    df["coherence"] = coh
    df["coh_lag0"] = lag_scores
    df["coh_scale"] = scale_scores
    df["critical"] = critical
    w = pd.Series(weights)
    df["score"] = (df[DIMENSIONS] * w[DIMENSIONS]).sum(axis=1) / w.sum()
    df.loc[~df["active"] | (df["n_lines"] == 0), "score"] = np.nan
    return df


def explain(row: pd.Series, weights: dict[str, float] = WEIGHTS_RESERVING, top: int = 2) -> str:
    """Explication lisible d'une note atomique : quelles dimensions coûtent le plus de points."""
    if row.get("critical", False):
        return ("Incident critique : écart de {:.0%} avec le flux comptable, "
                "bordereau mis en quarantaine.".format(row["gap"]))
    losses = {d: weights[d] * (1 - row[d]) for d in DIMENSIONS}
    worst = sorted(losses.items(), key=lambda kv: kv[1], reverse=True)[:top]
    worst = [(d, l) for d, l in worst if l > 0.005]
    if not worst:
        return "Aucun défaut significatif."
    detail = {
        "completude": lambda r: f"{r['miss_acc_rate']:.1%} de lignes sans date de survenance",
        "validite": lambda r: f"{r['invalid_rate']:.1%} de lignes invalides",
        "unicite": lambda r: f"{r['dup_rate']:.1%} de doublons",
        "exactitude": lambda r: f"écart de {r['gap']:.1%} avec le flux comptable",
        "fraicheur": lambda r: f"transmis {int(r['delay_days'])} jours après la fin du mois",
        "coherence": lambda r: f"part des paiements survenus le mois même anormalement haute ({r['lag0_share']:.0%})",
        "tracabilite": lambda r: f"{r['miss_id_rate']:.1%} de lignes sans identifiant",
    }
    parts = [f"{LABELS[d]} ({detail[d](row)}, -{l * 100:.1f} pts)" for d, l in worst]
    return "Note {:.2f}. Principales causes : ".format(row["score"]) + " ; ".join(parts) + "."
