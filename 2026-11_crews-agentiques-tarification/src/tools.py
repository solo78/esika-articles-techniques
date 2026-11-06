"""Outils déterministes mis à disposition des agents.

- eda_tool   : profil exploratoire complet (formes, manquants, distributions, fréquence
               observée par modalité, anomalies), sans LLM, donc sans hallucination de chiffres.
- asop_grid  : grille ASOP chargée en contexte mis en cache (« Cache-Augmented Generation »).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ASOP_GRID = (Path(__file__).resolve().parent / "asop_grid.md").read_text()


def eda_tool(df: pd.DataFrame, target: str = "ClaimNb", exposure: str = "Exposure") -> str:
    L = []
    L.append(f"Dimensions : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    L.append(f"Colonnes et types : {', '.join(f'{c} ({df[c].dtype})' for c in df.columns)}")
    na = df.isna().sum()
    L.append("Valeurs manquantes : " + (", ".join(f"{c}={v}" for c, v in na.items() if v) or "aucune"))
    y, e = df[target].astype(float), df[exposure].astype(float)
    L.append(f"Sinistres : total {y.sum():.0f}, exposition totale {e.sum():.1f} années, "
             f"fréquence observée {y.sum() / e.sum():.4f}")
    L.append("Distribution du nombre de sinistres : " + ", ".join(
        f"{int(k)}:{v}" for k, v in y.value_counts().sort_index().items()))
    L.append(f"Exposition : min {e.min():.4f}, médiane {e.median():.3f}, max {e.max():.3f}, "
             f"part > 1 an : {(e > 1).mean():.4%}")
    for c in df.columns:
        if c in (target, exposure):
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            q = s.quantile([0, .01, .25, .5, .75, .99, 1]).round(2).tolist()
            L.append(f"{c} (numérique) : unique {s.nunique()}, quantiles 0/1/25/50/75/99/100 % = {q}, "
                     f"asymétrie {s.skew():.2f}")
            try:
                b = pd.qcut(s, 5, duplicates="drop")
                f = (y.groupby(b, observed=True).sum() / e.groupby(b, observed=True).sum()).round(4)
                L.append(f"   fréquence observée par quintile de {c} : {f.tolist()}")
            except ValueError:
                pass
        else:
            vc = s.value_counts()
            f = (y.groupby(s).sum() / e.groupby(s).sum()).round(4)
            top = ", ".join(f"{k}:{vc[k]} (fréq {f[k]})" for k in vc.index[:12])
            L.append(f"{c} (catégorielle) : {s.nunique()} modalités ; {top}")
    num = df.select_dtypes("number").drop(columns=[target], errors="ignore")
    corr = num.corr()
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:]
             if abs(corr.loc[a, b]) > 0.3]
    L.append("Corrélations |r| > 0,3 : " + (", ".join(f"{a}-{b} {r:.2f}" for a, b, r in pairs) or "aucune"))
    return "\n".join(L)
