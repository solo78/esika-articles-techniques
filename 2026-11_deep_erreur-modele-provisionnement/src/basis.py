"""Base de fonctions du LASSO (papier, section 6.1.2, d'après McGuire et al. 2021).

- Rampes R_K(i), R_K(j), R_K(t) pour K = 0, ..., n - 1 (éq. 2 : R_K(x) = max(0, x - K)) ;
- Interactions de fonctions de Heaviside H_k(i) H_l(j), H_k(i) H_g(t), H_g(t) H_l(j), k, g, l = 2..n.

Les colonnes constantes sur le passé sont retirées, puis chaque colonne est centrée
et réduite sur les cellules passées.
Simplification : la mise à l'échelle exacte de McGuire et al. n'est pas publiée ;
on utilise l'écart type des cellules passées.
"""
import numpy as np
from .data import grid


def design(n):
    """Matrice (n*n) x p des fonctions de base non standardisées, avec leurs noms."""
    k, j, t = grid(n)
    v = {"i": k.ravel(), "j": j.ravel(), "t": t.ravel()}
    cols, names = [], []
    for name in ("i", "j", "t"):
        for K in range(0, n):
            cols.append(np.maximum(0, v[name] - K).astype(float))
            names.append(f"R{K}({name})")
    for a, b in (("i", "j"), ("i", "t"), ("t", "j")):
        for ka in range(2, n + 1):
            for kb in range(2, n + 1):
                cols.append(((v[a] >= ka) & (v[b] >= kb)).astype(float))
                names.append(f"H{ka}({a})H{kb}({b})")
    return np.column_stack(cols), np.array(names)


def standardized_design(n, past):
    """Base standardisée sur le passé ; renvoie X (toutes cellules), noms des colonnes gardées."""
    X, names = design(n)
    p = past.ravel()
    sd = X[p].std(0)
    keep = sd > 0
    X, names = X[:, keep], names[keep]
    m, s = X[p].mean(0), X[p].std(0)
    return (X - m) / s, names
