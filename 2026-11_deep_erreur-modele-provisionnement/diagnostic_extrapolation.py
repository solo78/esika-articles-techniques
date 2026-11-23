"""Diagnostic du biais d'extrapolation (ce que le papier ne fait pas, mais que nous ajoutons).

Pour les 6 premiers triangles du contrôle de réalisme de chaque jeu (graines 100 à 105, choisies d'avance),
on compare la provision du modèle 1se quand ses effets de calendrier sont prolongés (règle du papier)
ou figés au niveau du dernier trimestre observé (t = 40). Écrit results/diagnostic_extrapolation.csv.
"""
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from src.basis import design
from src.data import triangle

N = 40


def frozen_design(n, past):
    """Base standardisée comme src.basis, mais avec t remplacé par min(t, n) pour les cellules futures."""
    X, _ = design(n)
    k, j = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1), indexing="ij")
    t = (k + j - 1).ravel()
    # reconstruit les colonnes dépendant de t avec t gelé
    v = {"i": k.ravel(), "j": j.ravel(), "t": np.minimum(t, n)}
    cols = []
    for name in ("i", "j", "t"):
        for K in range(0, n):
            cols.append(np.maximum(0, v[name] - K).astype(float))
    for a, b in (("i", "j"), ("i", "t"), ("t", "j")):
        for ka in range(2, n + 1):
            for kb in range(2, n + 1):
                cols.append(((v[a] >= ka) & (v[b] >= kb)).astype(float))
    Xf = np.column_stack(cols)
    p = past.ravel()
    sd = X[p].std(0)
    keep = sd > 0
    X, Xf = X[:, keep], Xf[:, keep]
    m, s = X[p].mean(0), X[p].std(0)
    return (X - m) / s, (Xf - m) / s


def one(args):
    ds, seed = args
    y, mu, past = triangle(ds, seed)
    X, Xf = frozen_design(N, past)
    wd = tempfile.mkdtemp(dir="work")
    with open(os.path.join(wd, "meta.txt"), "w") as f:
        f.write(f"{X.shape[0]} {X.shape[1]}")
    np.asfortranarray(X).T.tofile(os.path.join(wd, "X.bin"))
    np.asfortranarray(Xf).T.tofile(os.path.join(wd, "Xf.bin"))
    past.ravel().astype(float).tofile(os.path.join(wd, "past.bin"))
    y.ravel()[past.ravel()].astype(float).tofile(os.path.join(wd, "Y.bin"))
    subprocess.run(["Rscript", "src/diag_extrapolation.R", wd], check=True)
    r1, r2 = [float(v) for v in open(os.path.join(wd, "diag.txt")).read().split()]
    for fn in ("X.bin", "Xf.bin"):
        os.remove(os.path.join(wd, fn))
    return dict(jeu=ds, graine=seed, provision_vraie=mu[~past].sum(), paiements_futurs_realises=y[~past].sum(),
                provision_1se_calendrier_prolonge=r1, provision_1se_calendrier_fige=r2)


if __name__ == "__main__":
    jobs = [(ds, 100 + i) for ds in (1, 2) for i in range(6)]
    with ThreadPoolExecutor(2) as ex:
        rows = list(ex.map(one, jobs))
    D = pd.DataFrame(rows)
    D["ecart_prolonge"] = D.provision_1se_calendrier_prolonge / D.paiements_futurs_realises - 1
    D["ecart_fige"] = D.provision_1se_calendrier_fige / D.paiements_futurs_realises - 1
    D.to_csv("results/diagnostic_extrapolation.csv", index=False)
    print(D.round(3).to_string())
