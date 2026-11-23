"""Diagnostic (ajouté après le contrôle de réalisme) : où se loge la surestimation du modèle 1se ?

Pour les graines 100 et 101 des jeux 1 et 2 (choisies d'avance, les deux premières du contrôle), on compare la
prévision du 1se à l'espérance vraie, par tranche de trimestres de survenance. Écrit results/diagnostic_biais_survenance.csv.
"""
import os
import numpy as np
import pandas as pd
from src.data import triangle
from src.primary import analyse

N = 40
rows = []
for ds in (1, 2):
    for seed in (100, 101):
        y, mu, past = triangle(ds, seed)
        wd = os.path.join("work", f"diag_{ds}_{seed}")
        os.makedirs(wd, exist_ok=True)
        prim = analyse(y, past, N, wd)
        pred = prim["fam"]["preds"][prim["i1se"]].reshape(N, N)
        fut = ~past
        for lo, hi in ((1, 20), (21, 30), (31, 35), (36, 38), (39, 40)):
            sel = np.zeros((N, N), bool)
            sel[lo - 1:hi, :] = True
            m = (fut & sel)
            rows.append(dict(jeu=ds, graine=seed, survenances=f"{lo}-{hi}", provision_vraie=mu[m].sum(),
                             prevision_1se=pred[m].sum(), ecart=pred[m].sum() / mu[m].sum() - 1,
                             part_provision_vraie=mu[m].sum() / mu[fut].sum()))
D = pd.DataFrame(rows)
D.to_csv("results/diagnostic_biais_survenance.csv", index=False)
print(D.round(3).to_string())
