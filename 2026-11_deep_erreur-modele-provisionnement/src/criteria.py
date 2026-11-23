"""Évaluation scriptée des critères fixés avant exécution (voir retro.md). Aucun décompte n'est écrit à la main."""
import numpy as np
import pandas as pd


def evaluate(P, Bt, S, Wt, out):
    rows = []
    for ds in (1, 2):
        for prior in ("1se", "minCV"):
            p = P[(P.jeu == ds) & (P.apriori == prior)].iloc[0]
            b = Bt[(Bt.jeu == ds) & (Bt.apriori == prior)].iloc[0]
            rows.append(dict(critere="C1 bootstrap (portes x 1,1) épaissit l'IMSE", jeu=ds, apriori=prior,
                             valeur=f"{100 * p.imse_cov:.2f} % -> {100 * b.imse_cov:.2f} %",
                             verifie=bool(b.imse_cov >= p.imse_cov)))
            d = S[(S.jeu == ds) & (S.apriori == prior)].set_index("elargissement_portes")
            if d.loc[1.0, "replications_gardees"] < 30:   # moins de 10 % des réplications : rapport non interprétable
                r_ = d.loc[1.1, "sous_total_cov"] / d.loc[1.0, "sous_total_cov"]
                rows.append(dict(critere="C3 portes x 1,1 / Table 2 : rapport entre 1,2 et 1,8", jeu=ds, apriori=prior,
                                 valeur=f"non évaluable : {r_:.2f} calculé sur {int(d.loc[1.0, 'replications_gardees'])} réplications gardées avec la Table 2",
                                 verifie=None))
            else:
                ratio = d.loc[1.1, "sous_total_cov"] / d.loc[1.0, "sous_total_cov"]
                rows.append(dict(critere="C3 portes x 1,1 / Table 2 : rapport entre 1,2 et 1,8", jeu=ds, apriori=prior,
                                 valeur=f"{ratio:.2f}", verifie=bool(1.2 <= ratio <= 1.8)))
            ratio2 = d.loc[1.2, "sous_total_cov"] / d.loc[1.1, "sous_total_cov"]
            rows.append(dict(critere="C3 bis (ajouté après exécution) portes x 1,2 / x 1,1 : rapport entre 1,2 et 1,8",
                             jeu=ds, apriori=prior, valeur=f"{ratio2:.2f}", verifie=bool(1.2 <= ratio2 <= 1.8)))
            rows.append(dict(critere="C5 sous-total entre 5 % et 20 %", jeu=ds, apriori=prior,
                             valeur=f"{100 * b.sous_total_cov:.1f} %",
                             verifie=bool(0.05 <= b.sous_total_cov <= 0.20)))
            ecart = abs(p.moyenne_aposteriori / p.provision_1se_brute - 1)
            rows.append(dict(critere="C6 moyenne a posteriori proche du 1se brut (< 3 %)", jeu=ds, apriori=prior,
                             valeur=f"{100 * ecart:.2f} %", verifie=bool(ecart < 0.03)))
        fam = pd.read_csv(f"{out}/jeu{ds}_famille.csv")
        elim = fam[~fam.admis & np.isfinite(fam.provision)]
        big = elim[elim.poids_avant_elagage >= 0.05]
        rows.append(dict(critere="C2 modèle bien ajusté éliminé par les portes (poids avant élagage >= 0,05)",
                         jeu=ds, apriori="1se",
                         valeur=f"{len(big)} modèle(s) ; {len(elim)} éliminés, poids total avant élagage {elim.poids_avant_elagage.sum():.4f}",
                         verifie=bool(len(big) > 0)))
        b = Bt[(Bt.jeu == ds) & (Bt.apriori == "1se")].iloc[0]
        g = Wt[(Wt.jeu == ds) & Wt.methode.str.startswith("GLM")].iloc[0]
        rows.append(dict(critere="C4 PaE LASSO 1se > PaE GLM structure connue", jeu=ds, apriori="1se",
                         valeur=f"{100 * b.parametre_cov:.1f} % contre {100 * g.parametre_cov:.1f} %",
                         verifie=bool(b.parametre_cov > g.parametre_cov)))
    return pd.DataFrame(rows)
