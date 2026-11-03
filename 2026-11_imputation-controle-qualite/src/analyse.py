"""Agrégation, tests appariés et figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

VERT, VERT_CLAIR, OR, ROUGE, GRIS = "#0B4F3C", "#5FA88A", "#C99A2E", "#B03A2E", "#6E7781"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "savefig.dpi": 150})

LIBELLES = {
    "C0": "C0 Règles métier",
    "C1": "C1 + statistiques robustes",
    "C2": "C2 = C1 + Isolation Forest\n(remplissage médiane)",
    "C3": "C3 = C1 + Isolation Forest\n(imputation multiple)",
    "C4": "C4 = C1 + predict-then-check",
}
TYPES = {"unite_salaire": "Unité du salaire\n(x1000 ou /1000)", "salaire_incoherent": "Salaire incohérent\navec la cotisation",
         "declaration_coherente": "Déclaration erronée\nmais cohérente", "date_naissance": "Date de naissance\nerronée",
         "date_embauche": "Date d'embauche\nerronée"}
SCENARIOS = {"base": "Base (MAR, 10 %)", "mcar": "MCAR, 10 %", "mnar": "MNAR, 10 %", "non_reponse_5": "MAR, 5 %",
             "non_reponse_20": "MAR, 20 %", "non_reponse_30": "MAR, 30 %",
             "erreurs_dans_incomplets": "Erreurs concentrées\ndans les lignes incomplètes",
             "modele_mal_specifie": "Structure des salaires\nmal spécifiée", "sans_cotisation": "Sans la colonne\ncotisation"}


def fr(v, nd=2):
    """Nombre au format français (virgule décimale)."""
    return f"{v:.{nd}f}".replace(".", ",")


def fr_pct(v, nd=1):
    return f"{100 * v:.{nd}f} %".replace(".", ",")


def axes_fr(ax, pct=False, axe="y"):
    f = (lambda v, p: f"{100 * v:.0f} %") if pct else (lambda v, p: f"{v:g}".replace(".", ","))
    (ax.yaxis if axe == "y" else ax.xaxis).set_major_formatter(matplotlib.ticker.FuncFormatter(f))


def ic95(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return m, m - h, m + h


def test_apparie(df, scenario, a, b, col):
    """Différence b - a, appariée par simulation (mêmes graines)."""
    s = df[df.scenario == scenario]
    xa = s[s.traitement == a].set_index("sim")[col]
    xb = s[s.traitement == b].set_index("sim")[col]
    d = (xb - xa.reindex(xb.index)).dropna()
    m, lo, hi = ic95(d)
    if np.allclose(d, 0):
        p_t = p_w = 1.0
    else:
        p_t = float(stats.ttest_1samp(d, 0).pvalue)
        p_w = float(stats.wilcoxon(d).pvalue)
    return {"scenario": scenario, "a": a, "b": b, "mesure": col, "n": int(len(d)),
            "moyenne_a": float(xa.mean()), "moyenne_b": float(xb.mean()),
            "diff_moyenne": float(m), "ic95_bas": float(lo), "ic95_haut": float(hi),
            "p_student": p_t, "p_wilcoxon": p_w, "part_sims_b_superieur": float((d > 0).mean())}


def resume(df, scenario, traitements, cols):
    s = df[(df.scenario == scenario) & df.traitement.isin(traitements)]
    out = []
    for t in traitements:
        st = s[s.traitement == t]
        row = {"scenario": scenario, "traitement": t}
        for c in cols:
            if c in st and st[c].notna().any():
                m, lo, hi = ic95(st[c])
                row[c] = m
                row[c + "_ic_bas"] = lo
                row[c + "_ic_haut"] = hi
        out.append(row)
    return pd.DataFrame(out)


def analyser(df: pd.DataFrame, cfg: dict, dossier: Path):
    fig_dir = dossier / "figures"
    cols = ["n_alertes", "precision", "rappel", "f1", "taux_fausses_alertes", "erreur_engagement",
            "erreur_residuelle", "precision_moyenne"]
    principaux = ["Sans contrôle", "C0", "C1", "C2", "C3", "C4", "C4_generique", "Toutes erreurs corrigées",
                  "budget_IF_naif", "budget_IF_impute", "budget_PTC",
                  "score_IF_naif", "score_IF_impute", "score_PTC", "score_PTC_generique",
                  "Oracle_C1", "Oracle_C2", "Oracle_C4", "Oracle_C4_generique",
                  "Oracle_score_IF", "Oracle_score_PTC", "Oracle_score_PTC_generique"]
    base = df[df.scenario == "base"]
    tab = resume(df, "base", principaux, cols)
    tab.to_csv(dossier / "tableau_base.csv", index=False)

    # Contexte du scénario de base
    contexte = {
        "n_simulations": int(base.sim.nunique()),
        "n_affilies": cfg["simulation"]["n_affilies"],
        "anormaux_moyen": float(base.drop_duplicates("sim").n_anormaux.mean()),
        "part_lignes_incompletes": float(base.drop_duplicates("sim").part_lignes_incompletes.mean()),
        # Oracle (aucune valeur manquante) : toutes les erreurs restent visibles
        "anormaux_moyen_oracle": float((base[base.traitement == "Oracle_C1"][["vp", "fn"]].sum(axis=1)).mean()),
    }

    # Tests appariés
    tests = []
    for sc in df.scenario.unique():
        tests += [
            test_apparie(df, sc, "C2", "C3", "taux_fausses_alertes"),
            test_apparie(df, sc, "C2", "C3", "precision"),
            test_apparie(df, sc, "C2", "C3", "rappel"),
            test_apparie(df, sc, "C2", "C3", "f1"),
            test_apparie(df, sc, "score_IF_naif", "score_IF_impute", "precision_moyenne"),
            test_apparie(df, sc, "C1", "C4", "rappel"),
            test_apparie(df, sc, "C1", "C4", "taux_fausses_alertes"),
            test_apparie(df, sc, "C1", "C4", "f1"),
            test_apparie(df, sc, "C3", "C4", "f1"),
            test_apparie(df, sc, "score_PTC_generique", "score_PTC", "precision_moyenne"),
            test_apparie(df, sc, "budget_IF_impute", "budget_PTC", "precision"),
        ]
    for a, b in [("C1", "C4"), ("C1", "C3"), ("C0", "C1"), ("C2", "C3"), ("C3", "C4")]:
        d = df[df.scenario == "base"].copy()
        d["abs_res"] = d["erreur_residuelle"].abs()
        tests.append(test_apparie(d, "base", a, b, "abs_res"))
    tests = pd.DataFrame(tests)
    tests.to_csv(dossier / "tests_apparies.csv", index=False)

    # Sensibilités
    sens = []
    for sc in SCENARIOS:
        s = df[df.scenario == sc]
        if s.empty:
            continue
        row = {"scenario": sc}
        for t, c in [("score_IF_naif", "precision_moyenne"), ("score_IF_impute", "precision_moyenne"),
                     ("score_PTC", "precision_moyenne"), ("score_PTC_generique", "precision_moyenne"),
                     ("C1", "rappel"), ("C2", "taux_fausses_alertes"), ("C3", "taux_fausses_alertes"),
                     ("C4", "precision"), ("C4", "rappel"), ("C4", "taux_fausses_alertes"), ("C4", "f1"),
                     ("C3", "f1"), ("C4", "n_alertes"), ("Toutes erreurs corrigées", "erreur_engagement")]:
            row[f"{t}|{c}"] = float(s[s.traitement == t][c].mean())
        row["anormaux_moyen"] = float(s.drop_duplicates("sim").n_anormaux.mean())
        row["part_lignes_incompletes"] = float(s.drop_duplicates("sim").part_lignes_incompletes.mean())
        sens.append(row)
    sens = pd.DataFrame(sens)
    sens.to_csv(dossier / "sensibilites.csv", index=False)

    # Rappel par type d'erreur
    r = base[base.traitement.str.startswith("rappel_type_")]
    r = r.groupby(["traitement", "type_erreur"])[["detectes", "total"]].sum()
    r["rappel"] = r.detectes / r.total
    rappel_type = r.rappel.unstack()
    rappel_type.index = [i.replace("rappel_type_", "") for i in rappel_type.index]
    rappel_type.to_csv(dossier / "rappel_par_type.csv")

    # Fausses alertes selon que la ligne saine est complète ou non
    f = base[base.traitement.str.startswith("fa_")]
    f = f.groupby(["traitement", "groupe"])[["fausses_alertes", "total"]].sum()
    f["taux"] = f.fausses_alertes / f.total
    fa = f.taux.unstack()
    fa.index = [i.replace("fa_", "") for i in fa.index]
    fa.to_csv(dossier / "fausses_alertes_completude.csv")

    # Seuil du predict-then-check
    ks = base[base.traitement.str.startswith("C4_k")].groupby("traitement")[
        ["n_alertes", "precision", "rappel", "taux_fausses_alertes"]].mean()
    ks.index = [float(i.replace("C4_k", "")) for i in ks.index]
    ks = ks.sort_index()
    ks.to_csv(dossier / "seuil_predict_then_check.csv", index_label="seuil_z")

    json.dump(contexte, open(dossier / "contexte.json", "w"), indent=2)
    figures(df, tab, rappel_type, sens, ks, fa, fig_dir)
    print(tab[["traitement", "n_alertes", "precision", "rappel", "f1", "taux_fausses_alertes",
               "erreur_engagement", "erreur_residuelle", "precision_moyenne"]].round(4).to_string())
    print(tests[["scenario", "a", "b", "mesure", "moyenne_a", "moyenne_b", "diff_moyenne", "ic95_bas",
                 "ic95_haut", "p_wilcoxon"]].round(4).to_string())


def figures(df, tab, rappel_type, sens, ks, fa, d: Path):
    t = tab.set_index("traitement")

    # Figure 1 : détection par couche (scénario de base)
    ordre = ["C0", "C1", "C2", "C3", "C4"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, (col, titre, coul) in zip(axes, [("precision", "Précision", VERT), ("rappel", "Rappel", VERT_CLAIR),
                                             ("taux_fausses_alertes", "Taux de fausses alertes", ROUGE)]):
        vals = [t.loc[o, col] for o in ordre]
        err = [[t.loc[o, col] - t.loc[o, col + "_ic_bas"] for o in ordre],
               [t.loc[o, col + "_ic_haut"] - t.loc[o, col] for o in ordre]]
        ax.bar(range(5), vals, color=coul, yerr=err, capsize=3, ecolor=GRIS)
        ax.set_xticks(range(5))
        ax.set_xticklabels(["C0\nrègles", "C1\n+ stats", "C2\n+ IF naïf", "C3\n+ IF imputé", "C4\n+ PTC"], fontsize=9)
        ax.set_title(titre)
        for i, (v, o) in enumerate(zip(vals, ordre)):
            ax.text(i, t.loc[o, col + "_ic_haut"], fr_pct(v, 2) if col == "taux_fausses_alertes" else fr(v),
                    ha="center", va="bottom", fontsize=8)
        axes_fr(ax, pct=(col == "taux_fausses_alertes"))
    fig.suptitle("Ajouter les couches une à une : ce que chacune apporte (300 simulations, scénario de base)", fontsize=11)
    fig.tight_layout()
    fig.savefig(d / "fig1_detection_par_couche.png")
    plt.close(fig)

    # Figure 2 : pouvoir de classement des scores (précision moyenne)
    noms = [("score_IF_naif", "Isolation Forest\nremplissage médiane", OR),
            ("score_IF_impute", "Isolation Forest\nimputation multiple", OR),
            ("score_PTC_generique", "Predict-then-check\ngénérique", VERT_CLAIR),
            ("score_PTC", "Predict-then-check\nstructurel", VERT)]
    oracle = {"score_IF_naif": "Oracle_score_IF", "score_IF_impute": "Oracle_score_IF",
              "score_PTC_generique": "Oracle_score_PTC_generique", "score_PTC": "Oracle_score_PTC"}
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for i, (n, lab, c) in enumerate(noms):
        v = t.loc[n, "precision_moyenne"]
        ax.bar(i - 0.18, v, width=0.36, color=c, label="Avec non-réponse" if i == 0 else None)
        ax.text(i - 0.18, v, fr(v), ha="center", va="bottom", fontsize=8)
        vo = t.loc[oracle[n], "precision_moyenne"]
        ax.bar(i + 0.18, vo, width=0.36, color="white", edgecolor=c, hatch="//", label="Oracle : sans non-réponse" if i == 0 else None)
        ax.text(i + 0.18, vo, fr(vo), ha="center", va="bottom", fontsize=8)
    base_rate = df[(df.scenario == "base")].drop_duplicates("sim").n_anormaux.mean() / 4000
    ax.axhline(base_rate, color=GRIS, ls="--", lw=1, label=f"Classement au hasard ({fr(base_rate, 3)})")
    axes_fr(ax)
    ax.set_xticks(range(4))
    ax.set_xticklabels([n[1] for n in noms], fontsize=9)
    ax.set_ylabel("Précision moyenne (aire sous la courbe précision-rappel)")
    ax.set_title("Classer les dossiers du plus au moins suspect : le score compte plus que l'imputation")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(d / "fig2_pouvoir_de_classement.png")
    plt.close(fig)

    # Figure 3 : rappel par type d'erreur
    rt = rappel_type.loc[["C0", "C1", "C2", "C3", "C4"], list(TYPES)]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.imshow(rt.to_numpy(), aspect="auto", cmap=matplotlib.colors.LinearSegmentedColormap.from_list("e", ["#FFFFFF", VERT_CLAIR, VERT]), vmin=0, vmax=1)
    for i in range(rt.shape[0]):
        for j in range(rt.shape[1]):
            v = rt.iloc[i, j]
            ax.text(j, i, fr_pct(v, 0), ha="center", va="center", color="white" if v > 0.6 else "black", fontsize=9)
    ax.set_xticks(range(len(TYPES)))
    ax.set_xticklabels(list(TYPES.values()), fontsize=8)
    ax.set_yticks(range(5))
    ax.set_yticklabels(["C0 règles", "C1 + stats", "C2 + IF naïf", "C3 + IF imputé", "C4 + PTC"], fontsize=9)
    ax.set_title("Part des erreurs détectées, par type d'erreur et par couche")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(d / "fig3_rappel_par_type.png")
    plt.close(fig)

    # Figure 4 : sensibilités (précision moyenne des scores)
    s = sens.set_index("scenario").loc[[k for k in SCENARIOS if k in set(sens.scenario)]]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    x = np.arange(len(s))
    w = 0.27
    for k, (col, lab, c) in enumerate([("score_IF_naif|precision_moyenne", "IF, remplissage médiane", GRIS),
                                       ("score_IF_impute|precision_moyenne", "IF, imputation multiple", OR),
                                       ("score_PTC|precision_moyenne", "Predict-then-check structurel", VERT)]):
        ax.bar(x + (k - 1) * w, s[col], width=w, color=c, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIOS[i] for i in s.index], fontsize=8)
    ax.set_ylabel("Précision moyenne")
    ax.set_title("Sensibilité : le classement des méthodes tient, l'écart dû à l'imputation reste faible")
    ax.set_ylim(0, 0.85)
    axes_fr(ax)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(d / "fig4_sensibilites.png")
    plt.close(fig)

    # Figure 5 : seuil du predict-then-check
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ks.index, ks["rappel"], "o-", color=VERT, label="Rappel")
    ax.plot(ks.index, ks["precision"], "s-", color=OR, label="Précision")
    ax2 = ax.twinx()
    ax2.bar(ks.index, ks["n_alertes"], width=0.3, color=GRIS, alpha=0.25)
    ax2.set_ylabel("Dossiers à revoir (sur 4 000)", color=GRIS)
    ax2.spines["top"].set_visible(False)
    ax.set_xlabel("Seuil z du predict-then-check")
    axes_fr(ax)
    axes_fr(ax, axe="x")
    ax.set_ylim(0, 1.05)
    ax.set_title("Régler le seuil : un arbitrage explicite entre rappel et charge de revue")
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    fig.tight_layout()
    fig.savefig(d / "fig5_seuil_ptc.png")
    plt.close(fig)

    # Figure 6 : engagement, effet des seules erreurs non détectées
    b = df[df.scenario == "base"]
    ordre_e = ["C0", "C1", "C2", "C3", "C4"]
    data = [b[b.traitement == o]["erreur_residuelle"].to_numpy() * 100 for o in ordre_e]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.55)
    for patch, c in zip(bp["boxes"], [GRIS, GRIS, OR, OR, VERT]):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    for m in bp["medians"]:
        m.set_color("black")
    ax.axhline(0, color=GRIS, lw=0.8)
    ax.set_xticklabels(["C0\nrègles", "C1\n+ stats", "C2\n+ IF naïf", "C3\n+ IF imputé", "C4\n+ PTC"], fontsize=9)
    ax.set_ylabel("Écart signé dû aux erreurs non détectées (%)")
    axes_fr(ax)
    sc = b[b.traitement == "Sans contrôle"]["erreur_engagement"].median() * 100
    nr = b[b.traitement == "Toutes erreurs corrigées"]["erreur_engagement"].mean() * 100
    ax.set_title(f"Engagement : écart laissé par les erreurs non détectées\n(sans contrôle : +{sc:.0f} % en médiane ; "
                 f"la non-réponse seule pèse {nr:.1f} %, non représentée)".replace(".", ","), fontsize=10)
    fig.tight_layout()
    fig.savefig(d / "fig6_engagement.png")
    plt.close(fig)
