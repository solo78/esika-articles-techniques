"""Figures de résultats (palette Esika, titres et axes en français, dpi 150)."""
import pickle

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import triangle

VF, VC, OR, RG, GR = "#0B4F3C", "#5FA88A", "#C99A2E", "#B03A2E", "#6E7781"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11})


def _save(fig, out, name):
    fig.tight_layout()
    fig.savefig(f"{out}/figures/{name}.png", dpi=150)
    plt.close(fig)


def fil_rouge_famille(out):
    f = pd.read_csv(f"{out}/fil_rouge_famille.csv")
    s = pd.read_csv(f"{out}/fil_rouge_resume.csv", index_col=0)["valeur"]
    qmax = int(2 * s["q_minCV"])
    f = f[f.q <= qmax]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[3, 2]))
    a1.plot(f.q, f.provision / 1e3, color=GR, lw=1.2, marker="o", ms=3)
    a1.scatter(f.q[f.poids_aposteriori > 1e-3], f.provision[f.poids_aposteriori > 1e-3] / 1e3, color=VF, s=40, zorder=3,
               label="modèles qui comptent dans le vote")
    a1.axhline(s["provision_vraie"] / 1e3, color=OR, ls="--", lw=1.2)
    a1.text(qmax, s["provision_vraie"] / 1e3 + 0.3, "provision vraie (espérance)", color=OR, ha="right", fontsize=9)
    a1.axhline(s["bma_complete_moyenne"] / 1e3, color=VF, lw=1)
    a1.text(qmax, s["bma_complete_moyenne"] / 1e3 + 0.3, "moyenne bayésienne", color=VF, ha="right", fontsize=9)
    a1.set_ylabel("Provision (milliers)")
    a1.set_title(f"Fil rouge : la provision de chaque modèle ({qmax} premiers sur {int(s['Q'])})")
    a1.legend(frameon=False, loc="center left", fontsize=9)
    a2.bar(f.q, f.poids_aposteriori, color=VF, width=0.8)
    a2.set_ylabel("Poids a posteriori")
    a2.set_xlabel("Modèle q (complexité croissante : pénalité plus faible)")
    for q, lab in ((int(s["q_1se"]), "1se"), (int(s["q_minCV"]), "minCV")):
        a2.annotate(lab, (q, f.poids_aposteriori[f.q == q].iloc[0]), xytext=(q + 6, f.poids_aposteriori.max() * 0.9), color=GR,
                    arrowprops=dict(arrowstyle="-", color=GR), fontsize=9)
    _save(fig, out, "fig_fil_rouge_famille")


def fil_rouge_vote(out):
    t = pd.read_csv(f"{out}/fil_rouge_5_modeles.csv")
    s = pd.read_csv(f"{out}/fil_rouge_resume.csv", index_col=0)["valeur"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8), gridspec_kw=dict(width_ratios=[1, 1.3]))
    y = np.arange(len(t))[::-1]
    a1.barh(y, t.poids, color=[VF if q == s["q_1se"] - 1 else VC for q in t.q])
    for yi, w in zip(y, t.poids):
        a1.text(w + 0.01, yi, f"{100 * w:.0f} %", va="center", fontsize=9)
    a1.set_yticks(y)
    a1.set_yticklabels([f"{m} ({int(v)} variables)" for m, v in zip(t.modele, t.nb_variables)], fontsize=9)
    a1.set_xlim(0, max(t.poids) * 1.3)
    a1.set_xlabel("Poids a posteriori")
    a1.set_title("1. Chaque modèle reçoit un poids")
    a2.scatter(t.provision, y, s=600 * t.poids + 20, color=[VF if q == s["q_1se"] - 1 else VC for q in t.q], zorder=3)
    for yi, r in zip(y, t.provision):
        a2.text(r, yi + 0.3, f"{r:,.0f}".replace(",", " "), ha="center", fontsize=8.5)
    a2.axvline(s["bma5_moyenne"], color=OR, lw=2)
    a2.text(s["bma5_moyenne"] + 15, -0.9, f"moyenne pondérée : {s['bma5_moyenne']:,.0f}".replace(",", " "),
            color=OR, ha="left", fontsize=9)
    a2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    a2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " ")))
    a2.set_yticks([])
    a2.set_ylim(-1.2, len(t) - 0.4)
    a2.set_xlabel("Provision du modèle")
    a2.set_title("2. La provision est la moyenne pondérée")
    fig.suptitle(f"Fil rouge : le vote de 5 modèles (écart type {s['bma5_ecart_type']:.0f}, soit un CoV de "
                 f"{100 * s['bma5_cov']:.2f} %)".replace(".", ","), fontweight="bold", fontsize=11)
    _save(fig, out, "fig_fil_rouge_vote")


def fil_rouge_compromis(out):
    f = pd.read_csv(f"{out}/fil_rouge_famille.csv")
    s = pd.read_csv(f"{out}/fil_rouge_resume.csv", index_col=0)["valeur"]
    f = f[f.q <= int(1.5 * s["q_minCV"])]
    post = f.log_vraisemblance + f.log_apriori
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(f.q, f.log_vraisemblance - f.log_vraisemblance.max(), color=VC, lw=2, label="log vraisemblance (ajustement)")
    ax.plot(f.q, f.log_apriori - f.log_apriori.max(), color=OR, lw=2, label="log a priori (simplicité)")
    ax.plot(f.q, post - post.max(), color=VF, lw=2.5, label="log a posteriori = somme des deux")
    qm = int(f.q[post.idxmax()])
    ax.axvline(qm, color=GR, ls=":", lw=1)
    ax.text(qm + 1, -90, f"maximum : modèle {qm} (le 1se)", color=GR, fontsize=9)
    ax.set_ylim(-300, 10)
    ax.set_xlabel("Modèle q (complexité croissante)")
    ax.set_ylabel("Écart au maximum de chaque courbe")
    ax.set_title("Le vote bayésien arbitre entre ajustement et simplicité")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _save(fig, out, "fig_fil_rouge_compromis")


def bon_ajustement(out, work, ds=2):
    prim = pickle.load(open(f"{work}/prim_{ds}.pkl", "rb"))
    y, mu, past = triangle(ds, {1: 1, 2: 2}[ds])
    ll, adm, i1 = prim["loglik"], prim["admissible"], prim["i1se"]
    # premier modèle éliminé par les portes qui s'ajuste au passé mieux que le 1se
    q_bad = int(np.where((~adm) & (ll >= ll[i1]) & (np.arange(len(ll)) > i1))[0][0])
    n = 40
    k, j = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1), indexing="ij")
    t = (k + j - 1).ravel()
    sel = k.ravel() > n - 10
    tt = np.arange(1, 2 * n)

    def by_t(v):
        return np.array([v[sel & (t == u)].sum() for u in tt]) / 1e9
    P = prim["fam"]["preds"]
    obs = by_t(np.where(past.ravel(), y.ravel(), 0.0))
    obs[tt > n] = np.nan
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.plot(tt[tt <= n], obs[tt <= n], "o", color=GR, ms=3, label="paiements observés")
    ax.plot(tt, by_t(mu.ravel()), color=OR, ls="--", lw=1.5, label="espérance vraie (générateur)")
    ax.plot(tt, by_t(P[prim["i1se"]]), color=VF, lw=2, label="modèle principal (1se)")
    gain = ll[q_bad] - ll[i1]
    ax.plot(tt, by_t(P[q_bad]), color=RG, lw=2,
            label=f"modèle q = {q_bad + 1} : mieux ajusté (+{gain:.0f} points de log-vraisemblance), éliminé par les portes")
    ax.axvline(n + 0.5, color=GR, lw=1)
    ax.text(n + 1, ax.get_ylim()[1] * 0.95, "futur", color=GR, fontsize=9)
    ax.text(n - 1, ax.get_ylim()[1] * 0.95, "passé", color=GR, fontsize=9, ha="right")
    ax.set_xlabel("Trimestre de paiement t")
    ax.set_ylabel("Paiements des 10 derniers trimestres de survenance (Md)")
    ax.set_title(f"Jeu {ds} : deux modèles presque identiques sur le passé, divergents sur le futur")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _save(fig, out, f"fig_jeu{ds}_bon_ajustement_mauvaise_prevision")
    sb, sp = by_t(P[q_bad])[tt > n].sum(), by_t(P[prim["i1se"]])[tt > n].sum()
    pd.Series(dict(q_elimine=q_bad + 1, q_1se=i1 + 1, gain_log_vraisemblance=gain,
                   nb_variables_elimine=prim["fam"]["df"][q_bad], nb_variables_1se=prim["fam"]["df"][i1],
                   provision_elimine=prim["reserves"][q_bad], provision_1se=prim["reserves"][i1],
                   ecart_provision=prim["reserves"][q_bad] / prim["reserves"][i1] - 1,
                   prevision_10_dernieres_survenances_elimine=sb, prevision_10_dernieres_survenances_1se=sp,
                   ecart_10_dernieres=sb / sp - 1)).to_csv(f"{out}/jeu{ds}_modele_elimine.csv", header=["valeur"])


def distribution(out, work, ds=2):
    res = np.load(f"{work}/matrix_{ds}_res.npy")
    W = np.load(f"{work}/matrix_{ds}_W.npy")
    keep = W.sum(1) > 0
    vals = res[keep].ravel()
    wts = (W[keep] / keep.sum()).ravel()
    ok = np.isfinite(vals) & (wts > 0)
    P = pd.read_csv(f"{out}/modele_principal.csv")
    p = P[(P.jeu == ds) & (P.apriori == "1se")].iloc[0]
    Bt = pd.read_csv(f"{out}/bootstrap.csv")
    b = Bt[(Bt.jeu == ds) & (Bt.apriori == "1se")].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(vals[ok] / 1e9, bins=40, weights=wts[ok], color=VC, edgecolor="white",
            label=f"BMA + bootstrap ({int(b.replications_gardees)} réplications gardées)")
    prim = pickle.load(open(f"{work}/prim_{ds}.pkl", "rb"))
    w0 = prim["priors"]["1se"]["w"]
    r0 = prim["reserves"]
    m = w0 > 1e-4
    ax2 = ax.twinx()
    ax2.vlines(r0[m] / 1e9, 0, w0[m], color=VF, lw=2)
    ax2.set_ylabel("Poids, modèle principal seul", color=VF)
    ax2.spines["top"].set_visible(False)
    ax.axvline(p.provision_vraie / 1e9, color=OR, ls="--", lw=1.5)
    ax.text(p.provision_vraie / 1e9, ax.get_ylim()[1] * 0.9, " provision vraie", color=OR, fontsize=9)
    ax.set_xlabel("Provision (Md)")
    ax.set_ylabel("Probabilité (bootstrap)")
    ax.set_title(f"Jeu {ds} : distribution de la provision, a priori 1se")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    _save(fig, out, f"fig_jeu{ds}_distribution_provision")


def decomposition(out):
    Bt = pd.read_csv(f"{out}/bootstrap.csv")
    Wt = pd.read_csv(f"{out}/temoins.csv")
    rows = []
    for ds in (1, 2):
        for pr in ("1se", "minCV"):
            b = Bt[(Bt.jeu == ds) & (Bt.apriori == pr)]
            if len(b):
                b = b.iloc[0]
                rows.append((f"Jeu {ds}, LASSO a priori {pr}", b.imse_cov, b.parametre_cov, b.processus_cov, b.sous_total_cov))
        g = Wt[(Wt.jeu == ds) & Wt.methode.str.startswith("GLM")].iloc[0]
        rows.append((f"Jeu {ds}, GLM de structure connue", np.nan, g.parametre_cov, g.processus_cov, g.parametre_plus_processus_cov))
        mk = Wt[(Wt.jeu == ds) & Wt.methode.str.startswith("Mack")].iloc[0]
        rows.append((f"Jeu {ds}, Mack", np.nan, np.nan, np.nan, mk.parametre_plus_processus_cov))
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ylab = [r[0] for r in rows][::-1]
    for i, r in enumerate(rows[::-1]):
        for v, c, mk_, lab in ((r[1], VF, "D", "erreur de modèle interne (IMSE)"), (r[2], OR, "o", "erreur de paramètre"),
                               (r[3], GR, "s", "erreur de processus"), (r[4], RG, "|", "total estimé")):
            if np.isfinite(v):
                ax.plot(100 * v, i, mk_, color=c, ms=9 if mk_ != "|" else 16, mew=2.5,
                        label=lab if i == len(rows) - 1 or (lab not in ax.get_legend_handles_labels()[1]) else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(ylab, fontsize=9)
    ax.set_xlim(left=0)
    ax.set_xlabel("Coefficient de variation de la provision (%)")
    ax.set_title("Composantes de l'erreur de prévision (bootstrap B = 300, portes x 1,1)")
    h, l = ax.get_legend_handles_labels()
    d = dict(zip(l, h))
    ax.legend(d.values(), d.keys(), frameon=False, fontsize=8.5, loc="lower right")
    _save(fig, out, "fig_decomposition_erreur")


def sensibilite(out):
    S = pd.read_csv(f"{out}/sensibilite_portes.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    cols = {(1, "1se"): VC, (1, "minCV"): GR, (2, "1se"): VF, (2, "minCV"): OR}
    for (ds, pr), c in cols.items():
        d = S[(S.jeu == ds) & (S.apriori == pr)].sort_values("elargissement_portes")
        d = d[d.replications_gardees >= 30]   # moins de 30 réplications sur 300 : point non interprétable, retiré
        if len(d):
            ax.plot(d.elargissement_portes, 100 * d.sous_total_cov, marker="o", color=c, lw=2, label=f"jeu {ds}, a priori {pr}")
    ax.set_xticks([1.0, 1.1, 1.2, 1.3])
    ax.set_xticklabels(["Table 2", "x 1,1\n(travail)", "x 1,2", "x 1,3"])
    ax.text(1.32, 3.2, "Jeu 1 avec la Table 2 : moins de 10 réplications\ngardées sur 300, point non représenté", fontsize=8, color=GR, ha="right")
    ax.set_xlim(0.95, 1.35)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Erreur totale estimée (CoV, %)")
    ax.set_title("Sensibilité de l'erreur estimée à la largeur des portes")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _save(fig, out, "fig_sensibilite_portes")


def realisme(out):
    R = pd.read_csv(f"{out}/realisme.csv")
    Bt = pd.read_csv(f"{out}/bootstrap.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharey=True)
    rng = np.random.default_rng(0)
    for ax, ds in zip(axes, (1, 2)):
        d = R[R.jeu == ds]
        sub = Bt[(Bt.jeu == ds) & (Bt.apriori == "1se")].sous_total_cov.iloc[0]
        ax.axhspan(-100 * 1.96 * sub, 100 * 1.96 * sub, color=VC, alpha=0.2,
                   label="± 1,96 x erreur annoncée (triangle principal)")
        ax.axhline(0, color=GR, lw=0.8)
        x = rng.uniform(-0.15, 0.15, len(d))
        ax.plot(0 + x, 100 * d.erreur_relative_bma, "o", color=VF, label="BMA LASSO (a priori 1se)")
        ax.plot(1 + x, 100 * d.erreur_relative_mack, "o", color=OR, label="Mack")
        ax.plot(2 + x, 100 * d.ecart_esperance_vs_realise, "o", color=GR, label="espérance vraie")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["BMA LASSO", "Mack", "espérance\nvraie"])
        ax.set_title(f"Jeu {ds} : {len(d)} triangles indépendants")
    axes[0].set_ylabel("Erreur relative vs paiements futurs réalisés (%)")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Contrôle de réalisme : l'erreur réellement commise", fontweight="bold")
    _save(fig, out, "fig_realisme")


def all_figures(out, work):
    fil_rouge_famille(out)
    fil_rouge_vote(out)
    fil_rouge_compromis(out)
    for ds in (1, 2):
        bon_ajustement(out, work, ds)
        distribution(out, work, ds)
    decomposition(out)
    sensibilite(out)
    realisme(out)
