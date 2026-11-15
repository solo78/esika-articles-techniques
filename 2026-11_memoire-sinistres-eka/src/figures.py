"""Figures de résultats (palette Esika, titres en français)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VERT, VERT_C, OR, ROUGE, GRIS = "#0B4F3C", "#5FA88A", "#C99A2E", "#B03A2E", "#6E7781"
COUL = {"BM25": GRIS, "RAG vectoriel": ROUGE, "Vue structurée": OR, "Double vue": VERT,
        "Double vue (balayage 0.95)": VERT, "Oracle": VERT_C}
TITRES = {"Q_part": "Part en contentieux\n(branche x pays)", "Q_top3": "3 facteurs les plus\nfréquents en contentieux",
          "Q_rare": "Comptage d'un\nfacteur rare", "Q_prec": "10 précédents\ncomparables"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def fig_succes(main, out):
    qs = ["Q_part", "Q_top3", "Q_rare", "Q_prec"]
    meths = ["BM25", "RAG vectoriel", "Vue structurée", "Double vue", "Oracle"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.6), sharey=True)
    for ax, q in zip(axes, qs):
        sub = main[(main.question == q) & ~main.dispositif.str.contains("calibré")]
        for y, m in enumerate(meths):
            r = sub[sub.dispositif.str.startswith(m) & ~((m == "Double vue") & sub.dispositif.str.contains("0.80"))]
            if m != "Double vue":
                r = r[r.dispositif == m]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.errorbar(r.succes, y, xerr=[[r.succes - r.ic95_bas], [r.ic95_haut - r.succes]],
                        fmt="o", color=COUL[m], ms=8, capsize=3)
            ax.text(min(r.succes + .04, .9), y + .18, f"{r.succes:.0%}".replace("%", " %"), fontsize=9, color=COUL[m])
        ax.set_xlim(0, 1.05); ax.set_title(TITRES[q], fontsize=11)
        ax.set_yticks(range(len(meths))); ax.set_yticklabels(meths)
        ax.grid(axis="x", alpha=.3); ax.invert_yaxis()
        ax.set_xlabel("Taux de succès (IC 95 %)")
    fig.suptitle("Questions L2 : taux de succès par dispositif (300 simulations, 1 500 questions par type, 1 455 pour les 3 facteurs)\n"
                 "RAG et BM25 : top-8 (top-10 pour les précédents) ; extraction à 92 % de champs corrects",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(out / "fig1_succes_par_dispositif.png", dpi=150); plt.close(fig)


def fig_sensibilite(sens, out):
    qs = ["Q_part", "Q_top3", "Q_rare", "Q_prec"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), sharey=True)
    ps = [.70, .85, .92, .98]
    for ax, q in zip(axes, qs):
        s = sens[sens.q == q]
        for m, c, lab in [("Vue structurée", OR, "Vue structurée"),
                          ("Double vue (balayage 0.95)" if q == "Q_rare" else "Double vue", VERT, "Double vue")]:
            v = [s[(s.method == m) & np.isclose(s.param, p)].succes.iloc[0] for p in ps]
            ax.plot(ps, v, marker="o", color=c, label=lab, lw=2 if lab == "Double vue" else 1.5,
                    ls="-" if lab == "Double vue" else "--")
        for m, k, c, ls in [("RAG vectoriel", 8, ROUGE, ":"), ("BM25", 50, GRIS, "-.")]:
            kk = 10 if q == "Q_prec" else k
            r = s[(s.method == m) & np.isclose(s.param, kk)]
            if not r.empty:
                ax.axhline(r.succes.iloc[0], color=c, ls=ls, lw=1.2,
                           label=f"{m} top-{kk}")
        ax.set_title(TITRES[q], fontsize=11); ax.set_xticks(ps)
        ax.set_xticklabels([f"{int(p*100)} %" for p in ps]); ax.set_ylim(0, 1.05)
        ax.set_xlabel("Champs correctement extraits"); ax.grid(alpha=.3)
    axes[0].set_ylabel("Taux de succès")
    h, l = axes[2].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Sensibilité à la qualité d'extraction de la vue structurée (300 simulations appariées)\n"
                 "Sur les deux premières questions, la double vue route vers la vue structurée : les deux courbes sont confondues",
                 fontsize=11)
    fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(out / "fig2_sensibilite_extraction.png", dpi=150); plt.close(fig)


def fig_biais(df, out):
    d = df[df.q == "Q_part"]
    cfg = [("Vue structurée", .98), ("Vue structurée", .92), ("Vue structurée", .85), ("Vue structurée", .70),
           ("BM25", 50), ("RAG vectoriel", 8)]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for y, (m, p) in enumerate(cfg):
        e = d[(d.method == m) & np.isclose(d.param, p)].err.dropna() * 100
        lo, med, hi = np.percentile(e, [10, 50, 90])
        lab = f"{m} ({int(p*100)} %)" if m == "Vue structurée" else f"{m} top-{int(p)}"
        ax.plot([lo, hi], [y, y], color=COUL[m], lw=3, alpha=.6)
        ax.plot(e.mean(), y, "o", color=COUL[m], ms=8)
        ax.text(hi + .5, y, f"moyenne {e.mean():+.1f} pts".replace(".", ","), va="center", fontsize=9)
    ax.axvline(0, color="k", lw=.8)
    ax.axvspan(-5, 5, color=VERT_C, alpha=.12, label="zone de succès (±5 pts)")
    ax.set_yticks(range(len(cfg)))
    ax.set_yticklabels([f"{m} ({int(p*100)} %)" if m == "Vue structurée" else f"{m} top-{int(p)}" for m, p in cfg])
    ax.invert_yaxis(); ax.set_xlabel("Écart à la vraie part en contentieux (points) : moyenne et intervalle 10 %-90 %")
    ax.set_title("Part en contentieux (environ 30 % en moyenne) : erreurs d'extraction et biais de l'estimation", fontsize=11)
    ax.legend(loc="upper right", frameon=False); ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(out / "fig3_biais_part_contentieux.png", dpi=150); plt.close(fig)


def make_all(df, main, sens, out):
    fig_succes(main, out)
    fig_sensibilite(sens, out)
    fig_biais(df, out)
