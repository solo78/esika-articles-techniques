"""Lance l'expérience complète et produit les tableaux et figures de l'article.

Usage : python run_all.py [--runs 300]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from src import experiment, quality, reputation
from src.simulation import PARTNERS

OUT = Path(__file__).parent / "results"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = "#1F2933", "#6E7781", "#E5E7EB"
GREEN, LIGHT, GOLD, RED, BLUE = "#0B4F3C", "#5FA88A", "#C99A2E", "#B03A2E", "#2D5C88"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight",
})
SHOWCASE_SEED = 7


def fig_beta_intuition():
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.linspace(0.3, 0.995, 600)
    cases = [
        ("Nouveau partenaire : 2 bons bordereaux", 1 + 2 * 8 * 0.99, 1 + 2 * 8 * 0.01, BLUE),
        ("Partenaire fiable : historique long et bon", 52.5, 0.8, GREEN),
        ("Partenaire en dégradation", 40.0, 13.0, RED),
    ]
    for label, a, b, col in cases:
        y = beta_dist.pdf(x, a, b)
        ax.plot(x, y / y.max(), color=col, lw=2, label=label)
        q = beta_dist.ppf(0.05, a, b)
        ax.axvline(q, color=col, ls=":", lw=1.3)
        ax.annotate(f"q5% = {q:.2f}", (q, 1.02), color=col, fontsize=8, ha="center")
    ax.axvspan(0.85, 1, color=LIGHT, alpha=0.12)
    ax.axvspan(0.70, 0.85, color=GOLD, alpha=0.10)
    ax.axvspan(0.3, 0.70, color=RED, alpha=0.06)
    ax.set_xlabel("Proportion de « bonnes transmissions » (réputation)")
    ax.set_yticks([])
    ax.set_ylim(0, 1.15)
    ax.set_title("La réputation est une loi de probabilité : on décide sur son quantile à 5 %")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(FIG / "fig2_intuition_beta.png")
    plt.close(fig)


def fig_timeline(details):
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharex=True, sharey=True)
    events = {"P3": (19, "début de la dégradation"), "P6": (25, "entrée"),
              "P7": (27, "erreur d'unité"), "P8": (24, "début du mauvais codage")}
    for ax, p in zip(axes.flat, PARTNERS):
        mon = details[p.code]["monitor"]
        ax.axhspan(0.85, 1.0, color=LIGHT, alpha=0.12, lw=0)
        ax.axhspan(0.70, 0.85, color=GOLD, alpha=0.12, lw=0)
        ax.axhspan(0.0, 0.70, color=RED, alpha=0.07, lw=0)
        ax.fill_between(mon.index, mon["q05"], mon["q95"], color=GREEN, alpha=0.18, lw=0)
        ax.plot(mon.index, mon["mean"], color=GREEN, lw=1.4)
        ax.plot(mon.index, mon["q05"], color=INK, lw=1.0, ls="--")
        crit = mon.index[mon["critical"].astype(bool)]
        ax.scatter(crit, mon.loc[crit, "q05"], color=RED, zorder=5, s=18)
        if p.code in events:
            m, txt = events[p.code]
            ax.axvline(m, color=RED, lw=1, ls=":")
            ax.text(m + 0.5, 0.42, txt, color=RED, fontsize=7.5, rotation=90, va="bottom")
        ax.set_title(f"{p.code} {p.label}", fontsize=9.5, loc="left")
        ax.set_ylim(0.4, 1.0)
        ax.grid(axis="y", color=GRID, lw=0.6)
    for ax in axes[1]:
        ax.set_xlabel("Mois")
    for ax in axes[:, 0]:
        ax.set_ylabel("Réputation")
    fig.suptitle("Réputation de chaque partenaire, mois après mois : moyenne (trait plein), "
                 "quantile 5 % (pointillés), bande 5 % à 95 %", fontsize=10.5, x=0.01, ha="left")
    fig.text(0.01, -0.01, "Zones : vert (q5% >= 0,85), orange (0,70 à 0,85), rouge (< 0,70). "
             "Point rouge : incident critique, mise en quarantaine immédiate. Simulation illustrative.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_reputation_partenaires.png")
    plt.close(fig)


def fig_heatmap(details):
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.8), sharex=True)
    for ax, code, title in [(axes[0], "P3", "P3 Bancassureur : dégradation progressive dès le mois 19"),
                            (axes[1], "P8", "P8 Gestionnaire auto : survenance mal codée dès le mois 24")]:
        s = details[code]["scores"][quality.DIMENSIONS].T
        im = ax.imshow(s.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                       extent=[0.5, 36.5, len(s) - 0.5, -0.5])
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels([quality.LABELS[d].split(" (")[0] for d in s.index], fontsize=8)
        ax.set_title(title, loc="left", fontsize=10)
    axes[1].set_xlabel("Mois de transmission du bordereau")
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01)
    cb.set_label("Note de la dimension (1 = aucun défaut)")
    fig.savefig(FIG / "fig4_notes_par_dimension.png")
    plt.close(fig)


def fig_reserve_error(port):
    order = experiment.STRATEGIES
    fig, ax = plt.subplots(figsize=(9, 4.2))
    data = [port.loc[port.strategy == s, "rel_err"] * 100 for s in order]
    bp = ax.boxplot(data, orientation="horizontal", widths=0.55, patch_artist=True, showfliers=False,
                    medianprops={"color": INK})
    colors = [MUTED, RED, GOLD, LIGHT, LIGHT, GREEN]
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.55); patch.set_edgecolor(col)
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels(order)
    ax.axvline(0, color=INK, lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("Erreur sur la provision du portefeuille, en % de la provision réelle")
    ax.set_title("Erreur de provisionnement selon le niveau de traitement des données")
    ax.grid(axis="x", color=GRID, lw=0.6)
    fig.text(0.01, -0.02, f"{port.seed.nunique()} simulations. Boîtes : 25e à 75e centile ; "
             "moustaches : 1,5 fois l'écart interquartile.", fontsize=8, color=MUTED)
    fig.savefig(FIG / "fig5_erreur_provision.png")
    plt.close(fig)


def fig_partner_bias(rec):
    order = experiment.STRATEGIES
    tab = rec.groupby(["partner", "strategy"]).rel_err.mean().unstack()[order] * 100
    labels = [f"{p.code} {p.label}" for p in PARTNERS]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    vals = tab.to_numpy()
    im = ax.imshow(np.clip(vals, -60, 60), cmap="RdBu_r", vmin=-60, vmax=60, aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(v) > 35 else INK)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontsize=9)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
    ax.spines[:].set_visible(False)
    ax.set_title("Biais moyen sur la provision de chaque partenaire (% de la provision réelle)", loc="left")
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
    cb.set_label("Sous-estimation (bleu) / surestimation (rouge), échelle bornée à ±60 %")
    fig.savefig(FIG / "fig6_biais_par_partenaire.png")
    plt.close(fig)


def fig_quarters(q8):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(q8))
    w = 0.26
    for k, (st, col) in enumerate([("Oracle", MUTED), ("Redressement", GOLD), ("Réputation", GREEN)]):
        ax.bar(x + (k - 1) * w, q8[st], w, color=col, alpha=0.85, label=st)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(q8.index)
    ax.set_ylabel("Biais moyen (% de l'ultime)")
    ax.set_xlabel("Trimestre de survenance")
    ax.set_title("P8 : le total est juste, la répartition par survenance ne l'est pas", loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    fig.savefig(FIG / "fig7_P8_biais_par_trimestre.png")
    plt.close(fig)


def main(runs: int):
    results = [experiment.run_one(s) for s in range(runs)]
    rec, port = experiment.summarize(results)

    def stats(df):
        g = df.groupby("strategy").rel_err
        return pd.DataFrame({
            "biais_moyen_pct": g.mean() * 100,
            "ecart_type_pct": g.std() * 100,
            "rmse_pct": g.apply(lambda x: np.sqrt((x**2).mean())) * 100,
            "p05_pct": g.quantile(0.05) * 100,
            "p95_pct": g.quantile(0.95) * 100,
        }).round(1)

    port_stats = stats(port).loc[experiment.STRATEGIES + experiment.SENSITIVITY]
    port_stats.to_csv(OUT / "erreur_portefeuille.csv")
    part = (rec.groupby(["partner", "strategy"]).rel_err
            .agg(biais_pct=lambda x: x.mean() * 100, rmse_pct=lambda x: np.sqrt((x**2).mean()) * 100)
            .round(1).reset_index())
    part.to_csv(OUT / "erreur_par_partenaire.csv", index=False)

    # erreur par trimestre de survenance pour P8 (survenance mal codée)
    q8 = {st: np.nanmean([r["quarters"]["P8"][st] for r in results], axis=0) * 100
          for st in ["Oracle", "Redressement", "Benktander", "Réputation"]}
    q8 = pd.DataFrame(q8, index=[f"T{i + 1}" for i in range(12)]).round(1)
    q8.index.name = "trimestre_survenance"
    q8.to_csv(OUT / "P8_erreur_par_trimestre.csv")
    fig_quarters(q8)

    alert = pd.DataFrame([r["alert"] for r in results])
    reject = pd.DataFrame([r["reject"] for r in results])
    fa = pd.DataFrame([r["false_alerts"] for r in results])
    zs = pd.DataFrame([r["Z"] for r in results])
    detection = {
        "P3_debut": 19, "P3_alerte_mediane": float(alert["P3"].median()),
        "P3_alerte_p10_p90": [float(alert["P3"].quantile(0.1)), float(alert["P3"].quantile(0.9))],
        "P3_detecte_pct": float(alert["P3"].notna().mean() * 100),
        "P3_controle_ponctuel_rejet_pct": float(reject["P3"].notna().mean() * 100),
        "P8_debut": 24, "P8_alerte_mediane": float(alert["P8"].median()),
        "P8_alerte_p10_p90": [float(alert["P8"].quantile(0.1)), float(alert["P8"].quantile(0.9))],
        "P8_detecte_pct": float(alert["P8"].notna().mean() * 100),
        "P8_controle_ponctuel_rejet_pct": float(reject["P8"].notna().mean() * 100),
        "fausses_alertes_pct": (fa.mean() * 100).round(2).to_dict(),
        "Z_moyen_cloture": zs.mean().round(2).to_dict(),
        "runs": runs,
    }
    (OUT / "detection.json").write_text(json.dumps(detection, indent=2, ensure_ascii=False))

    show = experiment.run_one(SHOWCASE_SEED, keep_details=True)["details"]
    lines = []
    for code, months in [("P3", [20, 23, 26, 30]), ("P7", [27]), ("P8", [24, 27]), ("P5", None)]:
        s = show[code]["scores"]
        if months is None:
            months = list(s.index[s["fraicheur"] < 0.5][:2])
        for m in months:
            lines.append(f"- {code}, mois {m} : {quality.explain(s.loc[m])}")
    (OUT / "explications_exemples.md").write_text("\n".join(lines), encoding="utf-8")
    for code in show:
        show[code]["monitor"].round(4).to_csv(OUT / f"reputation_{code}_seed{SHOWCASE_SEED}.csv")

    fig_beta_intuition()
    fig_timeline(show)
    fig_heatmap(show)
    fig_reserve_error(port)
    fig_partner_bias(rec)
    print(port_stats.to_string())
    print(json.dumps(detection, indent=2, ensure_ascii=False))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=300)
    main(ap.parse_args().runs)
