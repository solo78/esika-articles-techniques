"""Analyse de toutes les exécutions : écrit results/*.csv, results/*.json et results/figures/*.png.

Toutes les métriques de modèles sont recalculées ici de façon indépendante (audit déterministe),
jamais reprises des fichiers produits par les agents.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta

from src.data import reference_split
from src.gates import REQUIRED_DOC, audit_package
from src.metrics import paired_bootstrap_deviance

ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent / "runs"
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)
EXCLUDE = {"pilote_auto", "pilote_auto2", "test", "auto_r1_avorte"}  # pilotes et tentative interrompue (coût conservé dans le total)

C = {"vf": "#0B4F3C", "vc": "#5FA88A", "or": "#C99A2E", "rouge": "#B03A2E", "gris": "#6E7781"}
NOMS = {"naif_r1": "naïve 1", "naif_r2": "naïve 2", "naif_r3": "naïve 3", "auto_r1": "autonome 1",
        "auto_r2": "autonome 2", "auto_r3": "autonome 3", "auto_opus1": "autonome tout Opus"}


def clopper_pearson(k: int, n: int, a: float = 0.05) -> list[float]:
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return [round(lo, 3), round(hi, 3)]


def doc_check(text: str) -> tuple[bool, list[str]]:
    """Même règle que gates.gate_documentation, appliquée à un texte."""
    t = text.lower()
    miss = [k for k in REQUIRED_DOC if k not in t]
    if len(t) < 2500:
        miss.append("longueur insuffisante")
    return (not miss), miss


def doc_gate_refusals(calls: pd.DataFrame, rid: str) -> int:
    """Refus de la porte documentaire, reconstitués depuis le journal d'appels.

    Dans modeling.py, un refus de cette porte relance le rédacteur sans écrire dans l'historique de la tâche.
    Une tâche de rédaction n'a pas d'autre seconde tentative : deux appels consécutifs au rédacteur
    (sans autre agent entre eux) signalent donc un refus.
    """
    seq = calls.loc[calls.run_id == rid].sort_values("ts").agent.tolist()
    return sum(1 for a, b in zip(seq, seq[1:]) if a == b == "redacteur")


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})


def ledger() -> pd.DataFrame:
    return pd.DataFrame([json.loads(l) for l in (RES / "llm_calls.jsonl").read_text().splitlines() if l.strip()])


# ------------------------------------------------------------------ crews de modélisation
def crew_table(calls: pd.DataFrame) -> pd.DataFrame:
    _, test = reference_split()
    y = test["ClaimNb"].to_numpy(float)
    mu_lgb = np.load(RUNS / "baselines" / "mu_baseline_lgbm.npy")
    mu_glm = np.load(RUNS / "baselines" / "mu_baseline_glm.npy")
    rows = []
    for f in sorted((RES / "crew_runs").glob("*.json")):
        log = json.loads(f.read_text())
        rid = log["run_id"]
        if rid in EXCLUDE:
            continue
        wd = RUNS / rid
        row = {"run_id": rid, "variante": log["variant"], "routage": log["mode"], "statut": log.get("status"),
               "verdict_final": log.get("final_verdict"), "minutes": log.get("minutes"),
               "cout_usd": calls.loc[calls.run_id == rid, "cost_usd"].sum()}
        hist = [h for tl in log["tasks"].values() if isinstance(tl, list) for t in tl for h in t["history"]]
        row["appels_code"] = len(hist)
        row["echecs_execution"] = sum(1 for h in hist if h.get("exec_ok") is False)
        row["refus_portes"] = sum(1 for h in hist if h.get("gate_ok") is False)
        row["refus_porte_documentaire"] = doc_gate_refusals(calls, rid) if log["variant"] == "autonome" else None
        docf = wd / "artifacts" / "model_doc.md"
        row["doc_finale_caracteres"] = len(docf.read_text()) if docf.exists() else 0
        row["doc_finale_passe_porte"] = doc_check(docf.read_text())[0] if docf.exists() else False
        row["escalades"] = len(log.get("escalations", []))
        row["reprises_juge"] = len(log.get("rework", []))
        row["points_juge"] = sum(len(v.get("points", [])) for v in log.get("judge", []))
        tj = wd / "artifacts" / "tuning.json"
        try:
            row["modele_retenu"] = str(json.loads(tj.read_text()).get("best_model"))[:40]
        except Exception:  # noqa: BLE001
            row["modele_retenu"] = None
        audit = audit_package(wd / "artifacts", wd / "data" / "train.parquet", wd / "data" / "test.parquet") \
            if (wd / "artifacts" / "predict.py").exists() else {"errors": ["pas de predict.py"]}
        row["livrable_valide"] = not audit.get("errors") and "test" in audit
        if row["livrable_valide"]:
            t = audit["test"]
            row.update(deviance_x100=t["deviance_x100"], gini=t["gini"], ae_test=t["ae_global"],
                       ecart_max_decile=max(abs(r - 1) for r in t["ae_deciles"]),
                       utilise_id=audit["id_sensitivity"] > 1e-6,
                       exposition_ok=abs(audit["exposure_ratio_median"] - 2) < 1e-3 and audit["exposure_ratio_maxdev"] < 1e-3,
                       ae_train=audit.get("train_ae"))
            ev = wd / "artifacts" / "eval.json"
            try:
                rep = json.loads(ev.read_text())
                row["metriques_annoncees_exactes"] = abs(float(rep["deviance_x100"]) - t["deviance_x100"]) < 0.01
            except Exception:  # noqa: BLE001
                row["metriques_annoncees_exactes"] = False
            mu = np.load(wd / "artifacts" / "_mu_test.npy")
            for name, mu_b in (("lgbm", mu_lgb), ("glm", mu_glm)):
                b = paired_bootstrap_deviance(y, mu, mu_b, n_boot=300, seed=1)
                row[f"diff_vs_{name}"] = b["diff_mean"]
                row[f"ic95_vs_{name}"] = f"[{b['ci95'][0]:.3f} ; {b['ci95'][1]:.3f}]"
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ crew MRM
EXPECTED = {"P0_sain": None, "P1_fuite": {"fuite_cible"}, "P2_exposition": {"exposition"},
            "P3_contamination": {"contamination_test"}, "P4_calibration": {"calibration"},
            "P5_documentation": {"documentation"}, "P6_metriques": {"metriques_inexactes"}}


def mrm_table(calls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in sorted((RES / "mrm_runs").glob("*.json")):
        log = json.loads(f.read_text())
        pkg = log["package"]
        if pkg not in EXPECTED:
            continue
        v = log.get("verdict") or {}
        serious = [c for c in v.get("constats", []) if c["gravite"] in ("bloquant", "important")]
        cats = sorted({c["categorie"] for c in serious})
        exp = EXPECTED[pkg]
        blocking = {c["categorie"] for c in v.get("constats", []) if c["gravite"] == "bloquant"}
        rows.append({
            "run_id": log["run_id"], "dossier": pkg, "decision": v.get("decision"),
            "categories_graves": ", ".join(cats),
            "categories_bloquantes": ", ".join(sorted(blocking)),
            # défaut détecté : sa catégorie figure parmi les constats bloquants ou importants
            "detecte": (bool(exp & set(cats)) if exp else None),
            # défaut détecté ET jugé bloquant ET dossier refusé
            "detecte_bloquant_refuse": (bool(exp & blocking) and v.get("decision") == "refuse" if exp else None),
            # fausse alerte sur le dossier sain : refus ou constat bloquant (les réserves sont vérifiées à part)
            "fausse_alerte": ((v.get("decision") == "refuse" or len(blocking) > 0) if exp is None else None),
            "constats_graves": len(serious),
            "cout_usd": calls.loc[calls.run_id == log["run_id"], "cost_usd"].sum(), "minutes": log["minutes"],
        })
    return pd.DataFrame(rows)


def livrables_table(calls: pd.DataFrame) -> pd.DataFrame:
    """Revue MRM des modèles produits par les crews (chaîne complète sans humain)."""
    rows = []
    for f in sorted((RES / "mrm_runs").glob("mrm_livrable_*.json")):
        log = json.loads(f.read_text())
        v = log.get("verdict") or {}
        rows.append({
            "revue": log["run_id"], "livrable": log["package"], "decision": v.get("decision"),
            "bloquants": ", ".join(sorted({c["categorie"] for c in v.get("constats", []) if c["gravite"] == "bloquant"})),
            "importants": ", ".join(sorted({c["categorie"] for c in v.get("constats", []) if c["gravite"] == "important"})),
            "cout_usd": calls.loc[calls.run_id == log["run_id"], "cost_usd"].sum(), "minutes": log["minutes"],
            "synthese": (v.get("synthese") or "")[:600],
        })
    return pd.DataFrame(rows)


def exposure_structure(crews: pd.DataFrame) -> pd.DataFrame:
    """A/E par tranche d'exposition et sur le 1er décile du nombre attendu, pour tous les modèles.

    Sert à vérifier un constat récurrent de la crew MRM : la non-proportionnalité empirique à
    l'exposition est-elle propre à un modèle ou structurelle au jeu de données ?
    """
    _, test = reference_split()
    y, e = test["ClaimNb"].to_numpy(float), test["Exposure"].to_numpy(float)
    mus = {"temoin_glm": np.load(RUNS / "baselines" / "mu_baseline_glm.npy"),
           "temoin_lgbm": np.load(RUNS / "baselines" / "mu_baseline_lgbm.npy")}
    for rid in crews.run_id:
        f = RUNS / rid / "artifacts" / "_mu_test.npy"
        if f.exists():
            mus[rid] = np.load(f)
    rows = []
    for k, mu in mus.items():
        first = np.array_split(np.argsort(mu, kind="mergesort"), 10)[0]
        r = {"modele": k, "ae_1er_decile_mu": y[first].sum() / mu[first].sum()}
        for lo, hi, lab in ((0, .25, "expo_0_3m"), (.25, .5, "expo_3_6m"), (.5, .75, "expo_6_9m"), (.75, 9, "expo_9m_plus")):
            m = (e > lo) & (e <= hi) if lo else (e <= hi)
            r[f"ae_{lab}"] = y[m].sum() / mu[m].sum()
        rows.append(r)
    return pd.DataFrame(rows)


def deterministic_mrm() -> pd.DataFrame:
    """Ce que les seules vérifications déterministes (sans LLM) détectent sur les dossiers sabotés."""
    rows = []
    truth = json.loads((RUNS / "packages" / "packages_truth.json").read_text())
    for pkg in EXPECTED:
        d = RUNS / "packages" / f"{pkg}_data"
        a = audit_package(RUNS / "packages" / pkg, d / "train.parquet", d / "test.parquet")
        t = a["test"]
        rep = truth[pkg]["reported"]
        flags = set()
        if abs(a["exposure_ratio_median"] - 2) > 1e-3:
            flags.add("exposition")
        if abs(t["ae_global"] - 1) > 0.05 or not 0.97 <= a["train_ae"] <= 1.03:
            flags.add("calibration")
        if abs(rep["deviance_x100"] - t["deviance_x100"]) > 0.01:
            flags.add("metriques_inexactes")
        if a["id_sensitivity"] > 1e-6:
            flags.add("identifiant")
        exp = EXPECTED[pkg]
        # Porte documentaire du projet (gates.gate_documentation), comptée à part : elle n'est pas un audit chiffré
        doc = (RUNS / "packages" / pkg / "model_doc.md").read_text()
        doc_ok, doc_miss = doc_check(doc)
        flags_doc = flags | ({"documentation"} if not doc_ok else set())
        rows.append({"dossier": pkg, "alertes_deterministes": ", ".join(sorted(flags)),
                     "detecte": (bool(exp & flags) if exp else None),
                     "fausse_alerte": (len(flags) > 0 if exp is None else None),
                     "doc_caracteres": len(doc), "porte_documentaire": "OK" if doc_ok else "refus : " + ", ".join(doc_miss),
                     "detecte_avec_porte_doc": (bool(exp & flags_doc) if exp else None),
                     "fausse_alerte_avec_porte_doc": (len(flags_doc) > 0 if exp is None else None)})
        for f in (RUNS / "packages" / pkg).glob("_mu_test.npy"):
            f.unlink()
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def fig_costs(calls, crews):
    ok_runs = crews.run_id.tolist()
    sub = calls[calls.run_id.isin(ok_runs)].merge(crews[["run_id", "variante", "routage"]], on="run_id")
    lab = {("naif", "route"): "Naïve, routée", ("autonome", "route"): "Autonome, routée",
           ("autonome", "tout_opus"): "Autonome, tout Opus"}
    sub["config"] = [lab[(v, r)] for v, r in zip(sub.variante, sub.routage)]
    piv = sub.groupby(["config", "model"]).cost_usd.sum().unstack(fill_value=0)
    n = sub.groupby("config").run_id.nunique()
    piv = piv.div(n, axis=0)
    piv.index = [f"{i} (n={n[i]})" for i in piv.index]
    cols = [c for c in ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"] if c in piv.columns]
    colors = {"claude-haiku-5-5": C["vc"], "claude-sonnet-5-5": C["vf"], "claude-opus-5-5": C["or"]}
    fig, ax = plt.subplots(figsize=(8, 3.6))
    left = np.zeros(len(piv))
    for c in cols:
        ax.barh(piv.index, piv[c], left=left, color=colors[c], label=c.replace("claude-", "").replace("-5-5", " 5.5").title())
        left += piv[c].to_numpy()
    for i, v in enumerate(left):
        ax.text(v + 0.01, i, f"{v:.2f} $".replace(".", ","), va="center", fontsize=9)
    ax.set_xlabel("Coût moyen par exécution (dollars)")
    ax.set_title("Coût d'une exécution de la crew de modélisation, par modèle", loc="left")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_couts.png", dpi=150)
    plt.close(fig)


def fig_perf(crews, base):
    """Écart de déviance au témoin LightGBM, avec IC 95 % du bootstrap apparié (axe partant de zéro)."""
    v = crews[crews.livrable_valide == True].copy()  # noqa: E712
    _, test = reference_split()
    y = test["ClaimNb"].to_numpy(float)
    b = paired_bootstrap_deviance(y, np.load(RUNS / "baselines" / "mu_baseline_glm.npy"),
                                  np.load(RUNS / "baselines" / "mu_baseline_lgbm.npy"), n_boot=300, seed=1)
    kind = ["GLM" if "glm" in str(m).lower() else "GBM" for m in v.modele_retenu]
    labels = [f"{NOMS.get(r, r)} ({k})" for r, k in zip(v.run_id, kind)] + ["témoin GLM fait main"]
    diffs = list(v.diff_vs_lgbm) + [b["diff_mean"]]
    ics = [[float(x) for x in s.strip("[]").split(";")] for s in v.ic95_vs_lgbm] + [list(b["ci95"])]
    cols = [C["vc"] if r == "autonome" else C["or"] for r in v.variante] + [C["vf"]]
    fig, ax = plt.subplots(figsize=(8, 3.9))
    for i, (d, (lo, hi), c) in enumerate(zip(diffs, ics, cols)):
        ax.plot([lo, hi], [i, i], color=c, lw=2)
        ax.plot(d, i, "o", color=c, ms=7)
        ax.text(hi + 0.012, i, f"+{d:.3f}".replace(".", ","), va="center", fontsize=8)
    ax.axvline(0, color=C["gris"], lw=1, ls="--")
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.set_xlim(-0.05, max(h for _, h in ics) + 0.12)
    ax.invert_yaxis()
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=C["vc"], marker="o", label="Crew autonome"),
                       Line2D([], [], color=C["or"], marker="o", label="Crew naïve"),
                       Line2D([], [], color=C["vf"], marker="o", label="Témoin fait main")],
              frameon=False, fontsize=8, loc="upper right")
    ax.set_xlabel("Écart de déviance x100 au témoin LightGBM (positif = moins bon), IC 95 %")
    ax.set_title("Performance hors échantillon des modèles produits par les crews", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_performance.png", dpi=150)
    plt.close(fig)
    return b


def fig_mrm(mrm, det):
    pk = list(EXPECTED)
    reps = mrm.groupby("dossier")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    llm_rate, det_rate = [], []
    for p in pk:
        g = mrm[mrm.dossier == p]
        if p == "P0_sain":
            llm_rate.append(1 - g.fausse_alerte.mean() if len(g) else np.nan)
            det_rate.append(1 - float(det.loc[det.dossier == p, "fausse_alerte"].iloc[0]))
        else:
            llm_rate.append(g.detecte.mean() if len(g) else np.nan)
            det_rate.append(float(det.loc[det.dossier == p, "detecte"].iloc[0]))
    x = np.arange(len(pk))
    ax.bar(x - 0.2, det_rate, 0.38, color=C["gris"], label="Audit chiffré déterministe (sans IA)")
    ax.bar(x + 0.2, llm_rate, 0.38, color=C["vf"], label="Crew MRM autonome (part des répétitions)")
    ax.set_xticks(x, [p.replace("_", "\n") for p in pk], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Taux de détection\n(P0 : absence de fausse alerte)")
    ax.set_title("Défauts injectés : ce que voit chaque niveau de contrôle", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_detection_mrm.png", dpi=150)
    plt.close(fig)


def main():
    calls = ledger()
    base = {k: v for k, v in json.loads((RES / "baselines.json").read_text()).items()}
    crews = crew_table(calls)
    crews.to_csv(RES / "crew_runs.csv", index=False)
    mrm = mrm_table(calls)
    mrm.to_csv(RES / "mrm_runs.csv", index=False)
    det = deterministic_mrm()
    det.to_csv(RES / "mrm_deterministe.csv", index=False)
    by_agent = calls[~calls.run_id.isin(EXCLUDE)].groupby(["agent", "model"]).agg(
        appels=("cost_usd", "size"), cout_usd=("cost_usd", "sum"), jetons_sortie=("out", "sum"),
        jetons_reflexion=("thinking", "sum"), cache_lu=("cache_r", "sum")).reset_index()
    by_agent.to_csv(RES / "couts_par_agent.csv", index=False)
    summary = {"cout_total_usd": float(calls.cost_usd.sum()), "nb_appels": int(len(calls)),
               "cout_pilote_exclu_usd": float(calls[calls.run_id.isin(EXCLUDE)].cost_usd.sum())}
    if len(crews):
        g = crews.groupby(["variante", "routage"])
        summary["crews"] = g.agg(n=("run_id", "size"), cout_moyen=("cout_usd", "mean"),
                                 minutes_moy=("minutes", "mean"), valides=("livrable_valide", "sum")).reset_index().to_dict("records")
        fig_costs(calls, crews)
        b_glm = fig_perf(crews, base)
        summary["temoin_glm_vs_lgbm"] = {"diff": b_glm["diff_mean"], "ic95": list(b_glm["ci95"])}
        auto = crews[crews.variante == "autonome"]
        summary["refus_porte_documentaire"] = {"total": int(auto.refus_porte_documentaire.sum()),
                                               "par_execution": dict(zip(auto.run_id, auto.refus_porte_documentaire.astype(int)))}
        summary["doc_finale"] = dict(zip(crews.run_id, zip(crews.doc_finale_caracteres, crews.doc_finale_passe_porte)))
        # Routage : comparaison des coûts (une seule exécution tout Opus : n = 1)
        routed = auto[auto.routage == "route"].run_id.tolist()
        opus = auto[auto.routage == "tout_opus"].run_id.tolist()
        c = calls[calls.run_id.isin(routed + opus)]
        hors_red = c[c.agent != "redacteur"].groupby("run_id").cost_usd.sum()
        ext = c[c.agent == "extracteur"].groupby("run_id").cost_usd.sum()
        summary["routage"] = {
            "cout_moyen_route": float(auto[auto.routage == "route"].cout_usd.mean()),
            "cout_tout_opus": float(auto[auto.routage == "tout_opus"].cout_usd.mean()),
            "hors_redacteur_route_moyen": float(hors_red[routed].mean()),
            "hors_redacteur_tout_opus": float(hors_red[opus].mean()),
            "ratio_hors_redacteur": float(hors_red[opus].mean() / hors_red[routed].mean()),
            "extraction_route_moyen": float(ext[routed].mean()), "extraction_tout_opus": float(ext[opus].mean()),
            "redacteur_tout_opus": float(c[(c.agent == "redacteur") & c.run_id.isin(opus)].cost_usd.sum()),
            "n_tout_opus": len(opus)}
        _, test = reference_split()
        summary["part_test_exposition_moins_3_mois"] = float((test.Exposure <= 0.25).mean())
    if len(mrm):
        tab = mrm.groupby("dossier").agg(n=("run_id", "size"), detecte=("detecte", "mean"),
                                         detecte_bloquant_refuse=("detecte_bloquant_refuse", "mean"),
                                         fausse_alerte=("fausse_alerte", "mean"), cout=("cout_usd", "mean"))
        summary["mrm"] = tab.reset_index().to_dict("records")
        defect = mrm[mrm.dossier != "P0_sain"]
        summary["mrm_detection_globale"] = float(defect.detecte.mean())
        summary["mrm_detection_bloquant_refuse"] = float(defect.detecte_bloquant_refuse.mean())
        summary["mrm_p0_refus_ou_bloquant"] = float(mrm[mrm.dossier == "P0_sain"].fausse_alerte.mean())
        k, n = int(defect.detecte_bloquant_refuse.sum()), int(len(defect))
        summary["mrm_detection_ic95_clopper_pearson"] = {"k": k, "n": n, "ic95": clopper_pearson(k, n)}
        p0 = mrm[mrm.dossier == "P0_sain"]
        k0 = int(p0.fausse_alerte.sum())
        summary["mrm_p0_fausse_alerte_ic95"] = {"k": k0, "n": int(len(p0)), "ic95": clopper_pearson(k0, len(p0))}
        # Les contrôles déterministes sont reproductibles à l'identique : on compte les types de défauts
        # (pas de test statistique entre répétitions, qui n'aurait pas de sens pour eux).
        det_d = det[det.dossier != "P0_sain"]
        summary["types_defauts_detectes"] = {
            "crew_mrm_toutes_repetitions": int(defect.groupby("dossier").detecte.all().sum()),
            "controles_deterministes": int(det_d.detecte.astype(bool).sum()),
            "controles_deterministes_avec_porte_doc": int(det_d.detecte_avec_porte_doc.astype(bool).sum()),
            "fausse_alerte_p0_avec_porte_doc": bool(det.loc[det.dossier == "P0_sain", "fausse_alerte_avec_porte_doc"].iloc[0]),
            "sur": int(len(det_d))}
        fig_mrm(mrm, det)
    if len(crews):
        exposure_structure(crews).round(3).to_csv(RES / "exposition_structurelle.csv", index=False)
    liv = livrables_table(calls)
    if len(liv):
        liv.to_csv(RES / "mrm_livrables.csv", index=False)
        summary["mrm_livrables"] = liv.to_dict("records")
        if len(mrm):
            # Modèles sans défaut injecté : dossier sain P0 et livrables des crews
            sans_defaut = list(mrm[mrm.dossier == "P0_sain"].decision) + list(liv.decision)
            k = sum(d == "refuse" for d in sans_defaut)
            summary["refus_modeles_sans_defaut_injecte"] = {"k": k, "n": len(sans_defaut),
                                                            "ic95": clopper_pearson(k, len(sans_defaut))}
    (RES / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    print(json.dumps(summary, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
