"""Une simulation : corpus, vue structurée, questions L2 et réponses de chaque dispositif."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import corpus as C
from .extraction import ExtractionDraws, extract, induce_schema
from .retrieval import BM25, LSA

P_LEVELS = [.98, .92, .85, .70]
K_LEVELS = [8, 50]
SCAN_RECALL = [.95, .80]
SCAN_FP = .002
NQ = 5
CALIB = 0.0  # code du scénario calibré E1/E2 (faux positifs 2,6 %) dans la colonne param
CALIB_BAS = -1.0  # scénario calibré, faux positifs 0,46 %
NF = len(C.F_NAMES)
FCOLS = [f"f{j}" for j in range(NF)]


def top3(counts: np.ndarray) -> frozenset:
    order = np.lexsort((np.arange(len(counts)), -counts))
    return frozenset(order[:3].tolist())


def true_sim(df_like: pd.DataFrame, i: int, factor_cols) -> np.ndarray:
    """Similarité entre le dossier i et tous les autres, calculée sur les colonnes fournies."""
    b = df_like["branche"].to_numpy(); c = df_like["cause"].to_numpy()
    pa = df_like["pays"].to_numpy(); lit = df_like["litige"].to_numpy().astype(float)
    m = np.log(df_like["montant"].to_numpy().astype(float))
    F = np.nan_to_num(df_like[factor_cols].to_numpy().astype(float))
    valid = lambda x, xi: (x == xi) & (x >= 0) if x.dtype.kind in "iu" else (x == xi)
    s = 3 * valid(b, b[i]) + 3 * (valid(b, b[i]) & valid(c, c[i])) + 1 * valid(pa, pa[i])
    s = s + 1 * np.nan_to_num(np.abs(m - m[i]) < .5) + 1 * np.nan_to_num(lit == lit[i])
    inter = F @ F[i]; union = F.sum(1) + F[i].sum() - inter
    s = s + 2 * np.where(union > 0, inter / np.maximum(union, 1), 0)
    s = s.astype(float); s[i] = -np.inf
    return s


def run_sim(seed: int, n: int = 5000) -> list[dict]:
    rng = np.random.default_rng(seed)
    df, texts = C.generate(n, seed)
    schema = induce_schema(df, rng)
    draws = ExtractionDraws(df, rng)
    scan_u = rng.random((n, NF))
    ext = {p: extract(df, draws, schema, p) for p in P_LEVELS}
    # Scénario calibré sur le volet LLM (ajouté après E1/E2) : schéma retenant tout facteur vu au moins
    # une fois sur 100 synthèses, champs de base à 99,5 %, facteurs : rappel 0,984, faux positifs 4,9 %
    schema_llm = induce_schema(df, np.random.default_rng(seed + 10_000), n_sample=100, min_count=1)
    # Corrigé après relecture : faux positifs mesurés 2,6 % (tous facteurs) et 0,46 % (hors deux facteurs
    # à vocabulaire ambigu) ; la première version utilisait 4,9 %, qui comptait les null comme faux positifs.
    ext[CALIB] = extract(df, draws, schema_llm, .995, f_recall=.984, f_fp=.026)
    ext[CALIB_BAS] = extract(df, draws, schema_llm, .995, f_recall=.984, f_fp=.0046)
    schemas = {p: schema for p in P_LEVELS}; schemas[CALIB] = schema_llm; schemas[CALIB_BAS] = schema_llm
    full = list(range(NF))
    oracle = extract(df, draws, full, 1.0)  # table parfaite, schéma complet : témoin à auditer
    bm25, lsa = BM25(texts), LSA(texts, seed=seed)
    retr = {"BM25": bm25, "RAG vectoriel": lsa}
    out = []

    def rec(q, qi, method, param, success, err=np.nan, docs=0):
        out.append(dict(sim=seed, q=q, qi=qi, method=method, param=param,
                        success=bool(success), err=float(err), docs_read=docs, schema_size=len(schema)))

    B, P, L = df["branche"].to_numpy(), df["pays"].to_numpy(), df["litige"].to_numpy()
    Ftrue = df[FCOLS].to_numpy()

    # ---------- Q_part : part en contentieux d'une sous-population (branche, pays)
    combos = [(b, c) for b in range(len(C.BRANCH_NAMES)) for c in range(len(C.PAYS))
              if ((B == b) & (P == c)).sum() >= 15]
    for qi, idx in enumerate(rng.choice(len(combos), NQ, replace=False)):
        b, c = combos[idx]
        truth = L[(B == b) & (P == c)].mean()
        query = f"Parmi les sinistres {C.BRANCH_NAMES[b]} en {C.PAYS[c]}, quelle part a donné lieu à un contentieux ?"
        for name, r in retr.items():
            for k in K_LEVELS:
                got = r.search(query, k)
                sel = got[(B[got] == b) & (P[got] == c)]
                est = L[sel].mean() if len(sel) else np.nan
                rec("Q_part", qi, name, k, np.abs(est - truth) <= .05 if len(sel) else False, est - truth, k)
        for p, e in ext.items():
            m = (e["branche"] == b) & (e["pays"] == c) & e["litige"].notna()
            est = e.loc[m, "litige"].mean() if m.any() else np.nan
            ok = m.any() and abs(est - truth) <= .05
            rec("Q_part", qi, "Vue structurée", p, ok, est - truth)
            rec("Q_part", qi, "Double vue", p, ok, est - truth)
        m = (oracle["branche"] == b) & (oracle["pays"] == c)
        est = oracle.loc[m, "litige"].mean()
        rec("Q_part", qi, "Oracle", 1.0, abs(est - truth) <= .05, est - truth)

    # ---------- Q_top3 : 3 facteurs les plus fréquents parmi les dossiers d'une branche en contentieux
    good = []
    for b in range(len(C.BRANCH_NAMES)):
        cnt = Ftrue[(B == b) & L].sum(0)
        srt = np.sort(cnt)[::-1]
        if srt[2] - srt[3] >= 2:
            good.append(b)
    for qi, b in enumerate(rng.choice(good, min(NQ, len(good)), replace=False)):
        truth = top3(Ftrue[(B == b) & L].sum(0))
        query = f"Quels sont les facteurs les plus fréquents parmi les sinistres {C.BRANCH_NAMES[b]} en contentieux ?"
        for name, r in retr.items():
            for k in K_LEVELS:
                got = r.search(query, k)
                sel = got[(B[got] == b) & L[got]]
                ok = len(sel) > 0 and top3(Ftrue[sel].sum(0)) == truth
                rec("Q_top3", qi, name, k, ok, docs=k)
        for p, e in ext.items():
            m = ((e["branche"] == b) & (e["litige"] == 1)).to_numpy()
            cnt = np.nan_to_num(e.loc[m, FCOLS].to_numpy(dtype=float)).sum(0)
            ok = m.any() and top3(cnt) == truth
            rec("Q_top3", qi, "Vue structurée", p, ok)
            rec("Q_top3", qi, "Double vue", p, ok)
        m = ((oracle["branche"] == b) & (oracle["litige"] == 1)).to_numpy()
        rec("Q_top3", qi, "Oracle", 1.0, top3(oracle.loc[m, FCOLS].to_numpy(dtype=float).sum(0)) == truth)

    # ---------- Q_rare : comptage d'un facteur rare dans une branche
    pairs = [(b, j) for b in range(len(C.BRANCH_NAMES)) for j in range(C.N_FREQ, NF)]
    for qi, idx in enumerate(rng.choice(len(pairs), NQ, replace=False)):
        b, j = pairs[idx]
        truth = int(((B == b) & Ftrue[:, j]).sum())
        tol = max(2, .1 * truth)
        query = f"Combien de sinistres {C.BRANCH_NAMES[b]} mentionnent : {C.F_NAMES[j]} ?"
        for name, r in retr.items():
            for k in K_LEVELS:
                got = r.search(query, k)
                est = int(((B[got] == b) & Ftrue[got, j]).sum())
                rec("Q_rare", qi, name, k, abs(est - truth) <= tol, est - truth, k)
        for p, e in ext.items():
            if j in schemas[p]:
                est = int(((e["branche"] == b) & (e[f"f{j}"] == 1)).sum())
                rec("Q_rare", qi, "Vue structurée", p, abs(est - truth) <= tol, est - truth)
                for rr in SCAN_RECALL:
                    rec("Q_rare", qi, f"Double vue (balayage {rr:.2f})", p, abs(est - truth) <= tol, est - truth)
            else:
                rec("Q_rare", qi, "Vue structurée", p, False)  # champ absent : pas de réponse
                sub = (e["branche"] == b).to_numpy()
                for rr in SCAN_RECALL:
                    det = np.where(Ftrue[:, j], scan_u[:, j] < rr, scan_u[:, j] < SCAN_FP)
                    est = int((sub & det).sum())
                    rec("Q_rare", qi, f"Double vue (balayage {rr:.2f})", p, abs(est - truth) <= tol,
                        est - truth, int(sub.sum()))
        est = int(((oracle["branche"] == b) & (oracle[f"f{j}"] == 1)).sum())
        rec("Q_rare", qi, "Oracle", 1.0, abs(est - truth) <= tol, est - truth)

    # ---------- Q_prec : 10 précédents les plus comparables
    for qi, i in enumerate(rng.choice(n, NQ, replace=False)):
        s_true = true_sim(df, i, FCOLS)
        best = np.sort(s_true)[::-1][:10].mean()
        def ratio(sel):
            return s_true[sel].mean() / best
        r_l = ratio(lsa.similar_to(i, 10))
        rec("Q_prec", qi, "RAG vectoriel", 10, r_l >= .9, r_l, 10)
        got = [x for x in bm25.search(texts[i], 11) if x != i][:10]
        r_b = ratio(np.array(got))
        rec("Q_prec", qi, "BM25", 10, r_b >= .9, r_b, 10)
        cos = lsa.E @ lsa.E[i]
        for p, e in ext.items():
            s_ext = true_sim(e, i, [f"f{j}" for j in schemas[p]])
            sel = np.argsort(-s_ext, kind="stable")[:10]
            r_s = ratio(sel)
            rec("Q_prec", qi, "Vue structurée", p, r_s >= .9, r_s)
            s_d = s_ext + 2 * cos
            s_d[i] = -np.inf
            sel = np.argsort(-s_d, kind="stable")[:10]
            r_d = ratio(sel)
            rec("Q_prec", qi, "Double vue", p, r_d >= .9, r_d)
        s_o = true_sim(oracle, i, FCOLS)
        r_o = ratio(np.argsort(-s_o, kind="stable")[:10])
        rec("Q_prec", qi, "Oracle", 1.0, r_o >= .9, r_o)
    return out
