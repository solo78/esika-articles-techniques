"""Volet LLM (appels payants) : induction de schéma, extraction réelle, questions L2 réelles.

Prérequis : export ANTHROPIC_API_KEY=... (hors dépôt). Garde budgétaire : 5 USD.
Les réponses sont mises en cache dans results/llm/ : relancer ne refait pas les appels déjà faits.
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from src import corpus as C
from src.llm import Client, parse_json
from src.retrieval import LSA

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "llm"
OUT.mkdir(parents=True, exist_ok=True)
HAIKU, SONNET = "claude-haiku-4-5", "claude-sonnet-4-6"
N, SEED = 800, 1001
cli = Client(OUT / "journal_appels.jsonl", budget_usd=5.0)


def cp(k, n):
    return (beta.ppf(.025, k, n - k + 1) if k else 0.0, beta.ppf(.975, k + 1, n - k) if k < n else 1.0)


def norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s).lower()).encode("ascii", "ignore").decode()
    return s.replace("_", " ").replace("'", " ").strip()


def cached(name, fn):
    p = OUT / name
    if p.exists():
        return json.loads(p.read_text())
    v = fn()
    p.write_text(json.dumps(v, ensure_ascii=False, indent=1))
    return v


df, texts = C.generate(N, SEED)
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------ E1 induction de schéma
SYS_AN = ("Tu es analyste de données dans une compagnie d'assurance. On te donne des synthèses de "
          "sinistres graves au format courriel. Propose les champs d'un modèle de données structuré qui "
          "capture les informations récurrentes et utiles à la décision (gestion, provisionnement, contentieux). "
          "Réponds uniquement par une liste JSON d'objets {\"nom\": snake_case, \"type\": texte|entier|decimal|booleen|date|liste, "
          "\"description\": une phrase}.")
SYS_ME = ("Tu es architecte de données. On te donne plusieurs propositions de champs issues d'analyses de lots "
          "différents des mêmes archives de sinistres. Consolide-les en un schéma relationnel unique d'au plus 30 champs : "
          "fusionne les doublons, résous les incohérences, garde ce qui est récurrent et utile à la décision. "
          "Réponds uniquement en JSON : {\"champs\": [{\"nom\", \"type\", \"description\"}], \"ddl\": \"CREATE TABLE ...\"}.")
KEYWORDS = {  # fixés avant exécution
    "branche": ["branche", "line of business", "ligne d activite", "ligne de produit"],
    "pays": ["pays", "juridiction", "country"],
    "cause": ["cause", "origine"],
    "annee": ["annee", "date", "survenance"],
    "montant": ["dommage", "montant estime", "estimation", "montant du sinistre", "montant sinistre"],
    "reserve": ["reserve"],
    "contentieux": ["contentieux", "litige", "judiciaire", "procedure", "amiable"],
    "retard de déclaration": ["retard", "tardi"],
    "pluralité d'assureurs": ["coassurance", "co assurance", "assureurs", "porteurs"],
    "expertise contradictoire": ["expert"],
    "sous-traitant impliqué": ["sous traitant", "sous-traitant"],
    "clause d'exclusion ambiguë": ["exclusion"],
    "dommages corporels": ["corporel", "blesse"],
    "perte d'exploitation": ["exploitation", "interruption d activite", "arret de production"],
    "pluralité de victimes": ["tiers leses", "pluralite de victimes", "victimes multiples", "nombre de victimes",
                              "multiples tiers", "action groupee"],
    "sous-assurance": ["sous assurance", "sous-assurance", "proportionnelle", "valeur assuree"],
    "sanctions internationales": ["sanction"],
    "faute inexcusable": ["faute inexcusable"],
    "fraude suspectée": ["fraud"],
}


def e1():
    res = []
    for rep in range(3):
        idx = rng.choice(N, 100, replace=False)
        props = []
        for lot in range(4):
            sub = idx[lot * 25:(lot + 1) * 25]
            user = "\n\n---\n\n".join(texts[i] for i in sub)
            txt, _ = cli.call(f"E1_analyzer_r{rep}_l{lot}", HAIKU, SYS_AN, user, 4000)
            props.append(parse_json(txt))
        user = "\n\n".join(f"Proposition {k+1} :\n{json.dumps(p, ensure_ascii=False)}" for k, p in enumerate(props))
        txt, _ = cli.call(f"E1_merger_r{rep}", SONNET, SYS_ME, user, 8000)
        schema = parse_json(txt)
        res.append(dict(rep=rep, propositions=props, schema=schema))
    return res


E1 = cached("e1_schemas.json", e1)
cov_rows = []
for r in E1:
    champs = r["schema"]["champs"]
    blob = [norm(f"{c['nom']} {c.get('description', '')}") for c in champs]
    for attr, kws in KEYWORDS.items():
        hit = [c["nom"] for c, b in zip(champs, blob) if any(norm(k) in b for k in kws)]
        cov_rows.append(dict(rep=r["rep"], attribut=attr, couvert=bool(hit), champs=";".join(hit),
                             n_champs_schema=len(champs)))
cov = pd.DataFrame(cov_rows)
cov.to_csv(OUT / "e1_couverture.csv", index=False)

# ------------------------------------------------------------------ E2 extraction avec schéma de référence
CAUSES = [c for b in C.BRANCH_NAMES for c in C.BRANCHES[b]]
FREQ = [f[0] for f in C.FACTEURS]
SYS_EX = ("Tu remplis une table de sinistres à partir de synthèses au format courriel. Pour chaque dossier, "
          "renvoie un objet JSON avec exactement ces clés :\n"
          f"- id : référence du dossier (SIN-...)\n- branche : une valeur parmi {C.BRANCH_NAMES}\n"
          f"- pays : une valeur parmi {C.PAYS}\n- cause : une valeur parmi {CAUSES}\n"
          "- annee : entier\n- montant_dommage_eur : nombre en euros\n- reserve_eur : nombre en euros\n"
          "- contentieux : booléen, vrai seulement si une procédure judiciaire est engagée\n"
          + "".join(f"- f{j} : booléen, vrai si le texte mentionne : {n}\n" for j, n in enumerate(FREQ))
          + "Si une information est absente, mets null. Réponds uniquement par une liste JSON, un objet par dossier, dans l'ordre.")


def e2():
    rows = []
    for start in range(0, N, 10):
        user = "\n\n=====\n\n".join(texts[i] for i in range(start, min(N, start + 10)))
        txt, stop = cli.call(f"E2_extraction_{start}", HAIKU, SYS_EX, user, 5000)
        try:
            rows.extend(parse_json(txt))
        except Exception as ex:  # journalisé, compté comme manquant
            rows.append(dict(erreur_parse=str(ex), lot=start))
    return rows


E2 = cached("e2_extraction.json", e2)
ext = pd.DataFrame([r for r in E2 if "id" in r]).drop_duplicates("id").set_index("id")
truth = df.set_index("id")
fields = []
for did, t in truth.iterrows():
    e = ext.loc[did] if did in ext.index else None
    g = lambda k: None if e is None or pd.isna(e.get(k)) else e.get(k)
    def num_ok(v, ref):
        try:
            return v is not None and abs(float(v) - ref) <= .05 * ref
        except (TypeError, ValueError):
            return False
    checks = {
        "branche": g("branche") is not None and norm(g("branche")) == norm(C.BRANCH_NAMES[t.branche]),
        "pays": g("pays") is not None and norm(g("pays")) == norm(C.PAYS[t.pays]),
        "cause": g("cause") is not None and norm(g("cause")) == norm(C.BRANCHES[C.BRANCH_NAMES[t.branche]][t.cause]),
        "annee": g("annee") is not None and int(g("annee")) == int(t.annee),
        "montant": num_ok(g("montant_dommage_eur"), t.montant),
        "reserve": num_ok(g("reserve_eur"), t.reserve),
        "contentieux": g("contentieux") is not None and bool(g("contentieux")) == bool(t.litige),
    }
    for j in range(len(FREQ)):
        v = g(f"f{j}")
        # null sur un booléen = « non mentionné » = faux (même convention que le SQL de E3, fillna(False)).
        # Correction après relecture : la première version comptait null comme erreur (faux positif apparent 4,9 %).
        checks[f"f{j}"] = (bool(v) if v is not None else False) == bool(t[f"f{j}"])
    for k, ok in checks.items():
        vrai = bool(t[k]) if k.startswith("f") else None
        brut = None if e is None else g({"montant": "montant_dommage_eur", "reserve": "reserve_eur"}.get(k, k))
        fields.append(dict(id=did, champ=k, correct=ok, vrai_facteur=vrai, manquant=brut is None,
                           predit_vrai=(brut is True or brut == 1) if k.startswith("f") else None))
fd = pd.DataFrame(fields)
fd.to_csv(OUT / "e2_champs.csv", index=False)
k, n = int(fd.correct.sum()), len(fd)
fac = fd[fd.champ.str.startswith("f")]
pos, neg = fac[fac.vrai_facteur == True], fac[fac.vrai_facteur == False]
e2_summary = dict(
    n_dossiers=N, n_champs=n, dossiers_extraits=int(ext.shape[0]),
    exactitude_globale=k / n, ic95=cp(k, n),
    exactitude_par_champ=fd.groupby("champ").correct.mean().round(4).to_dict(),
    taux_manquants=float(fd.manquant.mean()),
    facteurs_rappel=float(pos.correct.mean()), facteurs_rappel_ic95=cp(int(pos.correct.sum()), len(pos)),
    facteurs_faux_positifs=float(1 - neg.correct.mean()),
    facteurs_faux_positifs_ic95=cp(int((~neg.correct).sum()), len(neg)),
    facteurs_null_sur_absents=float(neg.manquant.mean()),
    facteurs_faux_positifs_hors_f2_f7=float(1 - neg[~neg.champ.isin(["f2", "f7"])].correct.mean()),
    facteurs_faux_positifs_hors_f2_f7_ic95=cp(int((~neg[~neg.champ.isin(["f2", "f7"])].correct).sum()),
                                              int((~neg.champ.isin(["f2", "f7"])).sum())),
    faux_positifs_par_facteur=(1 - neg.groupby("champ").correct.mean()).round(4).to_dict(),
    n_facteurs_presents=len(pos), n_facteurs_absents=len(neg))

# ------------------------------------------------------------------ E3 questions L2 réelles
lsa = LSA(texts, seed=SEED)
ALLF = C.F_NAMES
SYS_RAG = ("Tu réponds à une question d'un gestionnaire de sinistres en t'appuyant UNIQUEMENT sur les extraits fournis. "
           "Réponds uniquement en JSON.")


def sql_ext():
    e = ext.copy()
    e["b"] = e["branche"].map(lambda x: next((i for i, b in enumerate(C.BRANCH_NAMES) if norm(b) == norm(x)), -1))
    return e


def e3():
    out = []
    for b, bn in enumerate(C.BRANCH_NAMES):
        q = f"Parmi les sinistres {bn}, quelle part a donné lieu à un contentieux ?"
        got = lsa.search(q, 8)
        ctx = "\n\n=====\n\n".join(texts[i] for i in got)
        txt, _ = cli.call(f"E3_part_{b}", HAIKU, SYS_RAG,
                          f"Extraits :\n{ctx}\n\nQuestion : {q}\nFormat : {{\"part\": nombre entre 0 et 1, \"justification\": texte court}}", 800)
        out.append(dict(type="Q_part", b=b, reponse=txt, docs=got.tolist()))
        q = f"Quels sont les 3 facteurs les plus fréquents parmi les sinistres {bn} en contentieux ?"
        got = lsa.search(q, 8)
        ctx = "\n\n=====\n\n".join(texts[i] for i in got)
        txt, _ = cli.call(f"E3_top3_{b}", HAIKU, SYS_RAG,
                          f"Extraits :\n{ctx}\n\nQuestion : {q}\nChoisis parmi : {ALLF}\n"
                          "Format : {\"facteurs\": [3 noms exacts de la liste], \"justification\": texte court}", 800)
        out.append(dict(type="Q_top3", b=b, reponse=txt, docs=got.tolist()))
    return out


E3 = cached("e3_reponses.json", e3)
e = sql_ext()
B, L = df.branche.to_numpy(), df.litige.to_numpy()
Ftrue = df[[f"f{j}" for j in range(len(ALLF))]].to_numpy()
rows = []
for r in E3:
    b = r["b"]
    if r["type"] == "Q_part":
        vrai = L[B == b].mean()
        try:
            rag = float(parse_json(r["reponse"])["part"])
        except Exception:
            rag = np.nan
        m = (e.b == b) & e.contentieux.notna()
        sql = e.loc[m, "contentieux"].astype(bool).mean()
        rows.append(dict(type="Q_part", branche=C.BRANCH_NAMES[b], vrai=vrai, rag_llm=rag, sql_extrait=sql,
                         contentieux_dans_top8=int(L[r["docs"]].sum()), meme_branche_dans_top8=int((B[r["docs"]] == b).sum()),
                         succes_rag=abs(rag - vrai) <= .05, succes_sql=abs(sql - vrai) <= .05))
    else:
        cnt = Ftrue[(B == b) & L].sum(0)
        order = np.lexsort((np.arange(len(cnt)), -cnt))
        vrai = {ALLF[i] for i in order[:3]}
        ecart34 = int(np.sort(cnt)[::-1][2] - np.sort(cnt)[::-1][3])
        try:
            rag = set(parse_json(r["reponse"])["facteurs"])
        except Exception:
            rag = set()
        m = ((e.b == b) & (e.contentieux == True)).to_numpy()
        sc = np.array([e.loc[m, f"f{j}"].fillna(False).astype(bool).sum() for j in range(len(FREQ))] + [0, 0, 0])
        o2 = np.lexsort((np.arange(len(sc)), -sc))
        sql = {ALLF[i] for i in o2[:3]}
        rows.append(dict(type="Q_top3", branche=C.BRANCH_NAMES[b], vrai=";".join(sorted(vrai)),
                         rag_llm=";".join(sorted(rag)), sql_extrait=";".join(sorted(sql)), ecart_3e_4e=ecart34,
                         succes_rag=rag == vrai, succes_sql=sql == vrai))
e3df = pd.DataFrame(rows)
from scipy.stats import binomtest
def mcnemar(t):
    x = e3df[e3df.type == t]
    a, b = int((x.succes_sql & ~x.succes_rag).sum()), int((~x.succes_sql & x.succes_rag).sum())
    return dict(sql_seul=a, rag_seul=b, p_exact=binomtest(a, a + b, .5).pvalue if a + b else None)
e3df.to_csv(OUT / "e3_questions.csv", index=False)

j = pd.read_json(OUT / "journal_appels.jsonl", lines=True)
summary = dict(
    e1=dict(repetitions=len(E1), taille_schemas=[len(r["schema"]["champs"]) for r in E1],
            couverture_par_attribut=cov.groupby("attribut").couvert.sum().astype(int).to_dict()),
    e2=e2_summary,
    e3={t: dict(n=int((e3df.type == t).sum()), succes_rag=int(e3df[e3df.type == t].succes_rag.sum()),
                succes_sql=int(e3df[e3df.type == t].succes_sql.sum()),
                ic95_rag=cp(int(e3df[e3df.type == t].succes_rag.sum()), int((e3df.type == t).sum())),
                ic95_sql=cp(int(e3df[e3df.type == t].succes_sql.sum()), int((e3df.type == t).sum())),
                mcnemar=mcnemar(t)) for t in ["Q_part", "Q_top3"]},
    appels=dict(n=len(j), cout_total_usd=round(j.cout_usd.sum(), 4), jetons_entree=int(j.jetons_entree.sum()),
                jetons_sortie=int(j.jetons_sortie.sum()), arrets_max_tokens=int((j.stop_reason == "max_tokens").sum()),
                par_modele=j.groupby("modele").cout_usd.sum().round(4).to_dict()))
(OUT / "synthese_llm.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
print(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
print(e3df.to_string())
