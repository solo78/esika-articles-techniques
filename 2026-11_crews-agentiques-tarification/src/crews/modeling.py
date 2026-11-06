"""Crew de modélisation autonome (fréquence sinistres RC auto, freMTPL2freq).

Deux variantes :
  - "autonome" : le jugement humain du papier est remplacé par un méthodologiste (Opus),
                 des portes de qualité déterministes, un orchestrateur d'escalade (Opus)
                 et un juge (Opus) qui peut renvoyer des tâches avec consignes.
  - "naif"     : les mêmes agents, sans méthodologiste, sans portes, sans juge : c'est le
                 papier auquel on a simplement retiré l'humain (seules les erreurs
                 d'exécution sont renvoyées à l'agent).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from .. import gates
from ..data import RAW, SAMPLE_FRAC, SPLIT_SEED, TEST_SIZE
from ..framework import Agent, Runner, Task
from ..llm import LLM
from ..tools import ASOP_GRID, eda_tool

ENV_NOTE = f"""
## Environnement d'exécution
- Python 3.13, pandas 3 (les colonnes texte ont le type « str », pas « object »), numpy 2, scikit-learn 1.8,
  lightgbm 4, xgboost 3, statsmodels, scipy, joblib. Pas d’accès réseau. 2 cœurs, 7 Go de mémoire au total :
  évite les matrices denses géantes (one-hot dense sur des centaines de milliers de lignes, copies multiples).
- Le script est lancé depuis le dossier de travail de l'exécution. Fichier brut : {RAW}
- Dossiers : data/ (jeux de données), artifacts/ (livrables du modèle).
- Chaque script doit tenir en moins de 20 minutes : grilles d'hyperparamètres réduites.
- Contrat du dossier de modèle (artifacts/) :
  * features.py : fonction prepare(df) -> DataFrame des variables explicatives, sans état appris ;
    df ne contient PAS la colonne ClaimNb au moment de tarifer.
  * model.joblib (ou autre fichier) : le modèle entraîné.
  * predict.py : fonction predict(df) -> numpy array du NOMBRE ATTENDU de sinistres de chaque police
    sur sa période d'exposition ; doit fonctionner sur un df sans ClaimNb ; charge ses fichiers par
    chemin relatif à predict.py (Path(__file__).parent) et importe features.py de la même façon.
"""

TASKS_TEXT = {
    "extraction": (
        "Charger le fichier brut (parquet). Créer data/train.parquet et data/test.parquet par "
        f"sklearn.model_selection.train_test_split(df, test_size={TEST_SIZE}, random_state={SPLIT_SEED}) "
        "appliqué au fichier complet, sans aucun filtrage préalable. Créer data/train_sample.parquet, "
        f"échantillon aléatoire de {int(SAMPLE_FRAC * 100)} % du train (random_state=0) pour le réglage.",
        "Les trois fichiers parquet et un rapport (tailles, fréquence observée de chaque jeu)."),
    "eda": (
        "Analyser le profil exploratoire ci-dessous, produit par l'outil EDA sur le train, et en tirer "
        "les constats utiles à un modèle de fréquence : anomalies, valeurs extrêmes, déséquilibres, "
        "variables suspectes, traitements à envisager.",
        "Un rapport structuré de constats chiffrés (pas de code)."),
    "features": (
        "Écrire artifacts/features.py avec prepare(df) pour un modèle de fréquence de sinistres. "
        "Vérifier le module sur data/train.parquet (sans ClaimNb) avant de terminer.",
        "artifacts/features.py opérationnel et un rapport des transformations."),
    "tuning": (
        "Sélectionner le modèle et ses hyperparamètres sur data/train_sample.parquet par validation "
        "croisée, en utilisant prepare(df) de artifacts/features.py. La cible est le nombre de "
        "sinistres ClaimNb observé sur l'exposition Exposure. Écrire artifacts/tuning.json avec les "
        "clés best_model, best_params, cv_results (score de chaque candidat) et metric.",
        "artifacts/tuning.json et un rapport comparant les candidats."),
    "training": (
        "Entraîner le modèle retenu (artifacts/tuning.json) sur tout data/train.parquet, sauvegarder "
        "le modèle et écrire artifacts/predict.py selon le contrat. Ne jamais utiliser data/test.parquet. "
        "Vérifier predict(df) sur un extrait du train sans ClaimNb.",
        "Le modèle entraîné, artifacts/predict.py et un rapport d'entraînement."),
    "evaluation": (
        "Évaluer le modèle sur data/test.parquet avec predict(df) appliqué au jeu sans ClaimNb. "
        "Écrire artifacts/eval.json avec : deviance_x100 = 200 x moyenne sur les polices de "
        "[y log(y/mu) - (y - mu)] (terme y log(y/mu) nul si y = 0), y = ClaimNb brut, mu = predict ; "
        "gini = Gini ordonné (polices triées par fréquence prédite mu/Exposure croissante, courbe "
        "exposition cumulée / sinistres cumulés, gini = 1 - 2 x aire sous la courbe) ; "
        "ae_global = somme(y) / somme(mu) ; ae_deciles = A/E par décile de fréquence prédite.",
        "artifacts/eval.json et un rapport commentant les métriques."),
}


def make_agents() -> dict[str, Agent]:
    return {
        "extraction": Agent("extracteur", "analyste de données", "extraire et découper les données sans fuite",
                            "rigoureux sur la traçabilité des jeux de données", "leger"),
        "eda": Agent("analyste_eda", "data scientist chargé de l'analyse exploratoire",
                     "repérer tout ce qui peut biaiser un modèle de fréquence", "habitué aux données d'assurance auto",
                     "leger"),
        "methodologue": Agent("methodologue", "actuaire senior en tarification non vie",
                              "définir la spécification méthodologique du modèle de fréquence",
                              "15 ans de tarification automobile, GLM de Poisson et boosting, normes ASOP", "jugement"),
        "features": Agent("feature_engineer", "data scientist senior", "construire la préparation des variables",
                          "expert scikit-learn et pandas", "code"),
        "tuning": Agent("meta_tuner", "ingénieur machine learning senior",
                        "choisir le modèle et ses hyperparamètres", "connaît forces et faiblesses des modèles", "code"),
        "training": Agent("entraineur", "ingénieur machine learning senior", "entraîner et livrer le modèle",
                          "soigneux sur la sérialisation et les contrats d'interface", "code"),
        "evaluation": Agent("evaluateur", "ingénieur machine learning senior", "évaluer le modèle sur le test",
                            "précis sur les définitions de métriques", "code"),
        "documentation": Agent("redacteur", "rédacteur technique actuariel",
                               "documenter le modèle pour qu'un autre actuaire puisse en juger",
                               "connaît les exigences de documentation des ASOP 56, 23 et 41", "leger"),
        "juge": Agent("juge", "responsable de la modélisation (manager)",
                      "examiner le travail de l'équipe et décider s'il est acceptable",
                      "actuaire certifié, exigeant sur les fuites, l'exposition et la calibration", "jugement"),
        "orchestrateur": Agent("orchestrateur", "orchestrateur de l'équipe",
                               "débloquer une tâche en échec", "chef de projet technique", "jugement"),
    }


SPEC_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["objectif", "traitements_donnees", "variables", "variables_exclues", "traitement_exposition",
                 "modeles_candidats", "critere_selection", "validation", "risques_a_surveiller"],
    "properties": {
        "objectif": {"type": "string"},
        "traitements_donnees": {"type": "array", "items": {"type": "string"}},
        "variables": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                      "required": ["nom", "transformation", "justification"],
                      "properties": {"nom": {"type": "string"}, "transformation": {"type": "string"},
                                     "justification": {"type": "string"}}}},
        "variables_exclues": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                              "required": ["nom", "raison"],
                              "properties": {"nom": {"type": "string"}, "raison": {"type": "string"}}}},
        "traitement_exposition": {"type": "string"},
        "modeles_candidats": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                              "required": ["nom", "librairie", "grille"],
                              "properties": {"nom": {"type": "string"}, "librairie": {"type": "string"},
                                             "grille": {"type": "string"}}}},
        "critere_selection": {"type": "string"},
        "validation": {"type": "string"},
        "risques_a_surveiller": {"type": "array", "items": {"type": "string"}},
    },
}

REWORKABLE = ["features", "tuning", "training", "evaluation", "documentation"]
JUDGE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["verdict", "points", "synthese"],
    "properties": {
        "verdict": {"type": "string", "enum": ["approuve", "a_reprendre"]},
        "points": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                   "required": ["gravite", "tache", "constat", "consigne"],
                   "properties": {"gravite": {"type": "string", "enum": ["bloquant", "important", "mineur"]},
                                  "tache": {"type": "string", "enum": REWORKABLE},
                                  "constat": {"type": "string"}, "consigne": {"type": "string"}}}},
        "synthese": {"type": "string"},
    },
}
ESC_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["decision", "consigne"],
    "properties": {"decision": {"type": "string", "enum": ["reessayer", "abandonner"]},
                   "consigne": {"type": "string"}},
}

ORDER = ["extraction", "eda", "features", "tuning", "training", "evaluation", "documentation"]
GATES = {"extraction": gates.gate_extraction, "features": gates.gate_features, "tuning": gates.gate_tuning,
         "training": gates.gate_training, "evaluation": gates.gate_evaluation}
CONTEXT = {"extraction": [], "eda": ["extraction"], "features": ["eda", "methodologie"],
           "tuning": ["methodologie", "features"], "training": ["methodologie", "tuning"],
           "evaluation": ["training"],
           "documentation": ["extraction", "eda", "methodologie", "features", "tuning", "training", "evaluation"]}


class ModelingCrew:
    def __init__(self, llm: LLM, workdir: Path, variant: str = "autonome"):
        assert variant in ("autonome", "naif")
        self.variant = variant
        self.autonomous = variant == "autonome"
        self.wd = workdir
        if workdir.exists():
            shutil.rmtree(workdir)
        (workdir / "data").mkdir(parents=True)
        (workdir / "artifacts").mkdir()
        self.runner = Runner(llm, workdir, ENV_NOTE, use_gates=self.autonomous)
        self.agents = make_agents()
        self.log = {"variant": variant, "tasks": {}, "escalations": [], "judge": [], "rework": []}

    # ------------------------------------------------------------------ tâches
    def _task(self, tid: str) -> Task:
        a = self.agents[tid]
        if tid == "eda":
            train = pd.read_parquet(self.wd / "data" / "train.parquet")
            desc = TASKS_TEXT["eda"][0] + "\n\n### Sortie de l'outil EDA\n" + eda_tool(train)
            return Task("eda", a, desc + "\nSois synthétique : 500 mots au maximum.", TASKS_TEXT["eda"][1],
                        kind="text", context=CONTEXT["eda"], max_tokens=6000)
        if tid == "documentation":
            desc = ("Rédiger la documentation complète du modèle en markdown, en français, en suivant la grille "
                    "ASOP fournie (objet, données et revue des données, hypothèses, méthode, validation, limites "
                    "et faiblesses, dépendances envers des outils ou modèles tiers dont l'IA). N'invente aucun "
                    "chiffre : utilise uniquement ceux des rapports.")
            return Task("documentation", a, desc, "Le document markdown seul.", kind="text",
                        context=CONTEXT["documentation"], cached_context=ASOP_GRID, max_tokens=8000)
        desc, exp = TASKS_TEXT[tid]
        return Task(tid, a, desc, exp, kind="code", context=CONTEXT[tid], gate=GATES.get(tid),
                    timeout_s=1500 if tid in ("tuning", "training") else 900)

    def _methodology(self):
        a = self.agents["methodologue"]
        desc = ("À partir du rapport d'analyse exploratoire, fixer la spécification du modèle de fréquence "
                "que l'équipe doit implémenter : traitements de données, variables et transformations, variables "
                "à exclure, traitement de l'exposition, 2 ou 3 modèles candidats avec grilles réduites, critère "
                "de sélection, schéma de validation, risques à surveiller. Respecte la grille ASOP fournie. "
                "Sois concis : une ou deux phrases par champ, pas de répétition.")
        t = Task("methodologie", a, desc, "La spécification au format JSON demandé.", kind="json",
                 context=["eda"], json_schema=SPEC_SCHEMA, cached_context=ASOP_GRID, max_tokens=16000)
        r = self.runner.run(t)
        self.log["tasks"]["methodologie"] = {k: r[k] for k in ("ok", "attempts")}
        self.log["spec"] = r.get("data")

    def _run_task(self, tid: str, instruction: str = "") -> bool:
        task = self._task(tid)
        r = self.runner.run(task, extra_instruction=instruction)
        if tid == "documentation" and r["ok"]:
            doc = r["report"].strip()
            if doc.startswith("```"):
                doc = doc.strip("`").removeprefix("markdown").strip()
            (self.wd / "artifacts" / "model_doc.md").write_text(doc)
            if self.autonomous:
                ok, msg = gates.gate_documentation(self.wd)
                if not ok:
                    # Limite connue (relevée en relecture, laissée telle quelle pour rester fidèle aux exécutions publiées) :
                    # ce refus n'est pas inscrit dans l'historique de la tâche et la version réécrite n'est pas recontrôlée.
                    # analyse.py reconstitue ces refus depuis results/llm_calls.jsonl et recontrôle la documentation finale.
                    r = self.runner.run(task, extra_instruction=f"La porte documentaire a refusé le document : {msg}")
                    (self.wd / "artifacts" / "model_doc.md").write_text(r["report"])
        self.log["tasks"].setdefault(tid, []).append(
            {"ok": r["ok"], "attempts": r["attempts"], "history": r.get("history", []), "instruction": instruction[:500]})
        if not r["ok"] and self.autonomous:
            # Escalade : l'orchestrateur lit l'échec et décide (remplace l'humain qui débloquait l'agent)
            esc = Task("escalade_" + tid, self.agents["orchestrateur"],
                       f"La tâche « {tid} » a échoué après {task.max_attempts} tentatives. Dernier retour :\n"
                       f"{r['report'][:3000]}\nDécide : réessayer avec une consigne précise, ou abandonner.",
                       "Décision JSON.", kind="json", context=CONTEXT.get(tid, []), json_schema=ESC_SCHEMA,
                       max_tokens=8000)
            e = self.runner.run(esc)
            d = e.get("data") or {"decision": "abandonner", "consigne": ""}
            self.log["escalations"].append({"task": tid, **d})
            if d["decision"] == "reessayer":
                r = self.runner.run(task, extra_instruction=d["consigne"])
                self.log["tasks"][tid].append({"ok": r["ok"], "attempts": r["attempts"],
                                               "history": r.get("history", []), "instruction": d["consigne"][:500]})
        return r["ok"]

    def _judge(self) -> dict:
        audit = gates.audit_package(self.wd / "artifacts", self.wd / "data" / "train.parquet",
                                    self.wd / "data" / "test.parquet")
        codes = []
        for f in ("features.py", "predict.py"):
            p = self.wd / "artifacts" / f
            if p.exists():
                codes.append(f"### artifacts/{f}\n```python\n{p.read_text()[:4000]}\n```")
        for tid in ("tuning", "training", "evaluation"):
            e = self.runner.memory.last(tid)
            if e and e.get("code"):
                codes.append(f"### Script de la tâche {tid}\n```python\n{e['code'][:5000]}\n```")
        doc = (self.wd / "artifacts" / "model_doc.md")
        evidence = (f"## Spécification\n{json.dumps(self.log.get('spec'), ensure_ascii=False)[:5000]}\n\n"
                    f"## Audit déterministe indépendant (portes)\n{json.dumps(audit, ensure_ascii=False)[:3000]}\n\n"
                    + "\n\n".join(codes)
                    + f"\n\n## Documentation\n{doc.read_text()[:6000] if doc.exists() else 'absente'}")
        t = Task("jugement", self.agents["juge"],
                 "Examiner le travail de l'équipe (spécification, code, audit, documentation) et rendre un verdict. "
                 "Ne renvoie une tâche que pour un défaut réel et important ; donne une consigne précise et "
                 "actionnable. Les défauts de fond priment : fuite, exposition, calibration, contamination du "
                 "test, métriques fausses, documentation trompeuse.\n\n" + evidence,
                 "Verdict JSON concis (une ou deux phrases par champ).", kind="json", context=["eda"], json_schema=JUDGE_SCHEMA,
                 cached_context=ASOP_GRID, max_tokens=16000)
        r = self.runner.run(t)
        v = r.get("data") or {"verdict": "approuve", "points": [], "synthese": "verdict illisible"}
        self.log["judge"].append(v)
        return v

    # ------------------------------------------------------------------ pipeline
    def run(self) -> dict:
        for tid in ORDER:
            if tid == "features" and self.autonomous:
                self._methodology()
            if not self._run_task(tid):
                self.log["status"] = f"echec:{tid}"
                break
        else:
            self.log["status"] = "termine"
            if self.autonomous:
                v = self._judge()
                if v["verdict"] == "a_reprendre":
                    pts = [p for p in v["points"] if p["gravite"] in ("bloquant", "important")]
                    if pts:
                        first = min(REWORKABLE.index(p["tache"]) for p in pts)
                        for tid in REWORKABLE[first:]:
                            instr = "\n".join(f"- {p['constat']} => {p['consigne']}" for p in pts if p["tache"] == tid)
                            self.log["rework"].append(tid)
                            if not self._run_task(tid, instruction=instr):
                                self.log["status"] = f"echec_reprise:{tid}"
                                break
                        else:
                            self.log["final_verdict"] = self._judge()["verdict"]
                else:
                    self.log["final_verdict"] = "approuve"
        (self.wd / "crew_log.json").write_text(json.dumps(self.log, ensure_ascii=False, indent=1))
        return self.log
