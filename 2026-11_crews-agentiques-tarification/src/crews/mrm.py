"""Crew de revue indépendante des modèles (MRM), autonome, guidée par la grille ASOP.

Elle reçoit un dossier de modèle (package/) et les données (data/), sans savoir s'il contient
un défaut. Quatre agents spécialisés puis un juge MRM, comme dans le papier, mais :
  - la grille ASOP (paraphrasée) est chargée en contexte mis en cache (CAG) ;
  - les agents de code ont une boucle d'auto-correction et une porte « contrat » (fichier JSON
    attendu présent et lisible) ;
  - le juge MRM tranche seul (plus d'humain) et classe chaque constat dans une taxonomie fermée.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..framework import Agent, Runner, Task
from ..llm import LLM
from ..tools import ASOP_GRID

CATEGORIES = ["fuite_cible", "exposition", "contamination_test", "calibration", "documentation",
              "metriques_inexactes", "identifiant", "autre"]

ENV_NOTE = """
## Environnement d'exécution
- Python 3.13, pandas 3 (colonnes texte de type « str »), numpy 2, scikit-learn 1.8, lightgbm 4, statsmodels,
  scipy, joblib. Pas de réseau. 2 cœurs. Chaque script doit tenir en moins de 15 minutes.
- Lancé depuis le dossier de revue. Contenu :
  * package/ : le dossier de modèle à revoir (features.py avec prepare(df), predict.py avec predict(df) qui
    renvoie le nombre attendu de sinistres par police, model.joblib, model_doc.md, eval.json, tuning.json,
    train_script.py écrit par l'équipe de modélisation).
  * data/ : train.parquet, test.parquet et éventuellement d'autres fichiers cités par l'équipe.
- Pour charger le modèle : importlib.util.spec_from_file_location("predict", "package/predict.py").
  predict(df) doit recevoir un df sans la colonne ClaimNb.
- Écris tes résultats dans mrm/<nom>.json (le dossier mrm/ existe).
"""


def make_agents():
    return {
        "conformite": Agent("controle_conformite", "data scientist chargé de la conformité documentaire",
                            "vérifier la documentation du modèle point par point contre la grille ASOP",
                            "lit les documentations avec un œil d'auditeur", "code"),
        "replication": Agent("replicateur", "ingénieur machine learning de la validation",
                             "répliquer indépendamment l'entraînement et vérifier les chiffres annoncés",
                             "ne fait confiance à aucun chiffre qu'il n'a pas recalculé", "code"),
        "conceptuel": Agent("analyste_conceptuel", "analyste validation de modèles (solidité conceptuelle)",
                            "vérifier hypothèses, variables, relativités et traitement de l'exposition",
                            "actuaire tarification, connaît les pièges classiques des modèles de fréquence", "code"),
        "resultats": Agent("analyste_resultats", "analyste validation de modèles (analyse des résultats)",
                           "tester calibration, stabilité et comportement sous stress",
                           "spécialiste du backtesting et des tests de résistance", "code"),
        "juge": Agent("juge_mrm", "responsable du risque de modèle",
                      "rendre une décision motivée sur l'utilisation du modèle",
                      "actuaire certifié, indépendant de l'équipe de modélisation", "jugement"),
    }


def _gate_json(name):
    def g(wd: Path):
        f = wd / "mrm" / f"{name}.json"
        if not f.exists():
            return False, f"mrm/{name}.json absent."
        try:
            json.loads(f.read_text())
        except json.JSONDecodeError as e:
            return False, f"mrm/{name}.json illisible : {e}"
        return True, "OK"
    return g


CONF_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["points", "synthese"],
    "properties": {
        "points": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                   "required": ["id", "statut", "constat"],
                   "properties": {"id": {"type": "string"},
                                  "statut": {"type": "string", "enum": ["conforme", "partiel", "non_conforme", "non_applicable"]},
                                  "constat": {"type": "string"}}}},
        "synthese": {"type": "string"},
    },
}
VERDICT_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["decision", "constats", "synthese"],
    "properties": {
        "decision": {"type": "string", "enum": ["approuve", "approuve_sous_reserve", "refuse"]},
        "constats": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                     "required": ["categorie", "gravite", "description", "preuve"],
                     "properties": {"categorie": {"type": "string", "enum": CATEGORIES},
                                    "gravite": {"type": "string", "enum": ["bloquant", "important", "mineur"]},
                                    "description": {"type": "string"}, "preuve": {"type": "string"}}}},
        "synthese": {"type": "string"},
    },
}

TASKS = {
    "replication": (
        "Répliquer le modèle de façon indépendante : (1) recalculer toi-même sur data/test.parquet les métriques "
        "du modèle livré (package/predict.py) : deviance_x100 = 200 x moyenne de [y log(y/mu) - (y - mu)], gini "
        "ordonné sur la fréquence prédite, ae_global = somme(y)/somme(mu) ; (2) les comparer à package/eval.json et "
        "à la documentation ; (3) réentraîner un modèle de même spécification (mêmes variables via "
        "package/features.py, même algorithme et hyperparamètres) uniquement sur data/train.parquet, avec ton "
        "propre code, et comparer ses métriques de test à celles du modèle livré ; (4) examiner "
        "package/train_script.py et les données qu'il utilise. Écrire mrm/replication.json.",
        "mrm/replication.json (métriques annoncées, recalculées, répliquées, écarts) et un rapport."),
    "conceptuel": (
        "Évaluer la solidité conceptuelle : liste des variables utilisées et lien plausible avec la sinistralité ; "
        "recherche de variables issues de la cible ou de la gestion des sinistres (pouvoir prédictif anormal, "
        "fréquence observée par modalité) ; sens et amplitude des effets (relativités ou importance) ; "
        "traitement de l'exposition (le nombre attendu doit être proportionnel à l'exposition : tester en "
        "doublant Exposure pour des polices de moins de 6 mois) ; usage éventuel de l'identifiant IDpol "
        "(tester en le permutant). Écrire mrm/conceptuel.json.",
        "mrm/conceptuel.json et un rapport."),
    "resultats": (
        "Analyser les résultats hors échantillon sur data/test.parquet : A/E global et par décile de fréquence "
        "prédite, A/E par segment (Area, classes d'âge conducteur, VehGas), comparaison au modèle nul, puis "
        "tests de résistance comme dans une revue MRM (variables numériques décalées, valeurs extrêmes) en "
        "mesurant la dégradation. Écrire mrm/resultats.json.",
        "mrm/resultats.json et un rapport."),
}


class MRMCrew:
    def __init__(self, llm: LLM, workdir: Path, package: Path, data: Path):
        self.wd = workdir
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True)
        shutil.copytree(package, workdir / "package")
        shutil.copytree(data, workdir / "data")
        (workdir / "mrm").mkdir()
        self.runner = Runner(llm, workdir, ENV_NOTE, use_gates=True)
        self.agents = make_agents()
        self.log = {"tasks": {}}

    def run(self) -> dict:
        pkg = self.wd / "package"
        listing = "\n".join(sorted(p.name for p in (self.wd / "data").iterdir()))
        doc_pack = (f"## model_doc.md\n{(pkg / 'model_doc.md').read_text()}\n\n## eval.json\n"
                    f"{(pkg / 'eval.json').read_text()}\n\n## tuning.json\n{(pkg / 'tuning.json').read_text()}\n\n"
                    f"## train_script.py\n```python\n{(pkg / 'train_script.py').read_text()}\n```\n\n"
                    f"## features.py\n```python\n{(pkg / 'features.py').read_text()}\n```\n\n"
                    f"## predict.py\n```python\n{(pkg / 'predict.py').read_text()}\n```\n\n"
                    f"## Fichiers du dossier data/\n{listing}")
        conf = Task("conformite", self.agents["conformite"],
                    "Vérifier la documentation et les pièces du dossier de modèle contre chaque point de la grille "
                    "ASOP (A1 à E5). Pour chaque point : statut et constat factuel. Pièces :\n\n" + doc_pack,
                    "JSON concis (une phrase par constat).", kind="json", json_schema=CONF_SCHEMA,
                    cached_context=ASOP_GRID, max_tokens=12000)
        r = self.runner.run(conf)
        self.log["tasks"]["conformite"] = {"ok": r["ok"], "attempts": r["attempts"]}
        (self.wd / "mrm" / "conformite.json").write_text(json.dumps(r.get("data"), ensure_ascii=False, indent=1))
        for tid, (desc, exp) in TASKS.items():
            t = Task(tid, self.agents[tid], desc + "\n\nPièces du dossier :\n" + doc_pack, exp, kind="code",
                     gate=_gate_json(tid), timeout_s=900, max_attempts=3)
            r = self.runner.run(t)
            self.log["tasks"][tid] = {"ok": r["ok"], "attempts": r["attempts"], "history": r.get("history", [])}
        evidence = "\n\n".join(
            f"## {n}\n{(self.wd / 'mrm' / f'{n}.json').read_text()[:6000] if (self.wd / 'mrm' / f'{n}.json').exists() else 'absent'}"
            + (f"\n### Rapport de l'agent\n{self.runner.memory.last(n)['report'][:2500]}" if self.runner.memory.last(n) else "")
            for n in ("conformite", "replication", "conceptuel", "resultats"))
        jt = Task("verdict", self.agents["juge"],
                  "Consolider les travaux des quatre agents de revue et rendre la décision MRM. Chaque constat doit "
                  "être étayé par une preuve chiffrée ou une citation des pièces ; classe-le dans la catégorie la "
                  "plus précise. Ne retiens pas un constat sans preuve.\n\n" + evidence
                  + "\n\n## Documentation revue\n" + (pkg / "model_doc.md").read_text()[:5000],
                  "Décision JSON concise.", kind="json", json_schema=VERDICT_SCHEMA, cached_context=ASOP_GRID,
                  max_tokens=16000)
        r = self.runner.run(jt)
        self.log["verdict"] = r.get("data")
        (self.wd / "mrm_log.json").write_text(json.dumps(self.log, ensure_ascii=False, indent=1))
        return self.log
