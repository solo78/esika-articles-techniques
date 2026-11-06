"""Portes de qualité déterministes : elles remplacent les vérifications que l'humain faisait
dans le papier (« tu as oublié d'appeler predict », « charge le modèle avec joblib »...).

Chaque porte renvoie (ok, message). Le message est renvoyé tel quel à l'agent fautif.
Les portes ne révèlent jamais la bonne réponse : elles disent quelle propriété est violée.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .data import ids_hash, reference_test_hash

CHECKER = Path(__file__).resolve().parent / "package_check.py"
REF_HASH = None


def audit_package(pkg: Path, train: Path | str, test: Path, timeout=900) -> dict:
    p = subprocess.run([sys.executable, "-I", str(CHECKER), str(pkg), str(train), str(test)],
                       capture_output=True, text=True, timeout=timeout)
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {"errors": [f"audit impossible : {p.stderr[-800:]}"]}


def gate_extraction(wd: Path):
    global REF_HASH
    REF_HASH = REF_HASH or reference_test_hash()
    d = wd / "data"
    need = ["train.parquet", "test.parquet", "train_sample.parquet"]
    miss = [f for f in need if not (d / f).exists()]
    if miss:
        return False, f"Fichiers manquants dans data/ : {miss}"
    tr, te, sa = (pd.read_parquet(d / f, columns=["IDpol"]) for f in need)
    msgs = []
    if ids_hash(te["IDpol"]) != REF_HASH:
        msgs.append("Le jeu de test ne correspond pas au découpage imposé "
                    "(train_test_split, test_size=0.2, random_state=2026, sur le fichier brut complet).")
    if set(tr.IDpol) & set(te.IDpol):
        msgs.append("Des polices sont à la fois dans train et test.")
    if not set(sa.IDpol) <= set(tr.IDpol):
        msgs.append("L'échantillon de réglage contient des polices hors du train.")
    if not 0.38 <= len(sa) / len(tr) <= 0.42:
        msgs.append(f"L'échantillon de réglage fait {len(sa) / len(tr):.1%} du train au lieu de 40 %.")
    return (not msgs), ("OK" if not msgs else " ".join(msgs))


def gate_features(wd: Path):
    f = wd / "artifacts" / "features.py"
    if not f.exists():
        return False, "artifacts/features.py absent."
    code = f"""
import importlib.util, pandas as pd, numpy as np, json
spec = importlib.util.spec_from_file_location('f', 'artifacts/features.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
te = pd.read_parquet('data/test.parquet').head(20000).drop(columns=['ClaimNb'])
a = m.prepare(te.copy()); b = m.prepare(te.copy())
num = a.select_dtypes('number')
print(json.dumps({{'cols': list(map(str, a.columns)), 'n': len(a), 'nan': int(a.isna().sum().sum()),
  'inf': int(np.isinf(num.to_numpy(dtype=float)).sum()) if num.shape[1] else 0,
  'det': bool(a.equals(b))}}))
"""
    p = subprocess.run([sys.executable, "-I", "-c", code], cwd=wd, capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        return False, ("prepare(df) échoue sur le jeu de test privé de la colonne ClaimNb (la cible n'est "
                       f"pas disponible au moment de tarifer) :\n{p.stderr[-1500:]}")
    r = json.loads(p.stdout.strip().splitlines()[-1])
    msgs = []
    if r["n"] != 20000:
        msgs.append("prepare(df) ne conserve pas une ligne par police.")
    if r["nan"] or r["inf"]:
        msgs.append(f"prepare(df) produit {r['nan']} valeurs manquantes et {r['inf']} infinies.")
    if any(c.lower() in ("claimnb", "idpol") for c in r["cols"]):
        msgs.append("Les variables produites contiennent la cible ou l'identifiant de police.")
    if not r["det"]:
        msgs.append("prepare(df) n'est pas déterministe.")
    return (not msgs), ("OK : " + ", ".join(r["cols"]) if not msgs else " ".join(msgs))


def gate_tuning(wd: Path):
    f = wd / "artifacts" / "tuning.json"
    if not f.exists():
        return False, "artifacts/tuning.json absent."
    try:
        t = json.loads(f.read_text())
    except json.JSONDecodeError as ex:
        return False, f"tuning.json illisible : {ex}"
    miss = [k for k in ("best_model", "best_params", "cv_results") if k not in t]
    return (not miss), ("OK" if not miss else f"Clés manquantes dans tuning.json : {miss}")


def gate_training(wd: Path):
    for f in ("predict.py",):
        if not (wd / "artifacts" / f).exists():
            return False, f"artifacts/{f} absent."
    r = audit_package(wd / "artifacts", wd / "data" / "train.parquet", wd / "data" / "test.parquet")
    msgs = list(r.get("errors", []))
    if not msgs:
        if not (r.get("finite") and r.get("positive")):
            msgs.append("predict(df) renvoie des valeurs non finies ou non strictement positives.")
        if r.get("id_sensitivity", 0) > 1e-6:
            msgs.append("Les prédictions changent quand on permute l'identifiant IDpol : le modèle "
                        "utilise un identifiant administratif.")
        dev = abs(r.get("exposure_ratio_median", 2) - 2)
        if dev > 1e-3 or r.get("exposure_ratio_maxdev", 0) > 1e-3:
            msgs.append("Doubler l'exposition d'une police (<= 0,5 an) ne double pas son nombre attendu "
                        "de sinistres : predict(df) doit renvoyer fréquence x exposition.")
        ae = r.get("train_ae", 1)
        if not 0.97 <= ae <= 1.03:
            msgs.append(f"Sur le train, sinistres observés / attendus = {ae:.3f} : le modèle n'est pas "
                        "équilibré (propriété d'équilibre d'un GLM de Poisson).")
    return (not msgs), ("OK" if not msgs else " ".join(msgs))


def gate_evaluation(wd: Path):
    f = wd / "artifacts" / "eval.json"
    if not f.exists():
        return False, "artifacts/eval.json absent."
    rep = json.loads(f.read_text())
    r = audit_package(wd / "artifacts", "-", wd / "data" / "test.parquet")
    ind = r.get("test")
    if ind is None:
        return False, "Recalcul indépendant impossible : " + "; ".join(r.get("errors", []))
    msgs = []
    for k in ("deviance_x100", "gini", "ae_global"):
        if k not in rep:
            msgs.append(f"Clé {k} absente de eval.json.")
        elif abs(float(rep[k]) - ind[k]) > max(2e-3, 2e-3 * abs(ind[k])):
            msgs.append(f"{k} annoncé = {float(rep[k]):.4f}, ne correspond pas au recalcul indépendant "
                        "(vérifie la définition exacte demandée).")
    return (not msgs), ("OK" if not msgs else " ".join(msgs))


REQUIRED_DOC = ["objet", "donn", "hypoth", "méthod", "validation", "limite", "dépendance"]


def gate_documentation(wd: Path):
    f = wd / "artifacts" / "model_doc.md"
    if not f.exists():
        return False, "artifacts/model_doc.md absent."
    t = f.read_text().lower()
    miss = [k for k in REQUIRED_DOC if k not in t]
    if len(t) < 2500:
        miss.append("longueur insuffisante")
    return (not miss), ("OK" if not miss else f"Sections manquantes ou trop brèves : {miss}")
