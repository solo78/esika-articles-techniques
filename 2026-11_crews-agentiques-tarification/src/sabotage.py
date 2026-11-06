"""Fabrication de dossiers de modèle avec défauts injectés, pour tester la crew MRM.

Chaque dossier est construit sans LLM, à partir du GLM témoin, avec un seul défaut connu :
  P0_sain           aucun défaut (mesure des fausses alertes)
  P1_fuite          variable « IndicDossierActif » construite à partir de la cible
  P2_exposition     modèle ajusté sur le nombre de sinistres sans tenir compte de l'exposition
  P3_contamination  LightGBM entraîné sur train + test, métriques de test optimistes
  P4_calibration    prédictions multipliées par 0,85 (« ajustement de tendance » non justifié)
  P5_documentation  modèle correct, documentation lacunaire au regard des ASOP
  P6_metriques      modèle correct, métriques annoncées embellies
Le nom du dossier est masqué pour la crew (dossier « package »).
"""
from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import GLM_FEATURES, evaluate_package, fit_glm, fit_lgbm
from .data import reference_split

DEFECTS = {
    "P0_sain": None, "P1_fuite": "fuite_cible", "P2_exposition": "exposition",
    "P3_contamination": "contamination_test", "P4_calibration": "calibration",
    "P5_documentation": "documentation", "P6_metriques": "metriques_inexactes",
}

LEAK_FEATURES = GLM_FEATURES.replace(
    "    return out\n",
    "    out[\"IndicDossierActif\"] = df[\"IndicDossierActif\"].astype(str)\n    return out\n")
assert "IndicDossierActif" in LEAK_FEATURES

TRAIN_SCRIPT = textwrap.dedent('''
    # Script d'entraînement du modèle (extrait du dépôt de l'équipe tarification)
    import pandas as pd, numpy as np, joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import PoissonRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from features import prepare

    base = pd.read_parquet("data/{base}")
    X = prepare(base)
    cats = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    nums = [c for c in X.columns if c not in cats]
    model = Pipeline([("pre", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), cats),
                                                 ("num", StandardScaler(), nums)])),
                      ("glm", PoissonRegressor(alpha=1e-6, max_iter=1000))])
    expo = np.minimum(base["Exposure"], 1.0)
    y = np.minimum(base["ClaimNb"], 4)
    {fit_line}
    joblib.dump(model, "model.joblib")
''')

LGB_TRAIN_SCRIPT = textwrap.dedent('''
    # Script d'entraînement du modèle (extrait du dépôt de l'équipe tarification)
    import pandas as pd, numpy as np, joblib, lightgbm as lgb
    from features import prepare
    base = pd.read_parquet("data/base_modelisation.parquet")
    X = prepare(base); expo = np.minimum(base["Exposure"], 1.0); y = np.minimum(base["ClaimNb"], 4)
    rng = np.random.default_rng(0); val = rng.random(len(base)) < 0.1
    model = lgb.LGBMRegressor(objective="poisson", learning_rate=0.05, n_estimators=2000, num_leaves=31,
                              min_child_samples=500, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              reg_lambda=1.0, random_state=0, n_jobs=2, verbose=-1)
    model.fit(X[~val], y[~val] / expo[~val], sample_weight=expo[~val],
              eval_X=(X[val],), eval_y=(y[val] / expo[val],), eval_sample_weight=[expo[val]],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    joblib.dump(model, "model.joblib")
''')


def doc_template(model_name: str, variables: str, metrics: dict, base: str, extra_limits: str = "") -> str:
    return textwrap.dedent(f"""
    # Documentation du modèle de fréquence RC automobile

    ## 1. Objet et usage prévu
    Modèle de fréquence des sinistres responsabilité civile automobile, destiné à la tarification
    technique des contrats de particuliers (estimation du nombre attendu de sinistres par police et par
    an). Il ne couvre pas le coût moyen des sinistres. Public : équipe tarification et validation.

    ## 2. Données et revue des données
    Source : portefeuille RC auto français (jeu public freMTPL2freq), 677 991 polices.
    Base de modélisation : {base}. Jeu de test : 20 % des polices, tiré aléatoirement (graine 2026),
    jamais utilisé pour l'apprentissage ni le réglage.
    Revue réalisée : absence de valeurs manquantes ; 1 224 polices ont une exposition supérieure à un an
    (plafonnée à 1 an) ; quelques polices déclarent plus de 4 sinistres (plafonnées à 4 pour
    l'apprentissage) ; l'identifiant IDpol présente une corrélation résiduelle avec la fréquence
    (artefact de constitution du fichier) et n'est pas utilisé. Données fournies par un tiers (CASdatasets) :
    confiance accordée sans audit.

    ## 3. Hypothèses
    Nombre de sinistres de Poisson conditionnellement aux variables, proportionnel à l'exposition
    (offset log(exposition) via poids d'exposition). Indépendance entre polices.

    ## 4. Méthode
    {model_name}. Variables : {variables}.
    Classes de puissance plafonnées à 9, âge du véhicule en 3 classes, âge du conducteur en 7 classes,
    bonus-malus plafonné à 150, densité en logarithme.

    ## 5. Validation
    Métriques sur le jeu de test : déviance de Poisson x100 = {metrics['deviance_x100']:.3f},
    Gini = {metrics['gini']:.3f}, sinistres observés / attendus = {metrics['ae_global']:.3f}.
    Comparaison au modèle nul et contrôle de la calibration par décile de prédiction.

    ## 6. Limites et faiblesses connues
    Pas de variable de sinistralité antérieure hors bonus-malus ; pas d'interaction explicite ;
    données d'une seule période, sans tendance. {extra_limits}

    ## 7. Dépendances
    scikit-learn (PoissonRegressor), LightGBM selon le cas ; aucun modèle externe ; pas d'expert tiers.
    """).strip()


def build_packages(outdir: Path) -> dict:
    """Construit les 7 dossiers et renvoie les métriques réelles (recalculées) de chacun."""
    train, test = reference_split()
    rng = np.random.default_rng(7)
    out = {}
    glm_vars = "puissance, âge véhicule, âge conducteur, bonus-malus, densité, zone, marque, carburant, région"

    def finish(pkg, base, metrics_reported, model_name, variables, doc=None, train_script=None):
        (pkg / "train_script.py").write_text(train_script or TRAIN_SCRIPT.format(
            base=base, fit_line='model.fit(X, y / expo, glm__sample_weight=expo)'))
        (pkg / "eval.json").write_text(json.dumps(metrics_reported, indent=2))
        (pkg / "tuning.json").write_text(json.dumps({"best_model": model_name, "alpha": 1e-6}))
        (pkg / "model_doc.md").write_text(doc or doc_template(model_name, variables, metrics_reported, base))
        for f in pkg.glob("_mu_test.npy"):
            f.unlink()

    def data_dir(pkg, tr, te, extra=None):
        d = pkg.parent / (pkg.name + "_data")
        d.mkdir(parents=True, exist_ok=True)
        tr.to_parquet(d / "train.parquet", index=False)
        te.to_parquet(d / "test.parquet", index=False)
        if extra is not None:
            extra.to_parquet(d / "base_modelisation.parquet", index=False)

    for name in DEFECTS:
        pkg = outdir / name
        if pkg.exists():
            shutil.rmtree(pkg)
        tr, te = train.copy(), test.copy()
        if name == "P1_fuite":
            for d in (tr, te):
                has = d["ClaimNb"].to_numpy() > 0
                u = rng.random(len(d))
                d["IndicDossierActif"] = np.where(has, u < 0.6, u < 0.03).astype(int)
            fit_glm(tr, pkg, features_src=LEAK_FEATURES)
        elif name == "P2_exposition":
            fit_glm(tr, pkg, offset=False)
            (pkg / "predict.py").write_text((pkg / "predict.py").read_text().replace(
                "return freq * np.minimum(df[\"Exposure\"].to_numpy(float), 1.0)", "return freq"))
        elif name == "P3_contamination":
            fit_lgbm(pd.concat([tr, te], ignore_index=True), pkg)
        else:
            fit_glm(tr, pkg)
        if name == "P4_calibration":
            src = (pkg / "predict.py").read_text()
            src = src.replace("freq = _MODEL.predict(X)",
                              "freq = _MODEL.predict(X)\n    freq = freq * 0.85  # ajustement de tendance")
            (pkg / "predict.py").write_text(src)
        true_m, _ = evaluate_package(pkg, te)
        reported = dict(true_m)
        model_name = "GLM de Poisson (scikit-learn PoissonRegressor)"
        variables = glm_vars
        base, doc, script = "data/train.parquet", None, None
        if name == "P1_fuite":
            variables += ", indicateur de dossier client actif (IndicDossierActif, issu du système de gestion)"
        if name == "P2_exposition":
            script = TRAIN_SCRIPT.format(base="train.parquet", fit_line="model.fit(X, y)")
        if name == "P3_contamination":
            model_name = "LightGBM, objectif Poisson, arrêt anticipé"
            variables = "puissance, âge véhicule, âge conducteur, bonus-malus, log densité, zone, marque, carburant, région"
            base = "data/base_modelisation.parquet"
            script = LGB_TRAIN_SCRIPT
        if name == "P5_documentation":
            doc = textwrap.dedent(f"""
            # Modèle fréquence RC auto
            GLM Poisson sur les variables du portefeuille. Déviance test x100 = {true_m['deviance_x100']:.3f},
            Gini = {true_m['gini']:.3f}. Le modèle est performant et prêt pour la production.
            """).strip()
        if name == "P6_metriques":
            reported = {"deviance_x100": true_m["deviance_x100"] - 1.2, "gini": true_m["gini"] + 0.06,
                        "ae_global": 1.0}
        data_dir(pkg, tr, te, extra=pd.concat([tr, te], ignore_index=True) if name == "P3_contamination" else None)
        finish(pkg, base.replace("data/", ""), reported, model_name, variables, doc=doc, train_script=script)
        if name not in ("P5_documentation",) and doc is None:
            (pkg / "model_doc.md").write_text(doc_template(model_name, variables, reported, base))
        out[name] = {"defect": DEFECTS[name], "true": true_m, "reported": reported}
    (outdir / "packages_truth.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out
