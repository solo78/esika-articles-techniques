"""Témoins forts faits main par un actuaire (pas d'IA) au format « dossier de modèle ».

Un dossier de modèle contient :
  features.py   prepare(df) -> DataFrame des variables explicatives (sans la cible)
  predict.py    predict(df) -> nombre attendu de sinistres (fréquence x exposition)
  model.joblib  objet entraîné
  model_doc.md  documentation
  eval.json     métriques annoncées sur le jeu de test
C'est le même contrat que celui imposé aux agents, et c'est ce que la crew MRM revoit.

Témoins :
  - GLM de Poisson « à la Noll, Salzmann et Wüthrich (2018) » : classes d'âge,
    plafonds, log densité, offset log(exposition) via poids = exposition.
  - LightGBM Poisson avec arrêt anticipé sur 10 % du train.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import reference_split
from .metrics import all_metrics

# ---------------------------------------------------------------- GLM témoin
GLM_FEATURES = textwrap.dedent('''
    import numpy as np
    import pandas as pd

    def prepare(df: pd.DataFrame) -> pd.DataFrame:
        """Préparation sans état (pas d'apprentissage, donc pas de fuite)."""
        out = pd.DataFrame(index=df.index)
        out["VehPower"] = np.minimum(df["VehPower"], 9).astype(str)
        out["VehAge"] = pd.cut(df["VehAge"], [-1, 0, 10, np.inf], labels=["0", "1-10", "11+"]).astype(str)
        out["DrivAge"] = pd.cut(df["DrivAge"], [17, 20, 25, 30, 40, 50, 70, np.inf],
                                labels=["18-20", "21-25", "26-30", "31-40", "41-50", "51-70", "71+"]).astype(str)
        out["BonusMalus"] = np.minimum(df["BonusMalus"], 150).astype(float)
        out["LogDensity"] = np.log(df["Density"].astype(float))
        out["Area"] = df["Area"].map({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}).astype(float)
        for c in ["VehBrand", "VehGas", "Region"]:
            out[c] = df[c].astype(str)
        return out
''')

GLM_PREDICT = textwrap.dedent('''
    from pathlib import Path
    import joblib, numpy as np
    import importlib.util
    _HERE = Path(__file__).resolve().parent
    _spec = importlib.util.spec_from_file_location("features", _HERE / "features.py")
    _features = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_features)
    _MODEL = joblib.load(_HERE / "model.joblib")

    def predict(df):
        """Nombre attendu de sinistres = fréquence prédite x exposition (plafonnée à 1)."""
        X = _features.prepare(df)
        freq = _MODEL.predict(X)
        return freq * np.minimum(df["Exposure"].to_numpy(float), 1.0)
''')

LGB_FEATURES = textwrap.dedent('''
    import numpy as np
    import pandas as pd
    CATS = ["VehBrand", "VehGas", "Region", "Area"]

    def prepare(df: pd.DataFrame) -> pd.DataFrame:
        out = df[["VehPower", "VehAge", "DrivAge", "BonusMalus"]].astype(float).copy()
        out["LogDensity"] = np.log(df["Density"].astype(float))
        for c in CATS:
            out[c] = pd.Categorical(df[c].astype(str))
        return out
''')

LGB_PREDICT = GLM_PREDICT.replace("_MODEL.predict(X)", "_MODEL.predict(X)")


def _write(pkg: Path, features_src: str, predict_src: str):
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "features.py").write_text(features_src)
    (pkg / "predict.py").write_text(predict_src)


def _load_module(path: Path, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit_glm(train: pd.DataFrame, pkg: Path, features_src: str = GLM_FEATURES, offset: bool = True):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import PoissonRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    _write(pkg, features_src, GLM_PREDICT)
    feats = _load_module(pkg / "features.py", "features_glm")
    X = feats.prepare(train)
    cats = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    nums = [c for c in X.columns if c not in cats]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), cats),
        ("num", StandardScaler(), nums),
    ])
    model = Pipeline([("pre", pre), ("glm", PoissonRegressor(alpha=1e-6, max_iter=1000))])
    expo = np.minimum(train["Exposure"].to_numpy(float), 1.0)
    y = np.minimum(train["ClaimNb"].to_numpy(float), 4)
    if offset:
        model.fit(X, y / expo, glm__sample_weight=expo)
    else:  # défaut volontaire (dossier saboté) : exposition ignorée
        model.fit(X, y)
    joblib.dump(model, pkg / "model.joblib")
    return model


def fit_lgbm(train: pd.DataFrame, pkg: Path, seed: int = 0):
    import lightgbm as lgb

    _write(pkg, LGB_FEATURES, LGB_PREDICT)
    feats = _load_module(pkg / "features.py", "features_lgb")
    rng = np.random.default_rng(seed)
    is_val = rng.random(len(train)) < 0.1
    X = feats.prepare(train)
    expo = np.minimum(train["Exposure"].to_numpy(float), 1.0)
    y = np.minimum(train["ClaimNb"].to_numpy(float), 4)
    model = lgb.LGBMRegressor(objective="poisson", learning_rate=0.05, n_estimators=2000,
                              num_leaves=31, min_child_samples=500, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=seed, n_jobs=2, verbose=-1)
    model.fit(X[~is_val], y[~is_val] / expo[~is_val], sample_weight=expo[~is_val],
              eval_X=(X[is_val],), eval_y=(y[is_val] / expo[is_val],), eval_sample_weight=[expo[is_val]],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    joblib.dump(model, pkg / "model.joblib")
    return model


def evaluate_package(pkg: Path, test: pd.DataFrame) -> dict:
    """Évaluation indépendante d'un dossier : on importe predict.py et on recalcule tout."""
    mod = _load_module(pkg / "predict.py", f"predict_{pkg.name}")
    test_in = test.drop(columns=["ClaimNb"])
    mu = np.asarray(mod.predict(test_in), float)
    return all_metrics(test["ClaimNb"].to_numpy(float), mu, test["Exposure"].to_numpy(float)), mu


def build_baselines(outdir: Path) -> dict:
    train, test = reference_split()
    res = {}
    fit_glm(train, outdir / "baseline_glm")
    res["glm"], mu_glm = evaluate_package(outdir / "baseline_glm", test)
    fit_lgbm(train, outdir / "baseline_lgbm")
    res["lgbm"], mu_lgb = evaluate_package(outdir / "baseline_lgbm", test)
    # Modèle nul (fréquence moyenne du train) pour situer les gains
    f0 = np.minimum(train["ClaimNb"], 4).sum() / np.minimum(train["Exposure"], 1).sum()
    mu0 = f0 * np.minimum(test["Exposure"].to_numpy(float), 1)
    res["null"] = all_metrics(test["ClaimNb"].to_numpy(float), mu0, test["Exposure"].to_numpy(float))
    for k in ("glm", "lgbm"):
        (outdir / f"baseline_{k}" / "eval.json").write_text(json.dumps(res[k], indent=2))
    np.save(outdir / "mu_baseline_glm.npy", mu_glm)
    np.save(outdir / "mu_baseline_lgbm.npy", mu_lgb)
    return res
