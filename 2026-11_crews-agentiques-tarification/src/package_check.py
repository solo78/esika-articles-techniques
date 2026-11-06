"""Audit déterministe d'un dossier de modèle, lancé dans un processus isolé.

Usage : python -I package_check.py <dossier_modele> <train.parquet> <test.parquet>
Imprime un JSON. Sert aux portes de qualité de la crew autonome et à l'audit a posteriori.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import xlogy


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dev(y, mu):
    mu = np.clip(mu, 1e-12, None)
    return float(200 * np.mean(xlogy(y, y / mu) - (y - mu)))


def gini(y, mu, e):
    o = np.argsort(mu / e, kind="mergesort")
    ce = np.concatenate([[0], np.cumsum(e[o])]) / e.sum()
    cy = np.concatenate([[0], np.cumsum(y[o])]) / y.sum()
    return float(1 - 2 * np.trapezoid(cy, ce))


def main(pkg, train_path, test_path):
    pkg = Path(pkg)
    out = {"errors": []}
    test = pd.read_parquet(test_path)
    train = pd.read_parquet(train_path).sample(n=100_000, random_state=0) if train_path != "-" else None
    y = test["ClaimNb"].to_numpy(float)
    e = test["Exposure"].to_numpy(float)
    X = test.drop(columns=["ClaimNb"])
    try:
        feats = load(pkg / "features.py", "features")
        F = feats.prepare(X.head(2000).copy())
        out["feature_columns"] = list(map(str, F.columns))
        out["features_use_id_or_target"] = any(c.lower() in ("idpol", "claimnb") for c in out["feature_columns"])
    except Exception as ex:  # noqa: BLE001
        out["errors"].append(f"features.prepare : {type(ex).__name__}: {ex}")
    try:
        pred = load(pkg / "predict.py", "predict_mod")
        mu = np.asarray(pred.predict(X.copy()), float)
        out["n_pred"] = int(mu.size)
        out["finite"] = bool(np.isfinite(mu).all())
        out["positive"] = bool((mu > 0).all())
        if mu.size == len(y) and out["finite"]:
            out["test"] = {"deviance_x100": dev(y, mu), "gini": gini(y, mu, e),
                           "ae_global": float(y.sum() / mu.sum())}
            fr = mu / e
            o = np.argsort(fr, kind="mergesort")
            out["test"]["ae_deciles"] = [float(y[g].sum() / mu[g].sum()) for g in np.array_split(o, 10)]
            np.save(pkg / "_mu_test.npy", mu)
        # Invariance à l'identifiant : on permute IDpol, les prédictions ne doivent pas bouger
        Xp = X.copy()
        Xp["IDpol"] = np.random.default_rng(1).permutation(Xp["IDpol"].to_numpy())
        mup = np.asarray(pred.predict(Xp), float)
        out["id_sensitivity"] = float(np.max(np.abs(mup - mu) / np.maximum(mu, 1e-12)))
        # Proportionnalité à l'exposition : doubler une exposition <= 0,5 doit doubler l'espérance
        m = X["Exposure"].to_numpy() <= 0.5
        Xe = X[m].copy()
        Xe["Exposure"] = Xe["Exposure"] * 2
        mue = np.asarray(pred.predict(Xe), float)
        r = mue / mu[m]
        out["exposure_ratio_median"] = float(np.median(r))
        out["exposure_ratio_maxdev"] = float(np.max(np.abs(r - 2)))
        if train is not None:
            mut = np.asarray(pred.predict(train.drop(columns=["ClaimNb"])), float)
            out["train_ae"] = float(train["ClaimNb"].sum() / mut.sum())
            out["train_deviance_x100"] = dev(train["ClaimNb"].to_numpy(float), mut)
    except Exception as ex:  # noqa: BLE001
        out["errors"].append(f"predict : {type(ex).__name__}: {str(ex)[:500]}")
    print(json.dumps(out))


if __name__ == "__main__":
    main(*sys.argv[1:4])
