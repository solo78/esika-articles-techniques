"""Brique 1 : générer la famille de modèles par le LASSO (papier, section 6.1.2, éq. 15 et 16).

Interface Python vers src/lasso_glmnet.R (glmnet, LASSO Poisson, comme les auteurs).
Chaque valeur de pénalité lambda_0^(q) donne un modèle M_q : ses prévisions sur toutes
les cellules, son intercept, la norme L1 de ses autres coefficients et son nombre de variables.
"""
import os
import subprocess
import tempfile
import numpy as np

R_SCRIPT = os.path.join(os.path.dirname(__file__), "lasso_glmnet.R")


def _write(wd, X, past, Y, mode, thresh, lam=None, npad=0):
    nall, p = X.shape
    npast = int(past.sum())
    nb = Y.shape[0]
    meta = [nall, p, npast, nb, mode, thresh] + ([len(lam)] if lam is not None else [npad])
    with open(os.path.join(wd, "meta.txt"), "w") as f:
        f.write(" ".join(str(m) for m in meta))
    np.asfortranarray(X, dtype=np.float64).T.tofile(os.path.join(wd, "X.bin"))  # ordre colonne
    past.ravel().astype(np.float64).tofile(os.path.join(wd, "past.bin"))
    np.ascontiguousarray(Y, dtype=np.float64).tofile(os.path.join(wd, "Y.bin"))  # npast x nb, ordre colonne
    if lam is not None:
        np.asarray(lam, np.float64).tofile(os.path.join(wd, "lambda.bin"))


def _read(wd, nall, nb):
    lam = np.fromfile(os.path.join(wd, "lambda.bin"))
    Q = len(lam)
    preds = np.fromfile(os.path.join(wd, "preds.bin"), dtype=np.float32).reshape(nb, Q, nall)
    stats = np.fromfile(os.path.join(wd, "stats.bin")).reshape(nb, Q, 3)
    return lam, preds, stats


def fit_cv(X, past, y_past, workdir, npad=5):
    """LASSO sur le triangle observé : chemin, validation croisée (minCV, 1se) et dispersion phi.

    npad : nombre de valeurs de lambda intercalées entre deux valeurs successives du chemin (5 chez les auteurs)."""
    wd = tempfile.mkdtemp(dir=workdir)
    _write(wd, X, past, y_past[None, :], "cv", 1e-8, npad=npad)
    subprocess.run(["Rscript", R_SCRIPT, wd], check=True)
    lam, preds, stats = _read(wd, X.shape[0], 1)
    preds = preds.astype(np.float64)
    lines = open(os.path.join(wd, "cv.txt")).read().split()
    cvt = np.loadtxt(os.path.join(wd, "cvtable.txt"))
    return dict(lam=lam, preds=preds[0], a0=stats[0, :, 0], l1=stats[0, :, 1], df=stats[0, :, 2],
                i1se=int(lines[0]) - 1, imin=int(lines[1]) - 1, phi=float(lines[2]),
                n_nz_1se=int(lines[3]), glm_converged=lines[4], cvtable=cvt)


def fit_paths(X, past, Y_past, lam, workdir, ncores=2):
    """LASSO sur plusieurs pseudo-triangles (bootstrap), même suite de lambda, en parallèle."""
    nb = Y_past.shape[0]
    chunks = np.array_split(np.arange(nb), ncores)
    procs, dirs = [], []
    for c in chunks:
        wd = tempfile.mkdtemp(dir=workdir)
        _write(wd, X, past, Y_past[c], "path", 1e-7, lam)
        procs.append(subprocess.Popen(["Rscript", R_SCRIPT, wd]))
        dirs.append((wd, len(c)))
    for pr in procs:
        if pr.wait() != 0:
            raise RuntimeError("échec de glmnet")
    out_p, out_s = [], []
    for wd, n in dirs:
        _, p_, s_ = _read(wd, X.shape[0], n)
        out_p.append(p_)
        out_s.append(s_)
        for f in ("X.bin", "preds.bin"):
            os.remove(os.path.join(wd, f))
    return np.concatenate(out_p), np.concatenate(out_s)
