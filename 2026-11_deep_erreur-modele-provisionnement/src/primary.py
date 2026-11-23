"""Analyse du modèle principal (papier, sections 6.1 à 6.2.6) : famille LASSO, phi, a priori, BMA, élagage."""
import numpy as np
from . import bma as B
from .basis import standardized_design
from .lasso import fit_cv

PRIORS = ("simple", "1se", "minCV", "complexe")


def analyse(y, past, n, workdir, npad=5):
    """Renvoie un dictionnaire avec la famille de modèles, les a priori calibrés et la BMA de chaque a priori."""
    X, names = standardized_design(n, past)
    p = past.ravel()
    yp = y.ravel()[p]
    fam = fit_cv(X, past, yp, workdir, npad)
    preds = fam["preds"]                                # Q x (n*n)
    valid = np.isfinite(preds).all(1)
    Q = len(fam["lam"])
    fut = ~p
    reserves = np.where(valid, np.nansum(preds[:, fut], 1), np.nan)
    ll = np.where(valid, B.gamma_loglik(yp, np.where(valid[:, None], preds[:, p], 1.0), fam["phi"]), -np.inf)
    m0 = np.log(yp.mean())
    agg = B.aggregates(np.where(valid[:, None], preds, np.nan), n)
    i1, im = fam["i1se"], fam["imin"]
    agg_ref = {k: v[i1] for k, v in agg.items()}
    adm = B.admissible(agg, agg_ref) & valid
    out = dict(X=X, names=names, fam=fam, reserves=reserves, loglik=ll, m0=m0, agg=agg, agg_ref=agg_ref,
               admissible=adm, valid=valid, Q=Q, i1se=i1, imin=im, priors={})
    s1, int1, ok1 = B.calibrate_scale(ll, fam["a0"], fam["l1"], m0, adm, i1)
    sm, intm, okm = B.calibrate_scale(ll, fam["a0"], fam["l1"], m0, adm, im)
    ss = B.calibrate_extreme(ll, fam["a0"], fam["l1"], m0, adm, i1, "simple")
    sc = B.calibrate_extreme(ll, fam["a0"], fam["l1"], m0, adm, im, "complexe")
    for name, s, extra in (("simple", ss, {}), ("1se", s1, dict(interval=int1, hit=ok1)),
                           ("minCV", sm, dict(interval=intm, hit=okm)), ("complexe", sc, {})):
        if not np.isfinite(s):
            out["priors"][name] = dict(s=np.nan)
            continue
        logw = ll + B.log_prior(fam["a0"], fam["l1"], s, m0)
        w = B.posterior(logw, adm)
        w_all = B.posterior(logw, valid)                 # avant élagage, pour mesurer son effet
        mean, var, cov = B.bma(w, reserves)
        cell = (w[:, None] * np.nan_to_num(preds)).sum(0)  # prévisions moyennes a posteriori par cellule
        out["priors"][name] = dict(s=s, w=w, w_all=w_all, mean=mean, var=var, cov=cov, cell=cell,
                                   n_models=int((w > 1e-4).sum()), mode=int(np.argmax(w)), **extra)
    return out
