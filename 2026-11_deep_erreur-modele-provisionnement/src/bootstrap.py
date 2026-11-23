"""Brique 6 : bootstrap du LASSO et matrice bootstrap (papier, sections 7.2 et 7.3, éq. 28 à 33)."""
import numpy as np
from . import bma as B
from .lasso import fit_paths

INIT_WIDEN = 1.4      # portes provisoires élargies avant recentrage (section 7.2)
MIN_MODELS = 5        # règle de rejet : au moins 5 modèles de probabilité > 1e-4
MIN_PROB = 1e-4


def residuals(prim, y, past, power):
    """Résidus standardisés du modèle 1se : rho = (y - mu) / mu^(power/2), centrés (le LASSO est biaisé).

    power = 2 : variance Gamma (phi mu^2), protocole littéral du papier (section 7.2) ;
    power = 1 : variance proportionnelle à mu (bootstrap ODP), retenue quand la vérification d'homogénéité
    des résidus exigée par le papier échoue avec power = 2 (cas des données fictives, voir retro.md).
    La constante de dispersion se simplifie dans Y = mu + rho mu^(power/2)."""
    p = past.ravel()
    yp = y.ravel()[p]
    mu = prim["fam"]["preds"][prim["i1se"], p]
    rho = (yp - mu) / mu ** (power / 2)
    return mu, rho - rho.mean()


def homogeneity(prim, y, past, n, power):
    """Écart type des résidus réduits par bande de développement (contrôle i.i.d. du papier, section 7.2)."""
    mu, rho = residuals(prim, y, past, power)
    r = rho / rho.std()
    _, j = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1), indexing="ij")
    jj = j.ravel()[past.ravel()]
    bands = ((1, 2), (3, 10), (11, 25), (26, n))
    return {f"dev {a}-{b}": float(r[(jj >= a) & (jj <= b)].std()) for a, b in bands}


def pseudo_data(prim, y, past, B_, seed, power=2):
    """Pseudo-triangles semi-paramétriques : Y[b] = mu + rho[b] mu^(power/2), rho rééchantillonnés avec remise.

    Les rares valeurs négatives sont ramenées à 0,1 % de mu (compté)."""
    mu, rho = residuals(prim, y, past, power)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rho), size=(B_, len(rho)))
    Y = mu + rho[idx] * mu ** (power / 2)
    floor = 1e-3 * mu
    n_floor = int((Y < floor).sum())
    return np.maximum(Y, floor), rho, n_floor


def fit(prim, y, past, n, B_, seed, workdir, ncores=2, power=2):
    """Ajuste le LASSO sur B_ pseudo-triangles avec la suite de lambda du triangle observé."""
    Yb, rho, n_floor = pseudo_data(prim, y, past, B_, seed, power)
    preds, stats = fit_paths(prim["X"], past, Yb, prim["fam"]["lam"], workdir, ncores)
    return dict(Yb=Yb, rho=rho, n_floor=n_floor, preds=preds, stats=stats, power=power, y_obs=y, past=past)


def matrix(prim, boot, past, n, prior, widen=1.0):
    """Matrice bootstrap pour un a priori donné, et décomposition PaE / IMSE (éq. 28 à 33).

    Étapes de la section 7.2 : (1) a posteriori de chaque réplication avec portes provisoires x 1,4 ;
    (2) facteur d'échelle ramenant la moyenne des moyennes a posteriori bootstrap à la moyenne
    a posteriori originale ; (3) application du facteur (décalage de l'intercept) ; (4) portes
    définitives ; (5) a posteriori ; (6) rejet des réplications à moins de 5 modèles.
    `widen` élargit les portes définitives (sensibilité, section 8.4)."""
    p = past.ravel()
    fut = ~p
    pr = prim["priors"][prior]
    s, phi, m0 = pr["s"], prim["fam"]["phi"], prim["m0"]
    a0, l1 = boot["stats"][..., 0], boot["stats"][..., 1]
    preds = boot["preds"]                                     # B x Q x n2 (float32)
    nb_, Q = preds.shape[:2]
    agg_ref = prim["agg_ref"]

    def one(b, scale):
        """log a posteriori non normalisée, provisions et agrégats des portes de la réplication b.

        Choix d'interprétation : le facteur d'échelle recentre les PRÉVISIONS (provisions et agrégats des
        portes) ; la vraisemblance et l'a priori restent ceux de l'ajustement du modèle à son propre
        pseudo-triangle (le recentrage corrige un biais de prévision, pas l'ajustement)."""
        P = preds[b].astype(np.float64)
        valid = np.isfinite(P).all(-1)
        Pv = np.where(valid[:, None], P, 1.0)
        ll = np.where(valid, B.gamma_loglik(boot["Yb"][b], Pv[:, p], phi), -np.inf)
        lp = B.log_prior(a0[b], l1[b], s, m0)
        res = np.where(valid, Pv[:, fut].sum(-1) * scale, np.nan)
        agg = B.aggregates(np.where(valid[:, None], P * scale, np.nan), n)
        return ll + lp, res, agg, valid

    # passe 1 : portes provisoires (x 1,4)
    m1 = []
    for b in range(nb_):
        logw, res, agg, valid = one(b, 1.0)
        w = B.posterior(logw, B.admissible(agg, agg_ref, INIT_WIDEN * widen) & valid)
        if w.sum() > 0:
            m1.append(B.bma(w, res)[0])
    scale = pr["mean"] / np.mean(m1)
    # passes 2 à 6 : recentrage, portes définitives, a posteriori, règle de rejet
    W = np.zeros((nb_, Q))
    RES = np.full((nb_, Q), np.nan)
    keep = np.zeros(nb_, bool)
    n_adm = np.zeros(nb_, int)
    m_b, v_b = np.full(nb_, np.nan), np.full(nb_, np.nan)
    for b in range(nb_):
        logw, res, agg, valid = one(b, scale)
        adm = B.admissible(agg, agg_ref, widen) & valid
        n_adm[b] = adm.sum()
        RES[b] = np.where(adm, res, np.nan)
        w = B.posterior(logw, adm)
        if (w > MIN_PROB).sum() >= MIN_MODELS:
            keep[b] = True
            W[b] = w
            m_b[b], v_b[b], _ = B.bma(w, res)
    mk, vk = m_b[keep], v_b[keep]
    m = mk.mean()                                  # éq. (28)
    s2_imse = vk.mean()                            # éq. (29)
    s2_pa = mk.var(ddof=1)                         # éq. (31)
    cell = pr["cell"]
    if boot.get("power", 2) == 2:
        w_pr = np.sqrt(phi * np.sum(cell[fut] ** 2)) / m   # erreur de processus, loi Gamma (papier)
    else:   # même fonction de variance que le bootstrap : Var = phi1 mu, phi1 estimé par les résidus de Pearson
        mu_p, rho1 = residuals(prim, boot["y_obs"], boot["past"], 1)
        phi1 = np.sum(rho1 ** 2) / (len(rho1) - prim["fam"]["n_nz_1se"] - 1)
        w_pr = np.sqrt(phi1 * np.sum(cell[fut])) / m
    w_imse, w_pa = np.sqrt(s2_imse) / m, np.sqrt(s2_pa) / m
    return dict(prior=prior, widen=widen, scale=scale, n_kept=int(keep.sum()), B=preds.shape[0], m=m,
                w_imse=w_imse, w_pa=w_pa, w_pa_imse=np.sqrt(w_pa ** 2 + w_imse ** 2), w_pr=w_pr,
                w_sub=np.sqrt(w_pa ** 2 + w_imse ** 2 + w_pr ** 2), res=RES,
                W=W, keep=keep, m_b=m_b, v_b=v_b, n_adm=n_adm)
