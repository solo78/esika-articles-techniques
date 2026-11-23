# Ajustement du LASSO Poisson par glmnet (papier, section 6.1.2 ; code des auteurs).
# Appelé par src/lasso.py. Échanges par fichiers binaires dans un dossier de travail.
#
# Mode "cv"  : un seul triangle ; chemin glmnet (nlambda = 50, prolongé par min * 0.85^(1:50)),
#              validation croisée à 8 blocs (déviance Poisson), modèles minCV et 1se,
#              puis dispersion phi : GLM Gamma (lien log) sur la structure du 1se,
#              estimation par MASS::gamma.dispersion (maximum de vraisemblance).
# Mode "path": plusieurs pseudo-triangles (bootstrap) ajustés sur la même suite de lambda.
#
# Comme les auteurs, npad valeurs sont intercalées (géométriquement) entre deux lambda successifs :
# la densité de la grille conditionne le nombre de modèles voisins qui se partagent la masse a posteriori.
suppressMessages({library(glmnet); library(MASS)})
args <- commandArgs(trailingOnly = TRUE)
wd <- args[1]
meta <- scan(file.path(wd, "meta.txt"), what = "", quiet = TRUE)
nall <- as.integer(meta[1]); p <- as.integer(meta[2]); npast <- as.integer(meta[3])
nb <- as.integer(meta[4]); mode <- meta[5]; thresh <- as.numeric(meta[6])
rd <- function(f, n) { con <- file(file.path(wd, f), "rb"); on.exit(close(con)); readBin(con, "double", n) }
X <- matrix(rd("X.bin", nall * p), nall, p)
past <- rd("past.bin", nall) > 0.5
Y <- matrix(rd("Y.bin", npast * nb), npast, nb)
Xp <- X[past, , drop = FALSE]

if (mode == "cv") {
  y <- Y[, 1]
  f0 <- glmnet(Xp, y, family = "poisson", nlambda = 50, lambda.min.ratio = 0, thresh = thresh,
               maxit = 1e6, standardize = FALSE)
  lv <- sort(unique(c(f0$lambda, min(f0$lambda) * 0.85^(1:50))), decreasing = TRUE)
  npad <- as.integer(meta[7])
  if (npad > 0) {  # intercalage géométrique de npad valeurs entre deux lambda successifs (code des auteurs)
    lv <- sort(unique(unlist(lapply(seq_len(length(lv) - 1), function(i)
      exp(seq(log(lv[i]), log(lv[i + 1]), length.out = npad + 2))))), decreasing = TRUE)
  }
  set.seed(87)
  cv <- cv.glmnet(Xp, y, family = "poisson", lambda = lv, nfolds = 8, thresh = thresh,
                  maxit = 1e6, standardize = FALSE)
  lv <- cv$glmnet.fit$lambda
  i1 <- which(lv == cv$lambda.1se); im <- which(lv == cv$lambda.min)
  b1 <- as.numeric(coef(cv$glmnet.fit, s = lv[i1]))[-1]
  nz <- which(b1 != 0)
  mu1 <- as.numeric(predict(cv$glmnet.fit, Xp, s = lv[i1], type = "response"))
  dat <- data.frame(y = y, Xp[, nz, drop = FALSE])
  g <- tryCatch(glm(y ~ ., data = dat, family = Gamma(link = "log"), mustart = mu1,
                    control = glm.control(maxit = 100)), error = function(e) NULL)
  if (!is.null(g)) {
    phi <- as.numeric(gamma.dispersion(g))
  } else {
    # repli (rare : GLM Gamma divergent) : estimateur de Pearson sur les valeurs ajustées du LASSO 1se
    phi <- sum(((y - mu1) / mu1)^2) / (length(y) - length(nz) - 1)
    g <- list(converged = "REPLI_PEARSON")
  }
  writeLines(as.character(c(i1, im, format(phi, digits = 10), length(nz), g$converged)), file.path(wd, "cv.txt"))
  write.table(cbind(lv, cv$cvm, cv$cvsd), file.path(wd, "cvtable.txt"), row.names = FALSE, col.names = FALSE)
  fits <- list(cv$glmnet.fit)
} else {
  lv <- rd("lambda.bin", as.integer(meta[7]))
  fits <- NULL
}
Q <- length(lv)
pcon <- file(file.path(wd, "preds.bin"), "wb"); scon <- file(file.path(wd, "stats.bin"), "wb")
write_fit <- function(f) {
  q <- length(f$lambda)
  pr <- matrix(NA_real_, nall, Q); st <- matrix(NA_real_, 3, Q)
  pr[, 1:q] <- predict(f, X, type = "response")
  B <- f$beta
  st[1, 1:q] <- f$a0; st[2, 1:q] <- colSums(abs(B)); st[3, 1:q] <- f$df
  writeBin(as.numeric(pr), pcon, size = 4); writeBin(as.numeric(st), scon)
}
if (is.null(fits)) {
  # un pseudo-triangle à la fois : ajustement, écriture, libération de la mémoire
  for (b in seq_len(nb)) {
    f <- glmnet(Xp, Y[, b], family = "poisson", lambda = lv, thresh = thresh, maxit = 1e6, standardize = FALSE)
    write_fit(f); rm(f); if (b %% 10 == 0) gc()
  }
} else {
  for (f in fits) write_fit(f)
}
close(pcon); close(scon)
con <- file(file.path(wd, "lambda.bin"), "wb"); writeBin(lv, con); close(con)
