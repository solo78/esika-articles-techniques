# Diagnostic : provision du modèle 1se avec les effets de calendrier prolongés (règle du papier, X** = 0)
# ou figés à leur niveau du dernier trimestre observé. Mêmes réglages que src/lasso_glmnet.R.
suppressMessages(library(glmnet))
wd <- commandArgs(trailingOnly = TRUE)[1]
meta <- scan(file.path(wd, "meta.txt"), quiet = TRUE)
nall <- meta[1]; p <- meta[2]
rd <- function(f, n) { con <- file(file.path(wd, f), "rb"); on.exit(close(con)); readBin(con, "double", n) }
X <- matrix(rd("X.bin", nall * p), nall, p); Xf <- matrix(rd("Xf.bin", nall * p), nall, p)
past <- rd("past.bin", nall) > 0.5; y <- rd("Y.bin", sum(past))
Xp <- X[past, ]
f0 <- glmnet(Xp, y, family = "poisson", nlambda = 50, lambda.min.ratio = 0, thresh = 1e-8, maxit = 1e6, standardize = FALSE)
lv <- sort(unique(c(f0$lambda, min(f0$lambda) * 0.85^(1:50))), decreasing = TRUE)
lv <- sort(unique(unlist(lapply(seq_len(length(lv) - 1), function(i) exp(seq(log(lv[i]), log(lv[i + 1]), length.out = 7))))), decreasing = TRUE)
set.seed(87)
cv <- cv.glmnet(Xp, y, family = "poisson", lambda = lv, nfolds = 8, thresh = 1e-8, maxit = 1e6, standardize = FALSE)
fut <- !past
r1 <- sum(predict(cv, X[fut, ], s = "lambda.1se", type = "response"))
r2 <- sum(predict(cv, Xf[fut, ], s = "lambda.1se", type = "response"))
writeLines(format(c(r1, r2), digits = 12), file.path(wd, "diag.txt"))
