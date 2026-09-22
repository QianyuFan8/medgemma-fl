# Rscript methylation/ridge.R prepared_dir output_dir [validation|test]
options(warn=1)
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value=TRUE)[1])
source(file.path(dirname(script), "runtime.R"))
args <- commandArgs(TRUE)
data_dir <- args[1]; out <- args[2]; split <- if(length(args)>2) args[3] else "validation"
stopifnot(split %in% c("validation", "test"))
if (dir.exists(out)) stop("Output exists: ", out)
for(p in c("glmnet", "data.table")) if(!requireNamespace(p, quietly=TRUE)) stop("Install ", p)
read <- function(s) as.data.frame(data.table::fread(file.path(data_dir, paste0(s, ".csv"))))
tr <- read("train"); va <- read("validation")
features <- names(tr)[-(1:2)]
x <- function(d) {
  stopifnot(identical(names(d)[-(1:2)], features))
  b <- as.matrix(d[, features, drop=FALSE]); b <- pmin(pmax(b, 1e-6), 1-1e-6)
  log2(b/(1-b))
}
classes <- sort(unique(tr$diagnosis))
lambda <- 10^seq(2, -4, length.out=25)
fit <- glmnet::glmnet(x(tr), factor(tr$diagnosis, levels=classes), alpha=0,
                      family=if(length(classes)==2) "binomial" else "multinomial", lambda=lambda,
                      standardize=TRUE)
pred <- predict(fit, newx=x(va), s=lambda, type="class")
ba <- function(p) mean(vapply(classes, function(c) mean(p[va$diagnosis==c] == c), numeric(1)))
scores <- apply(pred, 2, ba); best <- which.max(scores)
# Do NOT refit on validation: MedGemma also trains on train only.
d <- if(split == "validation") va else read("test")
p <- as.character(predict(fit, newx=x(d), s=lambda[best], type="class"))
dir.create(out, recursive=TRUE)
data.table::fwrite(data.frame(lambda=lambda, validation_balanced_accuracy=scores), file.path(out,"tuning.csv"))
data.table::fwrite(data.frame(patient_id=d$patient_id, truth=d$diagnosis, prediction=p), file.path(out,"predictions.csv"))
saveRDS(list(model=fit, lambda=lambda[best], features=features), file.path(out,"ridge.rds"))
message("Selected lambda using validation only: ", lambda[best], "; predictions for ", split)
