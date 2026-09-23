# Centralized training-only limma selection, NOT federated differential analysis.
# Usage: Rscript methylation/prepare.R config.json
options(warn = 1)
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value=TRUE)[1])
source(file.path(dirname(script), "runtime.R"))
for (p in c("data.table", "jsonlite", "limma"))
  if (!requireNamespace(p, quietly = TRUE)) stop("Install R package: ", p)
cfg <- jsonlite::fromJSON(commandArgs(TRUE)[1])
defaults <- list(seed=20260922L, top_k=100L, fdr=0.05, variance_cap=20000L,
                 max_sample_missing=0.2, max_probe_missing=0.05, detection_p=0.01)
cfg <- utils::modifyList(defaults, cfg)
stopifnot(cfg$top_k >= 2, cfg$fdr > 0, cfg$fdr <= 1)
if (dir.exists(cfg$output)) stop("Output exists; choose a new directory: ", cfg$output)
m <- as.data.frame(data.table::fread(cfg$metadata, colClasses="character"))
required <- c("sample_id", "patient_id", "diagnosis", "platform", "matrix_file", "beta_column", "include", "sample_type")
if (!all(required %in% names(m))) stop("Missing sample-sheet columns: ", paste(setdiff(required, names(m)), collapse=", "))
m <- m[!is.na(m$include) & tolower(m$include) == "true", , drop=FALSE]
if (!is.null(cfg$diagnoses)) m <- m[m$diagnosis %in% cfg$diagnoses, , drop=FALSE]
if (!is.null(cfg$diagnoses) && !all(cfg$diagnoses %in% m$diagnosis))
  stop("Requested diagnoses missing from reviewed metadata: ", paste(setdiff(cfg$diagnoses, m$diagnosis), collapse=", "))
if (anyNA(m[, required]) || any(!nzchar(m$diagnosis)) || !nrow(m)) stop("Missing required metadata.")
if (length(unique(tolower(m$platform))) != 1L) stop("Run one platform at a time.")
if (anyDuplicated(m$patient_id) || anyDuplicated(m$sample_id)) stop("Require one reviewed sample per patient.")
if (any(!m$sample_type %in% c("01", "03", "09"))) stop("Review non-primary samples before including them.")
if ("label_status" %in% names(m) && any(is.na(m$label_status) | m$label_status != "matched")) stop("Unresolved diagnosis labels.")
if (length(unique(m$diagnosis)) < 2L || any(table(m$diagnosis) < 5L)) stop("Need >=2 classes and >=5 patients per class.")
if (!isTRUE(cfg$metadata_reviewed)) stop("Confirm metadata_reviewed=true after checking label evidence and exclusions.")
root <- if (is.null(cfg$matrix_root)) dirname(normalizePath(cfg$metadata)) else cfg$matrix_root
paths <- ifelse(grepl("^/", m$matrix_file), m$matrix_file, file.path(root, m$matrix_file))
if (any(!file.exists(paths))) stop("Matrix not found; check matrix_root: ", paste(unique(paths[!file.exists(paths)]), collapse=", "))
m$matrix_file <- normalizePath(paths)
# jsonlite parses JSON [] as list(), which is not a valid data.frame index.
# Both an omitted field and an explicit empty array mean no covariates.
covariates <- if (!length(cfg$covariates)) character() else cfg$covariates
if (!is.character(covariates) || anyNA(covariates) || any(!nzchar(covariates)))
  stop("covariates must be an array of column-name strings or an empty array.")
if (!all(covariates %in% names(m)) || anyNA(m[, covariates, drop=FALSE])) stop("Missing covariates.")

# Deterministic patient split BEFORE any fitted feature filtering; one sample per patient.
set.seed(cfg$seed)
m$split <- "train"
for (dx in sort(unique(m$diagnosis))) {
  ix <- sample(which(m$diagnosis == dx))
  n <- max(1L, floor(length(ix)*0.2))
  m$split[ix[seq_len(n)]] <- "test"
  m$split[ix[n + seq_len(n)]] <- "validation"
}
mats <- list()
for (file in unique(m$matrix_file)) {
  s <- m[m$matrix_file == file, , drop=FALSE]
  header <- names(data.table::fread(file, nrows=0))
  dp <- if ("detection_column" %in% names(s)) s$detection_column else sub("\\.Ave_Beta$", ".Detection.Pval", s$beta_column)
  has_dp <- !is.na(dp) & nzchar(dp) & dp %in% header
  message("Reading ", file)
  d <- data.table::fread(file, select=unique(c(header[1], s$beta_column, dp[has_dp])))
  ids <- as.character(d[[header[1]]]); keep <- grepl("^cg[0-9]+$", ids)
  if (anyDuplicated(ids[keep])) stop("Duplicate CpG IDs: ", file)
  x <- as.matrix(d[, s$beta_column, with=FALSE]); storage.mode(x) <- "double"
  if (any(x < 0 | x > 1, na.rm=TRUE)) stop("Beta outside [0,1].")
  x[!is.finite(x)] <- NA_real_
  for (j in which(has_dp)) {
    p <- d[[dp[j]]]; x[is.na(p) | p > cfg$detection_p, j] <- NA_real_
  }
  if (!all(has_dp)) warning("Some beta columns have no detection p-values; detection filtering unavailable for those samples.")
  rownames(x) <- ids; colnames(x) <- s$sample_id
  mats[[file]] <- x[keep, , drop=FALSE]
}
common <- Reduce(intersect, lapply(mats, rownames))
beta <- do.call(cbind, lapply(mats, function(x) x[common, , drop=FALSE]))[, m$sample_id, drop=FALSE]
rm(mats); gc(FALSE)
if (!nrow(beta)) stop("No shared CpGs within platform.")
# Fixed sample QC independently of labels. Retain exclusions in audit.
m$missing_fraction <- colMeans(is.na(beta))
m$qc_pass <- m$missing_fraction <= cfg$max_sample_missing
audit <- m
beta <- beta[, m$qc_pass, drop=FALSE]; m <- m[m$qc_pass, , drop=FALSE]
classes <- sort(unique(audit$diagnosis))
counts <- table(factor(m$diagnosis, levels=classes), factor(m$split, levels=c("train", "validation", "test")))
if (any(counts == 0) || any(counts[, "train"] < 3)) stop("QC left missing classes or fewer than 3 train patients/class; review cohort.")
train <- m$split == "train"
beta <- beta[rowMeans(is.na(beta[, train, drop=FALSE])) <= cfg$max_probe_missing, , drop=FALSE]
fill <- apply(beta[, train, drop=FALSE], 1, median, na.rm=TRUE)
for (j in seq_len(ncol(beta))) { na <- is.na(beta[, j]); beta[na, j] <- fill[na] }
to_m <- function(x) { x <- pmin(pmax(x, 1e-6), 1-1e-6); log2(x/(1-x)) }
z <- to_m(beta[, train, drop=FALSE])
variance <- apply(z, 1, var)
idx <- order(variance, decreasing=TRUE); idx <- idx[is.finite(variance[idx]) & variance[idx] > 0]
idx <- head(idx, cfg$variance_cap)
z <- z[idx, , drop=FALSE]
design_data <- data.frame(diagnosis=factor(m$diagnosis[train], levels=classes))
for (v in covariates) {
  values <- m[[v]][train]
  design_data[[v]] <- if (!is.null(cfg$numeric_covariates) && v %in% cfg$numeric_covariates) as.numeric(values) else factor(values)
}
design <- model.matrix(reformulate(c("diagnosis", covariates)), design_data)
if (nrow(design) != sum(train) || qr(design)$rank != ncol(design) || nrow(design) <= ncol(design))
  stop("Invalid/confounded design or no residual degrees of freedom.")
fit <- limma::eBayes(limma::lmFit(z, design))
coefs <- which(attr(design, "assign") == 1L)
rank <- limma::topTable(fit, coef=coefs, number=Inf, sort.by=if(length(coefs)==1L) "P" else "F", adjust.method="BH")
rank$probe_id <- rownames(rank)
significant <- rank$probe_id[rank$adj.P.Val < cfg$fdr]
if (length(significant) < cfg$top_k) stop("Only ", length(significant), " significant CpGs; requested ", cfg$top_k, ". No non-significant fallback.")
panel <- head(significant, cfg$top_k)
dir.create(cfg$output, recursive=TRUE)
write_tsv <- function(x, name) data.table::fwrite(x, file.path(cfg$output, name), sep="\t")
write_tsv(audit, "sample_audit.tsv")
write_tsv(m, "splits.tsv")
write_tsv(rank, "limma_training_only.tsv")
write_tsv(data.frame(probe_id=panel, training_median=fill[panel]), "selected_cpgs.tsv")
for (split in c("train", "validation", "test")) {
  ix <- which(m$split == split)
  # Same rounded beta values supplied to ridge and to MedGemma.
  out <- data.frame(patient_id=m$patient_id[ix], diagnosis=m$diagnosis[ix],
                    t(round(beta[panel, ix, drop=FALSE], 6)), check.names=FALSE)
  data.table::fwrite(out, file.path(cfg$output, paste0(split, ".csv")))
}
saveRDS(list(probes=panel, median=fill[panel], config=cfg, classes=classes), file.path(cfg$output, "preprocessing.rds"))
jsonlite::write_json(cfg, file.path(cfg$output, "config.json"), pretty=TRUE, auto_unbox=TRUE)
writeLines(capture.output(sessionInfo()), file.path(cfg$output, "R_sessionInfo.txt"))
print(counts)
message("Prepared training-only limma panel: ", length(panel), " CpGs. No model test performance computed.")
