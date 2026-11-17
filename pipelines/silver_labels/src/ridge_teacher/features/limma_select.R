# Invoked once per fitted training fold by LimmaCpGSelector, never on holdouts.
# Reference: https://bioconductor.org/packages/limma
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 7L)
if (!requireNamespace("limma", quietly=TRUE) || !requireNamespace("matrixStats", quietly=TRUE))
  stop("Install the supplied environment: limma and matrixStats are required.")
n <- as.integer(args[2]); p <- as.integer(args[3])
y <- scan(args[4], what=integer(), quiet=TRUE)
k <- as.integer(args[6]); coverage <- as.numeric(args[7])
stopifnot(length(y) == n, all(sort(unique(y)) == c(0L,1L)), k > 0L,
          coverage > 0, coverage <= 1, min(table(y)) >= 2)
con <- file(args[1], "rb")
b <- readBin(con, what=numeric(), n=as.double(n)*p, size=4L, endian="little")
close(con)
stopifnot(length(b) == as.double(n)*p)
b <- matrix(b, nrow=p, ncol=n) # Python row is a patient; R column is a patient.
b[!is.finite(b)] <- NA_real_
if (any(b < 0 | b > 1, na.rm=TRUE)) stop("Beta outside [0,1].")
counts <- rowSums(!is.na(b))
v <- matrixStats::rowVars(b, na.rm=TRUE)
eligible <- which(counts >= max(4L, ceiling(n*coverage)) & is.finite(v) & v > 1e-10)
if (length(eligible) < 2L) stop("Insufficient nonconstant training CpGs after QC.")
b <- b[eligible,,drop=FALSE]
med <- matrixStats::rowMedians(b, na.rm=TRUE)
missing <- which(is.na(b), arr.ind=TRUE)
if (nrow(missing)) b[missing] <- med[missing[,1]]
# Clip only for the M-value logarithm; retain beta values for Ridge downstream.
b <- pmin(pmax(b, 1e-6), 1-1e-6)
M <- log2(b/(1-b))
rownames(M) <- as.character(eligible-1L) # Zero-based original feature positions.
design <- model.matrix(~factor(y, levels=c(0,1)))
fit <- limma::eBayes(limma::lmFit(M, design), trend=FALSE, robust=FALSE)
# BH correction is computed over ALL eligible CpGs before selecting top k.
rank <- limma::topTable(fit, coef=2L, number=Inf, sort.by="P", adjust.method="BH")
rank <- rank[order(rank$P.Value, as.integer(rownames(rank))),,drop=FALSE]
rank <- head(rank, k)
out <- data.frame(index0=as.integer(rownames(rank)), delta_M=rank$logFC,
                  moderated_t=rank$t, p_value=rank$P.Value, p_adj_BH=rank$adj.P.Val,
                  n_tested=length(eligible), n_train_patients=n,
                  limma_version=as.character(packageVersion("limma")))
stopifnot(all(is.finite(out$p_value)), all(is.finite(out$moderated_t)))
write.table(out, args[5], sep="\t", quote=FALSE, row.names=FALSE)
