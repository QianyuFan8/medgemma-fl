# Run with the same Rscript executable that will perform preprocessing.
options(repos=c(CRAN="https://cloud.r-project.org"))
for (p in c("data.table", "jsonlite", "glmnet", "BiocManager"))
  if (!requireNamespace(p, quietly=TRUE)) install.packages(p)
if (!requireNamespace("limma", quietly=TRUE)) BiocManager::install("limma", ask=FALSE, update=FALSE)
