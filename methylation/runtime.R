# Check previously recorded R library ABI before loading compiled packages.
for (lib in .libPaths()) {
  marker <- file.path(lib, "runtime.dcf")
  if (file.exists(marker)) {
    recorded <- read.dcf(marker)
    if (!identical(unname(recorded[1,"RHome"]), normalizePath(R.home())) ||
        !identical(unname(recorded[1,"Platform"]), R.version$platform))
      stop("R library belongs to a different R runtime: ", lib, ". Select matching Rscript or install a separate library.")
  }
}
