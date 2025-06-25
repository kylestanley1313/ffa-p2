library(argparse)
library(GPArotation)


parser <- ArgumentParser(description = "Rotate loadings matrix")
parser$add_argument("--path_in", type = "character", required = TRUE)
parser$add_argument("--path_out", type = "character", required = TRUE)
parser$add_argument("--path_rot", type = "character", required = TRUE)
parser$add_argument("--path_trg", type = "character", required = FALSE)
parser$add_argument(
    "--rot_method", type = "character", required=TRUE,
    choices = c('varimax', 'quartimin', 'targor', 'targob')
)
args <- parser$parse_args()

# Parse arguments
path_in <- args$path_in
path_out <- args$path_out
path_rot <- args$path_rot
path_trg <- args$path_trg
rot_method <- args$rot_method

# Read in loadings
loads <- as.matrix(read.csv(path_in, header=FALSE))

# Rotate
if (rot_method == 'varimax' & is.null(path_trg)) {
    out <- Varimax(loads)
} else if (rot_method == 'quartimin' & is.null(path_trg)) {
    out <- quartimin(loads)
} else if (rot_method == 'varimax' & !is.null(path_trg)) {
    target <- as.matrix(read.csv(path_trg, header=FALSE))
    out <- targetQ(loads, Target=target)
} else if (rot_method == 'quartimin' & !is.null(path_trg)) {
    target <- as.matrix(read.csv(path_trg, header=FALSE))
    out <- targetT(loads, Target=target)
} else {
    cat("Invalid rot_method: ", rot_method)
}
loads_rot <- out$loadings
rot_mat <- t(out$Th) 

# Save matrices
write.table(
  loads_rot, 
  file = gzfile(path_out), 
  sep = ",", 
  row.names = FALSE, 
  col.names = FALSE
)
write.table(
  rot_mat, 
  file = gzfile(path_rot), 
  sep = ",", 
  row.names = FALSE, 
  col.names = FALSE
)





