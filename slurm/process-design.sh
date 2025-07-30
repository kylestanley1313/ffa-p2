#!/bin/bash
# SBATCH --account=open
# SBATCH --job-name=process-designs
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kms8227@psu.edu
#SBATCH -N 1
#SBATCH -n 1

# Arguments: 
#   1. step
#   2. design
#   3. le methods (comma-separated)
#   4. fse methods (comma-separated)
#   5. rotations (comma-separated)
#   6. fse regimes (comma-separated)
# Ex: sbatch --job-name=test-1-1_setup-simulations --output=slurm/output/17071602_test-1-1_setup-simulations.out -n 1 --mem-per-cpu=1gb --time=00:05:00 slurm/s1_setup-simulations.sh test-1-1 lbfgs pls,pgls varimax,quartimin 1,2

# Parse command line arguments
step=""
design=""
methods_le=""
methods_fse=""
rotations=""
regimes_fse=""
last_step=""
for arg in "$@"; do
  case $arg in
    --step=*)
      step="${arg#*=}"
      shift
      ;;
    --design=*)
      design="${arg#*=}"
      shift
      ;;
    --methods-le=*)
      methods_le="${arg#*=}"
      shift
      ;;
    --methods-fse=*)
      methods_fse="${arg#*=}"
      shift
      ;;
    --rotations=*)
      rotations="${arg#*=}"
      shift
      ;;
    --regimes-fse=*)
      regimes_fse="${arg#*=}"
      shift
      ;;
    *)
      echo "Invalid argument: $arg"
      ;;
  esac
done

echo " "
echo "Started: $(date)"
echo " "

# Activate conda env
source ~/.bashrc
conda activate ffa-p2

# Source utils file
cd ~/work/ffa-p2-priv
source slurm/utils.sh

# Run command
CMD=$(build_run_simulations_cmd $step $design $methods_le $methods_fse $rotations $regimes_fse)
echo "Running: $CMD"
$CMD > logs/${design}_${step}

echo " "
echo "Completed: $(date)"
echo " "
