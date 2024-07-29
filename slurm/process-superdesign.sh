#!/bin/bash
# SBATCH --account=open
# SBATCH --job-name=process-superdesign
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kms8227@psu.edu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem-per-cpu=1gb
#SBATCH --time=00:05:00
# SBATCH --output=slurm/output/process-superdesign_%j.out

# Arguments: 
#   1. step
#   2. superdesign
#   3. le methods (comma-separated)
#   4. fse methods (comma-separated)
#   5. fse regimes (comma-separated)
#   6. last step (optional)
# Ex: sbatch slurm/s_process-designs.sh open kms8227@psu.edu simulate-data test-1 lbfgs pls,pgls 1,2,3 setup-simulations

echo " "
echo "Started: $(date)"
echo " "

# Source utils file
cd ~/work/ffa-p2-priv
source slurm/step-maps.sh

# Parse command line arguments
step=""
superdesign=""
methods_le=""
methods_fse=""
regimes_fse=""
last_step=""
for arg in "$@"; do
  case $arg in
    --step=*)
      step="${arg#*=}"
      shift
      ;;
    --superdesign=*)
      superdesign="${arg#*=}"
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
    --regimes-fse=*)
      regimes_fse="${arg#*=}"
      shift
      ;;
    --last-step=*)
      last_step="${arg#*=}"
      shift
      ;;
    *)
      echo "Invalid argument: $arg"
      ;;
  esac
done

# Start design jobs
JOB_IDS=""
while DESIGNS= read -r design; do

    # Set slurm parameters
    CMD="sbatch --parsable"
    CMD+=" --account=$SLURM_JOB_ACCOUNT"
    CMD+=" --job-name=process-design_${design}_${step}"
    CMD+=" --output=slurm/output/process-design_${design}_${step}_${SLURM_JOB_ID}.out"
    CMD+=" -n ${step_cpus[$step]}"
    CMD+=" --mem-per-cpu=${step_mem_per_cpu[$step]}gb"
    CMD+=" --time=${step_time[$step]}"

    # If their was a prior step, add it as a dependency
    if [ ! -z "$last_step" ]; then
        LAST_JOB_IDS=$(<"slurm/tmp/${superdesign}_${last_step}.txt")
        CMD+=" --dependency=afterok:$LAST_JOB_IDS"
    fi

    # Set design script parameters
    CMD+=" slurm/process-design.sh"
    CMD+=" --step=$step"
    CMD+=" --design=$design"
    CMD+=" --methods-le=$methods_le"
    CMD+=" --methods-fse=$methods_fse"
    CMD+=" --regimes-fse=$regimes_fse"
    # CMD+=" $1 $design $3 $4 $5"

    # Run design script
    echo "Running: $CMD"
    JOB_ID=$($CMD)

    # Add Job ID to Job IDs
    if [ -z "$JOB_IDS" ]; then
        JOB_IDS="$JOB_ID"
    else
        JOB_IDS="${JOB_IDS}:${JOB_ID}"
    fi

done < "slurm/superdesigns/$superdesign.txt"

# Write JOB_IDS to temp file so that run-simulation script knows which jobs
# it needs to wait for.
echo $JOB_IDS > "slurm/tmp/${superdesign}_${step}.txt"

echo " "
echo "Completed: $(date)"
echo " "
