#!/bin/bash
#SBATCH --account=open
#SBATCH --job-name=simulation-fse
#SBATCH --mail-type=END,FAIL                      
#SBATCH --mail-user=kms8227@psu.edu            
#SBATCH -N 1                                      
#SBATCH -n 1                                  
#SBATCH --mem-per-cpu=1gb                         
#SBATCH --time=00:60:00                           
#SBATCH --output=slurm/output/simulation-fse_%j.log

echo " "
echo "Started: $(date)"
echo " "

# Set variables
ROOT='/storage/home/kms8227/work/ffa-p2-priv'
SUPERDESIGN='bench1'
METHODS_LE='lbfgs,dsgd,dssgd'
METHODS_FSE=''
REGIMES_FSE=''

# TODO: Figure out how to get LE simulation working. Some difficulty with LAST_STEP...

cd $ROOT
source slurm/utils.sh

STEPS=(
    'setup-simulations'
    'simulate-data'
    'compute-covariance'
    # 'allocate-points'
    # 'initialize-loadings'
    # 'tune-alpha'
    # 'estimate-loadings'
    # 'compute-inv-err-cov'
    # 'tune-gamma'
    # 'estimate-factor-scores'
)

LAST_STEP=""
for step in ${STEPS[@]}; do

    echo "Queueing $step..."
    CMD="sbatch --parsable --wait"
    CMD+=" --account=$SLURM_JOB_ACCOUNT"
    CMD+=" --output=slurm/output/process-superdesign_${step}_%j.out"
    CMD+=" --job-name=process-superdesign_$step"

    # if [ ! -z "$JOB_IDS" ]; then
    #     CMD+=" --dependency=afterok:$JOB_IDS"
    # fi
    CMD+=" slurm/process-superdesign.sh" # $step $SUPERDESIGN $METHODS_LE $METHODS_FSE $REGIMES_FSE"
    CMD+=" --step=$step"
    CMD+=" --superdesign=$SUPERDESIGN"
    CMD+=" --methods-le=$METHODS_LE"
    CMD+=" --methods-fse=$METHODS_FSE"
    CMD+=" --regimes-fse=$REGIMES_FSE"
    CMD+=" --last-step=$LAST_STEP"

    # If their was a prior step, pass that step as a superdesign script argument
    # if [ ! -z "$LAST_STEP" ]; then
    #     CMD+=" $LAST_STEP"
    # fi
    
    echo "Running: $CMD"
    $CMD
    wait
    LAST_STEP=$step
    # JOB_IDS=$(<"slurm/tmp/${SUPERDESIGN}_${step}.txt")

done

echo " "
echo "Completed: $(date)"
echo " "
