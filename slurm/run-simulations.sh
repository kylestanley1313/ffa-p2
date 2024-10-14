#!/bin/bash
#SBATCH --account=open
#SBATCH --job-name=simulation-fse
#SBATCH --mail-type=END,FAIL                      
#SBATCH --mail-user=kms8227@psu.edu            
#SBATCH -N 1                                      
#SBATCH -n 1                                  
#SBATCH --mem-per-cpu=1gb                         
#SBATCH --time=03:00:00                           
#SBATCH --output=slurm/output/simulation_%j.log

echo " "
echo "Started: $(date)"
echo " "

# Set variables
ROOT='/storage/home/kms8227/work/ffa-p2-priv'
SUPERDESIGN='bench2f'
METHODS_LE='dssgd' # lbfgs,dsgd,dssgd'
METHODS_FSE='pls,pgls,rbels,rbegls' #,rbels,rbegls'
REGIMES_FSE='1,2,3'

cd $ROOT
source slurm/utils.sh

STEPS=(
    'setup-simulations'
    'simulate-data'
    'compute-covariance'
    'allocate-points'
    'initialize-loadings'
    # 'tune-alpha'
    'estimate-loadings'
    'compute-inv-err-cov'
    'tune-gamma'
    'estimate-factor-scores'
)

LAST_STEP=""
for step in ${STEPS[@]}; do

    echo "Queueing $step..."
    CMD="sbatch --parsable --wait"
    CMD+=" --account=$SLURM_JOB_ACCOUNT"
    CMD+=" --output=slurm/output/process-superdesign_${step}_%j.out"
    CMD+=" --job-name=process-superdesign_$step"

    CMD+=" slurm/process-superdesign.sh"
    CMD+=" --step=$step"
    CMD+=" --superdesign=$SUPERDESIGN"
    CMD+=" --methods-le=$METHODS_LE"
    CMD+=" --methods-fse=$METHODS_FSE"
    CMD+=" --regimes-fse=$REGIMES_FSE"
    CMD+=" --last-step=$LAST_STEP"
    
    echo "Running: $CMD"
    echo " "
    $CMD
    wait
    LAST_STEP=$step

done

echo " "
echo "Completed: $(date)"
echo " "
