#!/bin/bash
#SBATCH --account=nfl5182_sc
#SBATCH --job-name=simulation
#SBATCH --mail-type=END,FAIL                      
#SBATCH --mail-user=kms8227@psu.edu            
#SBATCH -N 1                                      
#SBATCH -n 1                                  
#SBATCH --mem-per-cpu=1gb                         
#SBATCH --time=06:00:00                           
#SBATCH --output=slurm/output/simulation_%j.log

echo " "
echo "Started: $(date)"
echo " "

# Set variables
ROOT='/storage/home/kms8227/work/ffa-p2-priv'
SUPERDESIGN='sim-test-3'
METHODS_LE='dssgd'
METHODS_FSE='rbels' #'pls,rbels'
ROTATIONS='varimax,quartimin'
REGIMES_FSE='3' #'1,3'

cd $ROOT
source slurm/utils.sh

STEPS=(

    # 'setup-simulations'
    # 'simulate-data'
    # 'compute-covariance'

    # 'allocate-points'
    # 'initialize-loadings'
    # # #  'tune-alpha'
    # 'estimate-loadings'
    # 'rotate'
    # 'tune-sigmas'
    # 'smooth-loadings'
    # # # 'compute-inv-err-cov' # NOTE: GLS methods not appropriate for multiple subjects
    # 'tune-kappas'
    # 'shrink-loadings'
    
    'tune-gammas'
    'estimate-factor-scores'

    # 'melodic-data-prep'
    # 'melodic-tune-sigma'
    # 'melodic-estimation'
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
    CMD+=" --rotations=$ROTATIONS"
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
