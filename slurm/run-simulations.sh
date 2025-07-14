#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=simulation
#SBATCH --mail-type=END,FAIL                      
#SBATCH --mail-user=<EMAIL>            
#SBATCH -N 1                                      
#SBATCH -n 1                                  
#SBATCH --mem-per-cpu=1gb                         
#SBATCH --time=06:00:00                           
#SBATCH --output=slurm/output/simulation_%j.log

echo " "
echo "Started: $(date)"
echo " "

# Set variables
ROOT='<ROOT>'
SUPERDESIGN=''
TYPE=''
METHODS_FSE='pls,rbels' #'pls,rbels'
ROTATIONS='varimax,quartimin'
REGIMES_FSE='1,2'

# Parse command-line arguments
for arg in "$@"; do
    case $arg in
        --superdesign=*)
            SUPERDESIGN="${arg#*=}"
            shift
            ;;
        --rotations=*)
            ROTATIONS="${arg#*=}"
            shift
            ;;
        --methods_fse=*)
            METHODS_FSE="${arg#*=}"
            shift
            ;;
        --regimes_fse=*)
            REGIMES_FSE="${arg#*=}"
            shift
            ;;
        --type=*)
            TYPE="${arg#*=}"
            shift
            ;;
        *)
            echo "Unknown argument: $arg"
            ;;
    esac
done

# Set steps based on type
if [[ "$TYPE" == "est" ]]; then
    STEPS=(
        'setup-simulations'
        'simulate-data'
        'compute-covariance'
        'allocate-points'
        'initialize-loadings'
        'estimate-loadings'
        'rotate'
        'tune-sigmas'
        'smooth-loadings'
        'tune-kappas'
        'shrink-loadings'
        # 'tune-gammas'
        # 'estimate-factor-scores'
        'melodic-data-prep'
        'melodic-tune-sigma'
        'melodic-estimation'
    )
elif [[ "$TYPE" == "exp" ]]; then
    STEPS=(
        'setup-simulations'
        'simulate-data'
        'compute-covariance'
        'allocate-points'
        'initialize-loadings'
        'estimate-loadings'
        'rotate'
        'tune-sigmas'
        'smooth-loadings'
        'tune-kappas'
        'shrink-loadings'
        # 'tune-gammas'
        # 'estimate-factor-scores'
        'melodic-data-prep'
        'melodic-tune-sigma'
        'melodic-estimation'
    )
elif [[ "$TYPE" == "fse" ]]; then
    STEPS=(
        'setup-simulations'
        'simulate-data'
        'compute-covariance'
        'allocate-points'
        'initialize-loadings'
        'estimate-loadings'
        'rotate'
        'tune-sigmas'
        'smooth-loadings'
        'tune-kappas'
        'shrink-loadings'
        'tune-gammas'
        'estimate-factor-scores'
        # 'melodic-data-prep'
        # 'melodic-tune-sigma'
        # 'melodic-estimation'
    )
else
    echo "Invalid type: $TYPE"
    exit 1
fi


cd $ROOT
source slurm/utils.sh

STEPS=(

    'setup-simulations'
    'simulate-data'
    'compute-covariance'

    'allocate-points'
    'initialize-loadings'
    'estimate-loadings'
    'rotate'
    'tune-sigmas'
    'smooth-loadings'
    'tune-kappas'
    'shrink-loadings'
    
    # 'tune-gammas'
    # 'estimate-factor-scores'

    'melodic-data-prep'
    'melodic-tune-sigma'
    'melodic-estimation'
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
    CMD+=" --methods-le=dssgd"
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