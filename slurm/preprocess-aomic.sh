#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=preprocess-aomic
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<EMAIL>
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c <WORLD_SIZE>
#SBATCH --mem-per-cpu=<MEM_PER_CPU>
#SBATCH --time=<TIME>
#SBATCH --output=slurm/output/preprocess-aomic_%j.out


echo "========================================"
echo "Start: $(date)"
echo "========================================"


# Activate conda env
source ~/.bashrc
conda activate ffa-p2
cd /storage/home/kms8227/work/ffa-p2-priv


DIR_IN='<DIR_IN>'  # Directory of dataset ds002785
DIR_OUT='<DIR_OUT>'  # Output directory
DIR_FSL='<DIR_FSL>'  # FSL directory
N_SUBS='<N_SUBS>'  # Number of subjects to preprocess (first N_SUBS)
N_SUBS_TRAIN='<N_SUBS_TRAIN>'  # Number of subjects used to train FIX classifier (first N_SUBS_TRAIN)
MODEL_NAME='<MODEL_NAME>'  # Name of FIX classifier
THRESHOLD='<THRESHOLD>'  # Classification threshold
WORLD_SIZE='<WORLD_SIZE>'  # Number of CPUs 


# Parse arguments
for arg in "$@"
do
    case $arg in
        --step=*)
        step="${arg#*=}"
        shift
        ;;
        *)
        # unknown option
        ;;
    esac
done


if [ "$step" = "bet" ]; then
    python /storage/home/kms8227/work/ffa-p2-priv/scripts/preprocess_aomic_data.py \
        --dir_in $DIR_IN \
        --dir_out $DIR_OUT \
        --dir_fsl $DIR_FSL \
        --step bet \
        --n_subs $N_SUBS \
        --world_size $WORLD_SIZE
fi


if [ "$step" = "feat" ]; then
    python /storage/home/kms8227/work/ffa-p2-priv/scripts/preprocess_aomic_data.py \
        --dir_in $DIR_IN \
        --dir_out $DIR_OUT \
        --dir_fsl $DIR_FSL \
        --step feat \
        --n_subs $N_SUBS \
        --world_size $WORLD_SIZE
fi


if [ "$step" = "fix-extract" ]; then
    python /storage/home/kms8227/work/ffa-p2-priv/scripts/preprocess_aomic_data.py \
        --dir_in $DIR_IN \
        --dir_out $DIR_OUT \
        --dir_fsl $DIR_FSL \
        --step fix-extract \
        --n_subs $N_SUBS \
        --world_size $WORLD_SIZE
fi


# Now download .feat files and hand classify...


if [ "$step" = "fix-train" ]; then
    python /storage/home/kms8227/work/ffa-p2-priv/scripts/preprocess_aomic_data.py \
        --dir_in $DIR_IN \
        --dir_out $DIR_OUT \
        --dir_fsl $DIR_FSL \
        --step fix-train \
        --n_subs $N_SUBS_TRAIN \
        --fix_model $MODEL_NAME \
        --world_size 1
fi


if [ "$step" = "fix-denoise" ]; then
    python /storage/home/kms8227/work/ffa-p2-priv/scripts/preprocess_aomic_data.py \
        --dir_in $DIR_IN \
        --dir_out $DIR_OUT \
        --dir_fsl $DIR_FSL \
        --step fix-denoise \
        --n_subs $N_SUBS \
        --fix_model $MODEL_NAME \
        --threshold $THRESHOLD \
        --world_size $WORLD_SIZE
fi


echo "========================================"
echo "End: $(date)"
echo "========================================"


