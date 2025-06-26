#!/bin/bash
#SBATCH --account=open
#SBATCH --job-name=preprocess-aomic
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kms8227@psu.edu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem-per-cpu=30gb
#SBATCH --time=4:00:00
#SBATCH --output=slurm/output/preprocess-aomic_%j.out

SUBS=(
    '0001'
    '0002'
    '0003'
    '0004'
    '0005'
    '0006'
    '0007'
    '0008'
    '0009'
    '0010'
)
SUBS_IDX=(0 1 2 3 4 5 6 7 8 9)


# GIVEN: 
#   [at (1)]
#       - Anatomical images for each subject
#       - Functional image for each subject
#   [at ()]
#       - 
# STEPS: 
#   (2) BET for each subject
#   (3) Create subject-specific design files
#   (4) FEAT for each subject
#   (5) Hand classification for each subject
#   (6) FIX
#       (a) Feature extraction
#       (b) Training (optional LOOCV)
#       (c) Classification  
 

# () BET for each subject
cd ~
for sub in ${SUBS[@]}; do
    CMD="bet /storage/home/kms8227/scratch/datasets/ds002785/derivatives/fmriprep/sub-${sub}/anat/sub-${sub}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
    CMD+=" /storage/home/kms8227/scratch/datasets/ds002785/derivatives/fmriprep/sub-${sub}/anat/sub-${sub}_space-MNI152NLin2009cAsym_desc-preproc_T1w_brain.nii.gz"
    $CMD
done

# (3) Generate subject-specific desgin files
cd ~
design_dir=/Users/kylestanley/repos/ffa-p2-priv/scratch/fix-ica/designs
for sub in ${SUBS[@]}; do
    cp ${design_dir}/design-template.fsf ${design_dir}/sub-${sub}.fsf
    sed -i "" "s/SUB/${sub}/g" "${design_dir}/sub-${sub}.fsf"
done


# (4) Run FEAT for each subject (15-20 min per subject)
# cd ~
# design_dir=/Users/kylestanley/repos/ffa-p2-priv/scratch/fix-ica/designs
# for sub in ${SUBS[@]}; do
#     echo "Running FEAT (with MELODIC) for sub-${sub}"
#     feat ${design_dir}/sub-${sub}.fsf &
# done
# wait
# echo "All FEAT processes completed!"

# (5) Hand classification: fsleyes --scene melodic -ad filtered_func_data.ica

# (6a) FIX -- Feature Extraction (~ 1 hr)
# cd /Users/kylestanley/datasets/ds002785-fix
# for sub in ${SUBS[@]}; do
#     echo "FIX feature extraction for sub-${sub}"
#     fix -f "sub-${sub}.feat" &
# done
# wait
# echo "All FIX processes completed!"

# (6b) FIX -- Training
# cd /Users/kylestanley/datasets/ds002785-fix
# CMD="fix -t model-test-1"
# for sub in ${SUBS[@]}; do
#     CMD+=" sub-${sub}.feat"
# done
# echo "Training FIX classifier..."
# $CMD
# wait
# echo "DONE!"

# (6c) FIX -- Denoising
# echo "FIX denoising scans..."
# cd /Users/kylestanley/datasets/ds002785-fix
# for sub in ${SUBS[@]}; do
#     fix sub-${sub}.feat model-test-1.pyfix_model 20
# done
# echo "DONE!"





