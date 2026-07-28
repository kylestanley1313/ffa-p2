#!/bin/bash
#SBATCH --account=nfl5182_sc
#SBATCH --job-name=testing
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kms8227@psu.edu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem-per-cpu=20gb
#SBATCH --time=2:00:00
#SBATCH --output=slurm/output/testing_%j.out



echo "========================================"
echo "Start: $(date)"
echo "========================================"


# module load python
# echo '' | python ~/Downloads/fslinstaller.py -d ~/work/fsl

# Activate conda env
source ~/.bashrc
conda activate ffa-p2
cd /storage/home/kms8227/work/ffa-p2-priv

# python create_plots_analysis.py
# python scratch_defense.py

# python scripts/create_scree_plot.py \
#     --config roar \
#     --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/clean-1_1/sim-0/rep-0 \
#     --dir_out_scratch /storage/home/kms8227/scratch/ffa-p2-priv/out/clean-1_1/sim-0/rep-0 \
#     --n_folds 3 \
#     --min_n_facs 1 \
#     --max_n_facs 3 \
#     --world_size 2 

# /storage/home/kms8227/work/.conda/envs/ffa-p2/bin/python \
#     /storage/work/kms8227/ffa-p2-priv/tune_sigmas.py \
#     --config roar \
#     --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/sim-test-3_regime-2_loads-BL_K-2/sim-0/rep-0 \
#     --dir_out_scratch /storage/home/kms8227/scratch/ffa-p2-priv/out/sim-test-3_regime-2_loads-BL_K-2/sim-0/rep-0 \
#     --sigma_grid 0 0.5 \
#     --sz_space 30 30 \
#     --est_method dssgd \
#     --n_folds 3 \
#     --rot_method varimax


# Rscript rotate.R --path_in=/storage/group/kms8227/default/ffa-p2-priv/out/sim-test-3_regime-2_loads-BL_K-2/sim-0/rep-0/tmp-rot/loads-dssgd-train-0.csv.gz --path_out=/storage/group/kms8227/default/ffa-p2-priv/out/sim-test-3_regime-2_loads-BL_K-2/sim-0/rep-0/tmp-rot/loads-dssgd-train-0-targor.csv.gz --path_rot=/storage/group/kms8227/default/ffa-p2-priv/out/sim-test-3_regime-2_loads-BL_K-2/sim-0/rep-0/tmp-rot/rot-targor-train-0.csv.gz --rot_method=targor --path_trg=/storage/group/kms8227/default/ffa-p2-priv/out/sim-test-3_regime-2_loads-BL_K-2/sim-0/rep-0/tmp-rot/model-dssgd-full-varimax.csv.gz



##########################################################
################### MISCELLANEOUS ########################
##########################################################

# python evaluate_model.py \
#     --config roar \
#     --dir_designs /storage/group/kms8227/default/ffa-p2-priv/designs \
#     --designs sim-est-2-1_regime-1_loads-BL_K-2 sim-est-2-1_regime-1_loads-BL_K-4 sim-est-2-1_regime-1_loads-NL_K-2 sim-est-2-1_regime-1_loads-NL_K-4 sim-est-2-1_regime-2_loads-BL_K-2 sim-est-2-1_regime-2_loads-BL_K-4 sim-est-2-1_regime-2_loads-NL_K-2 sim-est-2-1_regime-2_loads-NL_K-4 \
#     --sim_type load


##########################################################
##################### SIMULATION #########################
##########################################################

# ANALYSIS='aomic-test-2'

# python /storage/home/kms8227/work/ffa-p2-priv/simulate_data.py \
#     --config roar \
#     --dir_dataset /storage/group/kms8227/default/ffa-p2-priv/datasets/${ANALYSIS} \
#     --dir_dataset_truth /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --n_sub 20 \
#     --n_time 100 \
#     --sz_space 20 20 \
#     --load_scheme BumpScheme2D \
#     --err_scheme GaussProc_BumpTensor2D \
#     --n_facs 2 \
#     --delta 0.1 \
#     --prop_global 0.2 \
#     --kernel_length_fac 5 \
#     --kernel_length_err 1 \
#     --kernel_length_err_var 10 \
#     --batch_size 500 \
#     --refresh_dataset_dir \
#     --seed 12345

# python /storage/home/kms8227/work/ffa-p2-priv/compute_covariance.py \
#     --config roar \
#     --dir_dataset /storage/group/kms8227/default/ffa-p2-priv/datasets/${ANALYSIS} \
#     --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --dir_out_scratch /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --world_size 4 \
#     --delta 0.1 \
#     --n_folds_sub 3 \
#     --n_folds_space_by_dim 2 2 \
#     --n_sub 20 \
#     --bsz_time 500 \
#     --bsz_space 1000 \
#     --fse \
#     --refresh_dirs \
#     --seed 12345
#     # --splits full train valid \


# python /storage/home/kms8227/work/ffa-p2-priv/allocate_points.py \
#     --config roar \
#     --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --dir_out_scratch /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --est_method dssgd \
#     --world_size 4 \
#     --seed 12345


# SPLITS=('train')
# N_FOLDS_SUB=3
# N_FOLDS_SPACE=4
# ROT='varimax'

# python /storage/home/kms8227/work/ffa-p2-priv/create_scree_plot.py \
#     --config roar \
#     --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --dir_out_scratch /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --n_folds $N_FOLDS_SUB \
#     --min_n_facs 1 \
#     --max_n_facs 3 \
#     --prop_init 0.5 \
#     --world_size 4 \
#     --seed 12345



# python /storage/home/kms8227/work/ffa-p2-priv/initialize_loadings.py \
#     --config roar \
#     --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --dir_out_scratch /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --split full \
#     --n_folds $N_FOLDS_SUB \
#     --n_facs 2 \
#     --init_method pca_randomized \
#     --prop_init 1.0 \
#     --seed 12345
# for split in "${SPLITS[@]}"; do
#     for fold in $(seq 0 $((N_FOLDS_SUB - 1))); do
        
#         python /storage/home/kms8227/work/ffa-p2-priv/initialize_loadings.py \
#             --config roar \
#             --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#             --dir_out_scratch /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#             --split $split \
#             --fold $fold \
#             --n_folds $N_FOLDS_SUB \
#             --n_facs 2 \
#             --init_method pca_randomized \
#             --prop_init 1.0 \
#             --seed 12345

#     done
# done


# python /storage/home/kms8227/work/ffa-p2-priv/estimate_loads_dssgd.py \
#     --config roar \
#     --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --dir_out_scratch /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --world_size 4 \
#     --split full \
#     --n_vars 400 \
#     --n_facs 2 \
#     --batch_size 128 \
#     --lr 0.3 \
#     --tol 1e-7 \
#     --patience 5 \
#     --max_epochs 500 \
#     --benchmark \
#     --seed 12345
# for split in "${SPLITS[@]}"; do
#     for fold in $(seq 0 $((N_FOLDS_SUB - 1))); do
        
#         python /storage/home/kms8227/work/ffa-p2-priv/estimate_loads_dssgd.py \
#             --config roar \
#             --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#             --dir_out_scratch /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#             --world_size 4 \
#             --split $split \
#             --fold $fold \
#             --n_vars 400 \
#             --n_facs 2 \
#             --batch_size 128 \
#             --lr 0.3 \
#             --tol 1e-7 \
#             --patience 5 \
#             --max_epochs 500 \
#             --benchmark \
#             --seed 12345

#     done
# done


# python /storage/home/kms8227/work/ffa-p2-priv/tune_sigmas.py \
#     --config roar \
#     --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --dir_out_scratch /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --est_method dssgd \
#     --n_folds $N_FOLDS_SUB \
#     --sigma_grid 0 0.3 0.35 0.40


# python /storage/home/kms8227/work/ffa-p2-priv/smooth_loads.py \
#     --config roar \
#     --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --sigmas 0.5 0.5 \
#     --split full \
#     --est_method dssgd 
# for split in "${SPLITS[@]}"; do
#     for fold in $(seq 0 $((N_FOLDS_SUB - 1))); do
        
#         python /storage/home/kms8227/work/ffa-p2-priv/smooth_loads.py \
#             --config roar \
#             --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#             --sigmas 0.5 0.5 \
#             --split $split \
#             --fold $fold \
#             --est_method dssgd 

#     done
# done


# python /storage/home/kms8227/work/ffa-p2-priv/rotate.py \
#     --config roar \
#     --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --rot_method $ROT \
#     --est_method dssgd \
#     --split full 
# for split in "${SPLITS[@]}"; do
#     for fold in $(seq 0 $((N_FOLDS_SUB - 1))); do
        
#         python /storage/home/kms8227/work/ffa-p2-priv/rotate.py \
#             --config roar \
#             --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#             --rot_method $ROT \
#             --est_method dssgd \
#             --split $split \
#             --fold $fold

#     done
# done


# python /storage/home/kms8227/work/ffa-p2-priv/tune_kappas.py \
#     --config roar \
#     --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --dir_out_scratch /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --est_method dssgd \
#     --rot_method $ROT \
#     --n_folds $N_FOLDS_SUB \
#     --max_iters 10 \
#     --tol 1e-6 \
#     --radius 3

# python /storage/home/kms8227/work/ffa-p2-priv/shrink_loads.py \
#     --config roar \
#     --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --kappas 0.0001250000059371814 0.0002159999858122319 \
#     --split full \
#     --est_method dssgd \
#     --rot_method $ROT



# python /storage/home/kms8227/work/ffa-p2-priv/tune_gammas.py \
#     --config roar \
#     --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --dir_out_scratch /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --sub_num 0 \
#     --gamma_grid 1e-10 1e-9 1e-8 \
#     --n_folds $N_FOLDS_SPACE \
#     --n_time 100 \
#     --est_method_loads dssgd \
#     --rot_method $ROT \
#     --regime 3 \
#     --batch_size 1000
# for n in $(seq 0 9); do
#     echo "---------- subject $n ----------"
#     python /storage/home/kms8227/work/ffa-p2-priv/tune_gamma.py \
#         --config roar \
#         --dir_out /storage/home/kms8227/work/ffa-p2-priv/out/${ANALYSIS} \
#         --dir_out_scratch /storage/home/kms8227/scratch/ffa-p2-priv/out/${ANALYSIS} \
#         --est_method rbels \
#         --sub_num $n \
#         --n_time 500 \
#         --est_method_loads dssgd \
#         --regime 3 \
#         --batch_size 1000
# done


# for n in $(seq 0 9); do
#     echo "---------- subject $n ----------"
#     python /storage/home/kms8227/work/ffa-p2-priv/estimate_factor_scores.py \
#         --config roar \
#         --dir_out /storage/home/kms8227/work/ffa-p2-priv/out/${ANALYSIS} \
#         --dir_out_scratch /storage/home/kms8227/scratch/ffa-p2-priv/out/${ANALYSIS} \
#         --est_methods rbels \
#         --sub_num $n \
#         --split full \
#         --est_method_loads dssgd \
#         --regime 3 \
#         --batch_size 100
# done







##########################################################
##################### AOMIC ##############################
##########################################################

# ANALYSIS='aomic-prod-2'
# N_FOLDS_SUB=3
# N_FOLDS_SPACE_BY_DIM=(2 2 1)
# N_FOLDS_SPACE=1
# for num in "${N_FOLDS_SPACE_BY_DIM[@]}"; do
#     (( N_FOLDS_SPACE *= num ))
# done
# SPLITS=('train')
# N_FACS=7
# ROTS=('varimax' 'quartimin')

# echo "========== $ANALYSIS =========="

# python /storage/home/kms8227/work/ffa-p2-priv/sandbox_2.py

# BENCHMARKING: 
#   - BET: fast
#   - FEAT: ~25 min/sub
#   - Feature extraction: ~10 min/sub
#   - Training: ?
#   - Denoising: ~5 min/sub

python /storage/home/kms8227/work/ffa-p2-priv/preprocess_aomic_data.py \
    --config roar \
    --dir_in /storage/group/kms8227/default/datasets/ds002785 \
    --dir_out /storage/group/kms8227/default/datasets/ds002785-fix-1 \
    --dir_fsl /storage/home/kms8227/work/fsl \
    --step bet \
    --n_subs 216 \
    --fix_model model-n10 \
    --threshold 20 \
    --world_size 10


# FIX training (include -l for LOOCV)
# CMD="fix -t /storage/home/kms8227/scratch/datasets/ds002785-fix/model-n10"
# CMD+=" /storage/home/kms8227/scratch/datasets/ds002785-fix/sub-0001.feat"
# CMD+=" /storage/home/kms8227/scratch/datasets/ds002785-fix/sub-0002.feat"
# CMD+=" /storage/home/kms8227/scratch/datasets/ds002785-fix/sub-0003.feat"
# CMD+=" /storage/home/kms8227/scratch/datasets/ds002785-fix/sub-0004.feat"
# CMD+=" /storage/home/kms8227/scratch/datasets/ds002785-fix/sub-0005.feat"
# CMD+=" /storage/home/kms8227/scratch/datasets/ds002785-fix/sub-0006.feat"
# CMD+=" /storage/home/kms8227/scratch/datasets/ds002785-fix/sub-0007.feat"
# CMD+=" /storage/home/kms8227/scratch/datasets/ds002785-fix/sub-0008.feat"
# CMD+=" /storage/home/kms8227/scratch/datasets/ds002785-fix/sub-0009.feat"
# CMD+=" /storage/home/kms8227/scratch/datasets/ds002785-fix/sub-0010.feat"
# $CMD


# python /storage/home/kms8227/work/ffa-p2-priv/prepare_aomic_data.py \
#     --config roar \
#     --analysis ${ANALYSIS} \
#     --n_subs 218 \
#     --path_mask /storage/home/kms8227/work/ffa-p2-priv/masks/mask_aomic-allsubs.pt \
#     --world_size 1


# python /storage/home/kms8227/work/ffa-p2-priv/compute_covariance.py \
#     --config roar \
#     --dir_dataset /storage/group/kms8227/default/ffa-p2-priv/datasets/${ANALYSIS} \
#     --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --dir_out_scratch /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --world_size 15 \
#     --delta 0.1 \
#     --n_folds_sub $N_FOLDS_SUB \
#     --n_folds_space_by_dim "${N_FOLDS_SPACE_BY_DIM[@]}" \
#     --n_sub 210 \
#     --bsz_time 500 \
#     --bsz_space 1000 \
#     --path_mask /storage/home/kms8227/work/ffa-p2-priv/masks/mask_aomic-allsubs.pt \
#     --fse \
#     --refresh_dirs \
#     --seed 12345
#     # --splits full train valid \

# python /storage/home/kms8227/work/ffa-p2-priv/allocate_points.py \
#     --config roar \
#     --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --dir_out_scratch /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --est_method dssgd \
#     --world_size 16 \
#     --seed 12345

# python /storage/home/kms8227/work/ffa-p2-priv/create_scree_plot.py \
#     --config roar \
#     --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --dir_out_scratch /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --min_n_facs 1 \
#     --max_n_facs 20 \
#     --prop_init 0.5 \
#     --world_size 16 \
#     --seed 12345


# python /storage/home/kms8227/work/ffa-p2-priv/initialize_loadings.py \
#     --config roar \
#     --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --dir_out_scratch /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --split full \
#     --n_folds $N_FOLDS_SUB \
#     --n_facs $N_FACS \
#     --init_method pca_randomized \
#     --prop_init 0.1 \
#     --seed 12345
# for split in "${SPLITS[@]}"; do
#     for fold in $(seq 0 $((N_FOLDS_SUB - 1))); do
        
#         python /storage/home/kms8227/work/ffa-p2-priv/initialize_loadings.py \
#             --config roar \
#             --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#             --dir_out_scratch /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#             --split $split \
#             --fold $fold \
#             --n_folds $N_FOLDS_SUB \
#             --n_facs $N_FACS \
#             --init_method pca_randomized \
#             --prop_init 0.2 \
#             --seed 12345

#     done
# done


# NOTE: Original learning rate was 16.0
# python /storage/home/kms8227/work/ffa-p2-priv/estimate_loads_dssgd.py \
#     --config roar \
#     --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --dir_out_scratch /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --world_size 16 \
#     --split full \
#     --n_vars 71587 \
#     --n_facs $N_FACS \
#     --batch_size 4096 \
#     --lr 4.0 \
#     --tol 1e-9 \
#     --patience 5 \
#     --max_epochs 1000 \
#     --benchmark \
#     --seed 12345
# for split in "${SPLITS[@]}"; do
#     for fold in $(seq 0 $((N_FOLDS_SUB - 1))); do
        
#         python /storage/home/kms8227/work/ffa-p2-priv/estimate_loads_dssgd.py \
#             --config roar \
#             --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#             --dir_out_scratch /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#             --world_size 16 \
#             --split $split \
#             --fold $fold \
#             --n_vars 71587 \
#             --n_facs $N_FACS \
#             --batch_size 4096 \
#             --lr 16.0 \
#             --tol 1e-7 \
#             --patience 5 \
#             --max_epochs 1000 \
#             --benchmark \
#             --seed 12345

#     done
# done

# python /storage/home/kms8227/work/ffa-p2-priv/tune_sigmas.py \
#     --config roar \
#     --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --dir_out_scratch /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --est_method dssgd \
#     --path_mask /storage/home/kms8227/work/ffa-p2-priv/masks/mask_aomic-allsubs.pt \
#     --n_folds $N_FOLDS_SUB \
#     --sigma_grid 0 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0


# python /storage/home/kms8227/work/ffa-p2-priv/smooth_loads.py \
#     --config roar \
#     --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --sigmas 0.35 0.35 0.35 0.35 0.35 0.35 0.4 0.4 0.45 \
#     --split full \
#     --est_method dssgd \
#     --path_mask /storage/home/kms8227/work/ffa-p2-priv/masks/mask_aomic-allsubs.pt
# for split in "${SPLITS[@]}"; do
#     for fold in $(seq 0 $((N_FOLDS_SUB - 1))); do
#         python /storage/home/kms8227/work/ffa-p2-priv/smooth_loads.py \
#             --config roar \
#             --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#             --sigmas 0.35 0.35 0.35 0.35 0.35 0.35 0.4 0.4 0.45 \
#             --split $split \
#             --fold $fold \
#             --est_method dssgd \
#             --path_mask /storage/home/kms8227/work/ffa-p2-priv/masks/mask_aomic-allsubs.pt
#     done
# done



# for rot in "${ROTS[@]}"; do
#     python /storage/home/kms8227/work/ffa-p2-priv/rotate.py \
#         --config roar \
#         --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#         --rot_method $rot \
#         --est_method dssgd \
#         --split full 
#     for split in "${SPLITS[@]}"; do
#         for fold in $(seq 0 $((N_FOLDS_SUB - 1))); do
            
#             python /storage/home/kms8227/work/ffa-p2-priv/rotate.py \
#                 --config roar \
#                 --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#                 --rot_method $rot \
#                 --est_method dssgd \
#                 --split $split \
#                 --fold $fold

#         done
#     done
# done


# ANALYSIS='aomic-prod-2'
# ROT='quartimin'
# N_FOLDS_SUB=3
# echo "========== $ANALYSIS | $ROT =========="
# # python /storage/home/kms8227/work/ffa-p2-priv/tune_kappas.py \
# #     --config roar \
# #     --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
# #     --dir_out_scratch /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
# #     --est_method dssgd \
# #     --rot_method $ROT \
# #     --n_folds $N_FOLDS_SUB \
# #     --method sequential
# python /storage/home/kms8227/work/ffa-p2-priv/shrink_loads.py \
#     --config roar \
#     --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --path_kappas /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS}/kappas-dssgd-${ROT}.pt \
#     --split full \
#     --est_method dssgd \
#     --rot_method $ROT




# python /storage/home/kms8227/work/ffa-p2-priv/shrink_loads.py \
#     --config roar \
#     --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --kappas 0.0 9.999999974752427e-07 0.0 0.0 0.0 7.999999979801942e-06 0.0 \
#     --split full \
#     --est_method dssgd \
#     --rot_method varimax



# NOTE: Gamma tuning and FSE in testing_array.sh

# python /storage/home/kms8227/work/ffa-p2-priv/tune_gammas.py \
#     --config roar \
#     --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --dir_out_scratch /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#     --sub_num 0 \
#     --gamma_grid 1e-10 1e-9 1e-8 \
#     --n_folds $N_FOLDS_SPACE \
#     --n_time 100 \
#     --est_method_loads dssgd \
#     --rot_method varimax \
#     --regime 3 \
#     --path_mask /storage/home/kms8227/work/ffa-p2-priv/masks/mask_aomic-allsubs.pt \
#     --batch_size 1000



# MAX_PARALLEL=5  # Maximum concurrent jobs
# JOB_COUNT=0
# for n in $(seq 0 3); do
#     echo "---------- subject $n ----------"
#     python /storage/home/kms8227/work/ffa-p2-priv/tune_gamma.py \
#         --config roar \
#         --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#         --dir_out_scratch /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#         --est_method rbels \
#         --sub_num $n \
#         --n_time 470 \
#         --est_method_loads dssgd \
#         --rot_method varimax \
#         --regime 3 \
#         --path_mask /storage/home/kms8227/work/ffa-p2-priv/masks/mask_aomic-allsubs.pt \
#         --batch_size 1000 &  # Run in background
    
#     ((JOB_COUNT++))
#     if [[ $JOB_COUNT -ge $MAX_PARALLEL ]]; then
#         wait  # Wait for all background jobs to finish before starting more
#         JOB_COUNT=0
#     fi
# done
# wait  # Final wait to ensure all jobs finish


# seq 0 4 | xargs -I {} -P 5 bash -c '
#     echo "---------- subject {} ----------"
#     python /storage/home/kms8227/work/ffa-p2-priv/tune_gamma.py \
#         --config roar \
#         --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/'${ANALYSIS}' \
#         --dir_out_scratch /storage/group/kms8227/default/ffa-p2-priv/out/'${ANALYSIS}' \
#         --est_method rbels \
#         --sub_num {} \
#         --n_time 470 \
#         --est_method_loads dssgd \
#         --rot_method varimax \
#         --regime 3 \
#         --path_mask /storage/home/kms8227/work/ffa-p2-priv/masks/mask_aomic-allsubs.pt \
#         --batch_size 1000
# '
# for n in $(seq 0 4); do
#     echo "---------- subject $n ----------"
#     python /storage/home/kms8227/work/ffa-p2-priv/tune_gamma.py \
#         --config roar \
#         --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#         --dir_out_scratch /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#         --est_method rbels \
#         --sub_num $n \
#         --n_time 470 \
#         --est_method_loads dssgd \
#         --rot_method varimax \
#         --regime 3 \
#         --path_mask /storage/home/kms8227/work/ffa-p2-priv/masks/mask_aomic-allsubs.pt \
#         --batch_size 1000
# done
# wait


# TODO: Uncomment initial loadings code in fse script!
# for i in $(seq 0 0); do
#     echo "Processing subject ${i}..."
#     python /storage/home/kms8227/work/ffa-p2-priv/estimate_factor_scores.py \
#         --config roar \
#         --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#         --dir_out_scratch /storage/group/kms8227/default/ffa-p2-priv/out/${ANALYSIS} \
#         --sub_num $i \
#         --gamma 0 \
#         --est_methods pls \
#         --split full \
#         --est_method_loads init \
#         --rot_method varimax \
#         --regime 3 \
#         --path_mask /storage/home/kms8227/work/ffa-p2-priv/masks/mask_aomic-allsubs.pt \
#         --batch_size 100
# done



##########################################################
################# AOMIC (ICA) ############################
##########################################################

# ANALYSIS='aomic-prod-2'
# N_SUB=20
# N_FOLDS_SUB=3
# SIGMA=1
# N_COMPS=7

# N_FOLDS_SPACE_BY_DIM=(2 2 1)
# N_FOLDS_SPACE=1
# for num in "${N_FOLDS_SPACE_BY_DIM[@]}"; do
#     (( N_FOLDS_SPACE *= num ))
# done
# SPLITS=('train')
# N_FACS=9
# ROTS=('varimax' 'quartimin')



# echo "---------- MELODIC DATA PREP ----------"
# python melodic_data_prep_2.py \
#     --config roar \
#     --dir_dataset /storage/group/kms8227/default/ffa-p2-priv/datasets/$ANALYSIS \
#     --dir_out_scratch /storage/group/kms8227/default/ffa-p2-priv/out/$ANALYSIS \
#     --n_sub $N_SUB 


# echo "---------- MELODIC ESTIMATION ----------"
# python melodic_estimation.py \
#     --config roar \
#     --dir_out /storage/group/kms8227/default/ffa-p2-priv/out/$ANALYSIS \
#     --dir_out_scratch /storage/group/kms8227/default/ffa-p2-priv/out/$ANALYSIS \
#     --n_folds $N_FOLDS_SUB \
#     --n_comps $N_COMPS \
#     --sigma $SIGMA 



echo "========================================"
echo "End: $(date)"
echo "========================================"
