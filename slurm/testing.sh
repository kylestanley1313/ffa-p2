#!/bin/bash
#SBATCH --account=muh10
#SBATCH --job-name=testing
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kms8227@psu.edu
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --mem-per-cpu=5gb
#SBATCH --time=00:10:00
#SBATCH --output=slurm/output/testing_%j.out

# Get started
echo " "
echo "Job started on $(hostname) at $(date)"
echo " "

# Load modules
module purge
module load anaconda3/2021.05

# cd into project root
cd ~/work/ffa-p2-priv

# Activate conda environment
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate ffa-p2-priv

echo "Simulating DDP factor model..."
python simulate_data.py --dir ffa-ddp --num_vars 30 --num_train 100 --num_val 0 --batch_size 50
echo "DONE!"

echo "Estimating DDP factor model..."
python factor_model_ddp.py --world_size 2 --dir ffa-ddp --num_facs 2 --alpha 0 --delta 0.1 --lr 0.05 --max_epochs 100
echo "DONE!"

# echo "Simulating Dist factor model..."
# python simulate_data.py --dir ffa-dist --num_vars 30 --num_train 100 --num_val 0 --batch_size 50
# echo "DONE!"

# echo "Estimating Dist factor model..."
# python factor_model_dist.py --world_size 2 --dir ffa-dist --num_facs 2 --alpha 0 --delta 0.1 --lr 0.05 --max_epochs 100
# echo "DONE!"