#!/bin/bash
#SBATCH --account=muh10
#SBATCH --gpus=2
#SBATCH --job-name=testing
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kms8227@psu.edu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem-per-cpu=5gb
#SBATCH --time=00:10:00
#SBATCH --output=testing_%j.log

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

echo "Running test script..."
python testing.py --to_test=cuda
echo "DONE!"