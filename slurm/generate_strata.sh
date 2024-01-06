#!/bin/bash
SBATCH --account=muh10
SBATCH --job-name=testing
SBATCH --mail-type=END,FAIL
SBATCH --mail-user=kms8227@psu.edu
SBATCH -N 1
SBATCH -n 20
SBATCH --mem-per-cpu=5gb
SBATCH --time=12:00:00
SBATCH --output=slurm/output/testing_%j.out

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


nproc_list=()
for ((i=10; i<=20; i++)); do
    nproc_list+=("$i")
done

for nproc in "${nproc_list[@]}"; do
    echo "Generating strata for $nproc processes..."
    python generate_strata.py --world_size 20 --num_procs "$nproc"
    echo "DONE!"
done

