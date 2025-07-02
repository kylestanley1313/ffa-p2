#!/bin/bash
#SBATCH --account=nfl5182_sc
#SBATCH --job-name=create-env
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kms8227@psu.edu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem-per-cpu=20gb
#SBATCH --time=02:00:00
#SBATCH --output=slurm/output/create_env_%j.out

source ~/.bashrc
# conda create -n ffa-p2 python=3.10
conda activate ffa-p2
# conda install -c conda-forge mpi4py petsc4py slepc4py pytorch pyyaml scipy scikit-learn pandas scikit-fda
# conda install -c conda-forge seaborn
# conda install -c conda-forge r-base r-essentials r-gparotation -y
conda install -c conda-forge r-argparse -y
pip install -e .
