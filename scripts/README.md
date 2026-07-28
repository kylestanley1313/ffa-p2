# Simulations

```
$ python scripts/setup_superdesign.py --config roar --superdesign sim-est --split_on load_scheme regime n_facs
$ python scripts/setup_superdesign.py --config roar --superdesign sim-exp --split_on n_facs path_fac_cov
$ python scripts/setup_superdesign.py --config roar --superdesign sim-fse --split_on path_mask delta
```

After replacing all angle-bracketed values (e.g., `<ACCOUNT>`) in `run-simulations.sh`, use the below commands to run simulations for subspace estimation, subspace expression, and factor score estimation, respectively. 

```
$ sbatch slurm/run-simulations.sh --superdesign==sim-est --type=est --rotations=varimax
$ sbatch slurm/run-simulations.sh --superdesign==sim-exp --type=exp
$ sbatch slurm/run-simulations.sh --superdesign==sim-fse --type=fse --rotations=varimax --regimes_fse=2
```

# AOMIC Analysis

