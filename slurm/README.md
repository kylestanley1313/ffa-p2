## `create-env.sh`

## `preprocess-aomic.sh`

**Overview**

Before analyzing the AOMIC data, we used [FMRIB's ICA-based X-noiseifier (FIX)](https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/FIX.html) to remove noise components (e.g., head movement, respiratory motion, and scanner artifacts) from each subject's scan. This involved (i) performing single-subject ICA via MELODIC for all 210 subjects, (ii) manually labeling noise components for 10 subjects, (iii) training the FIX classifier on this hand-labeled data, (iv) automatically identifying noise components with this classifier for the remaining 200 subjects, then (v) regressing out noise components from each subject's scan. These steps may be carried out using the execution details below.

**Execution**

To prepare for AOMIC preprocessing, replace all fields in angled brackets with (e.g., `N_SUBS`, `WORLD_SIZE`) with appropriate values. Below are some of the selections used in the paper's preprocessing pipeline:
```
N_SUBS=210
N_SUBS_TRAIN=10
THRESHOLD=20
WORLD_SIZE=10
```
Note that the longest `step` (see ensuing commands) is `feat` which takes ~30 minutes per subject. Once all fields have been populated, perform the following: 

Before hand classification:
```console
$ sbatch slurm/preprocess-aomic.sh --step==bet
$ sbatch slurm/preprocess-aomic.sh --step==feat
$ sbatch slurm/preprocess-aomic.sh --step==fix-extract
```

To hand classify:
1. Download the `sub-<ID>.feat` directories to be hand classified. 
2. For each subject, open the terminal, execute the command below, then use the GUI to hand classify independent components. See this [post](https://www.caroline-nettekoven.com/post/ica-classification/) for guidance on how to classify components.
```console
$ fsleyes --scene melodic -ad ds002785-fix/sub-<ID>.feat/filtered_func_data.ica
```
3. For each subject, upload `hand_labels_noise.txt` to `ds002785-fix/sub-<ID>.feat`.

After hand classification:
```console
$ sbatch slurm/preprocess-aomic.sh --step==fix-train
$ sbatch slurm/preprocess-aomic.sh --step==fix-denoise
```

