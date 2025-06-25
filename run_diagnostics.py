import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy
import torch
from statistics import median



def plot_diagnostic(
        array, 
        error, 
        xlabel='Time', 
        ylabel='Covariance', 
        title='',
        title_sz=18,
        hline=None,
        path=None
    ):
    if len(array) != len(error):
        raise ValueError("`array` and `error` must have the same length.")
    
    x = np.arange(len(array))  # Use indices for the x-axis
    
    # Calculate confidence bounds
    lower_bound = array - error
    upper_bound = array + error
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, array, label='Mean Value', color='blue', linewidth=2)
    plt.fill_between(
        x, lower_bound, upper_bound, 
        color='blue', 
        alpha=0.2, 
        label='Confidence Region'
    )
    if hline is not None:
        plt.axhline(y=hline, linestyle=':', color='red')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=title_sz)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show or save plot
    if path: 
        plt.savefig(path)
        plt.close()
    else:
        plt.show()
    

# NOTE: (AOMIC Diagnostics)
#   [2025-04-14]
#       - 20-subject AOMIC
#           * Orthogonal --> passes independent tests
#           * Oblique --> diagonal passes independent tests
#       - 210-subject AOMIC
#           * Orthogonal --> nearly passes joint test (1 off-diagonal positive test)
#           * Oblique --> diagonal passes independent tests


# Globals
analysis = 'sim-defense_ktrue-2_kest-2_faccov-orth/sim-0/rep-0'
est_method = 'rbels'
rot_method = 'varimax'
n_facs = 2
n_subs = 20
n_time = 500
dir_out = os.path.join('/storage/group/kms8227/default/ffa-p2-priv', 'out', analysis)
dir_results = os.path.join(
    '/storage/group/kms8227/default/ffa-p2-priv', 
    'out', analysis, 'results'
)

# NOTE: For both orthogonal and oblique models, it is not critical that each
# factor have UNIT variance. What is important is that each factor have the
# SAME variance. This ensures that orthogonal rotation does not alter the 
# coavraince structure. This means universal scaling is permissible. We can 
# scale by any value. For now, these scales must be set manually.
scales = {
    'aomic-prod-1': {
        'varimax': 1.075, 
        'quartimin': 1.05,
    },
    'aomic-prod-2': {
        'varimax': 1.18,
        'quartimin': 1.14,
    },
    'sim-defense_ktrue-2_kest-2_faccov-orth/sim-0/rep-0': {
        'varimax': 1.08
    }
}

# Get multipliers for independent and Bonferroni confidence intervals
alpha = 0.05
n_tests = sum(list(range(1, n_facs + 1)))
alpha_bf = alpha / n_tests
mult = scipy.stats.t.ppf(1 - alpha / 2, n_subs - 1)
mult_bf = scipy.stats.t.ppf(1 - alpha_bf / 2, n_subs - 1)
# mult = scipy.stats.norm.ppf(1 - alpha / 2)
# mult_bf = scipy.stats.norm.ppf(1 - alpha_bf / 2)

# Read factors (aggregated)
path = os.path.join(dir_out, f'facs_split-full_m-{est_method}_rot-{rot_method}_reg-3.pt')
facs = torch.load(path)
facs /= scales[analysis][rot_method]


# Read factors (disaggregated)
# facs = torch.zeros(n_subs, n_facs, n_time)
# for n in range(n_subs):
#     # facs[n] = torch.load(os.path.join(dir_out, f'facs_n-{n}_.pt')).t()
#     # facs[n] = torch.load(os.path.join(dir_out, f'facs_split-full_reg-3_n-{n}_{method}.pt'))
#     facs[n] = torch.load(os.path.join(dir_out, 'est-facs', f'facs_split-full_m-{est_method}_rot-{rot_method}_reg-3_n-{n}.pt'))
# facs /= scales[analysis][rot_method]


# ---------- "On Average" Diagnostics ---------- #

df = pd.DataFrame(columns=['k1', 'k2', 'mean', 'lb', 'ub'])
n_tot = torch.tensor(facs.shape[0]*facs.shape[2])
for k1 in range(n_facs): 
    for k2 in range(k1, n_facs):

        # Get gamma_{i,k1,k2}, the integral approximations by subject
        tmp = facs[:,k1,:] * facs[:,k2,:]
        tmp = torch.mean(tmp, dim=1)

        # Get CI
        mean = torch.mean(tmp)
        sd = torch.std(tmp)
        row = {
            'k1': k1,
            'k2': k2,
            'mean': mean.item(),
            'lb': (mean - mult * sd / np.sqrt(facs.shape[0])).item(),
            'ub': (mean + mult * sd / np.sqrt(facs.shape[0])).item(),
            'lb_bf': (mean - mult_bf * sd / np.sqrt(facs.shape[0])).item(),
            'ub_bf': (mean + mult_bf * sd / np.sqrt(facs.shape[0])).item(),
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        # lb = round((mean - 2 * sd / np.sqrt(facs.shape[0])).item(), 3)
        # ub = round((mean + 2 * sd / np.sqrt(facs.shape[0])).item(), 3)
        # print(f"({k1},{k2}) --> [{lb}, {ub}]")

path = os.path.join(
    '/storage/group/kms8227/default/ffa-p2-priv', 
    'out', analysis, 'results', f'fac-corrs_m-{est_method}_rot-{rot_method}.csv'
)
df.to_csv(path, index=False)

