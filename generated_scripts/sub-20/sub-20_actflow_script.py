import os
import sys
import h5py
import numpy as np
import nibabel as nib
import copy
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import pickle
from sklearn.metrics import r2_score, mean_absolute_error
from joblib import Parallel, delayed
import logging
import time
import psutil

sys.path.append('/projects/f_mc1689_1/MeiranNext/docs/scripts/ArunScripts/ActflowTesting/FoundIVRegionwiseTaskFCPermTest/')

from ActivationPermutedMultiStepActflow import *

subj = 'sub-20'
output_subj_path = '/projects/f_mc1689_1/MeiranNext/data/results/ArunResults/ActivationPermTestFoundIV/sub-20'

n_jobs = 5  # Adjust as needed based on available CPUs

def main():
    # Number of permutations
    nIters = 1000

    allBetas_filt, allBetas_all, motor_responses = create_allBetas(subj, networkmappings)

    if allBetas_filt is None or len(allBetas_filt) == 0:
        print(f'No valid betas found for subject {subj}')
        return

    nRegions, nTrials = allBetas_all.shape[1], allBetas_all.shape[0]
    pred_betas_all_perms = np.zeros((nTrials, nRegions, nIters), dtype=np.float32)

    FC_non_shuffled_matrix = loadActualFC(subj)
    pred_betas = cross_validated_actflow_per_trial_50_50(
    n_jobs=n_jobs,
    allBetas_filt=allBetas_filt,
    allBetas_all=allBetas_all,
    motor_responses=motor_responses,
    FC_matrix=FC_non_shuffled_matrix,
    threshold=0.01,
    max_steps=10
    )
    # Run permutations
    for i in range(nIters):
        allBetas_filt, allBetas_all, motor_responses = create_allBetasShuffled(subj, networkmappings)
        predicted_betas = cross_validated_actflow_per_trial_50_50(
        n_jobs=n_jobs,
        allBetas_filt=allBetas_filt,
        allBetas_all=allBetas_all,
        motor_responses=motor_responses,
        FC_matrix=FC_non_shuffled_matrix,
        threshold=0.01,
        max_steps=10
        )
        pred_betas_all_perms[:, :, i] = predicted_betas

    # Save all permutations to HDF5
    output_file = os.path.join(output_subj_path, f'{subj}_PermPredBetas.h5')
    with h5py.File(output_file, 'w') as h5f:
        h5f.create_dataset('NonPermutedBetas', data = pred_betas)
        h5f.create_dataset('PermutedBetas', data=pred_betas_all_perms)
    print(f'Permutation predictions saved for subject {subj}', flush=True)

if __name__ == '__main__':
    main()
