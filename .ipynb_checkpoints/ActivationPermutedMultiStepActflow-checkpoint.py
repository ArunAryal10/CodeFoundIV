"""
Activation permuted multi-step activity Flow Modeling Pipeline

This script provides functions to:
- Load trial-wise beta activations for left/right button presses
- Create combined beta arrays (filtered and unfiltered) for analysis
- Load and shuffle actual betas for permutation testing
- Load precision-based FC matrices (graphLasso outputs)
- Compute activity flow predictions stepwise with convergence criteria
- Perform two-fold cross-validation on trial-wise predictions

"""

# Load required libraries and configure logging
import sys              
import os               
import h5py             
import numpy as np      
from scipy import stats 
import statsmodels.api as sm  
import nibabel as nib   
import copy             
import pandas as pd     
import pickle           
from sklearn.metrics import r2_score       
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler 
import logging          
import time             
import psutil           

# Set up logging format
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Constants and file paths specific to this environment
nregions = 360
# Parcels numbered 1 to 360 representing cortical regions
target_parcels = list(range(1, 361))

# Directory containing Cole-Anticevic network partition files
networkpartition_dir = '/projects/f_mc1689_1/AnalysisTools/ActflowToolbox/dependencies/ColeAnticevicNetPartition/'
# CIFTI dlabel file for network labels
dlabelfile = '/projects/f_mc1689_1/AnalysisTools/ActflowToolbox/dependencies/ColeAnticevicNetPartition/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii'

# Graphical Lasso outputs (precision FC matrices)
glasso_dir = '/projects/f_mc1689_1/MeiranNext/data/results/ArunResults/GlassoOutputTask/'
glasso_suffix = '-graphLasso_opt-R2.npy'

# Directory for saving permutation test results
output_dir = '/projects/f_mc1689_1/MeiranNext/data/results/ArunResults/ActivationPermTestFoundIV/'

# Directory and suffix for actual beta activations in HDF5 format
actual_data_dir = '/projects/f_mc1689_1/MeiranNext/data/results/ArunResults/ActualBetasParcelwiseH5/'
actual_suffix = '_actualBetas.h5'

# Directories for generated Python scripts
pythonScriptDir = "/projects/f_mc1689_1/MeiranNext/docs/scripts/ArunScripts/ActflowTesting/FoundIVRegionwiseTaskFCPermTest/generated_scripts"
ActflowFunc_dir = '/projects/f_mc1689_1/MeiranNext/docs/scripts/ArunScripts/ActflowTesting/FoundIVRegionwiseTaskFCPermTest/'

# Relevant conditions for Instruction and GO trials
relavantConds = ['Practice_Instruction', 'Novel_Instruction', 'Practice_Correct_GO', 'Novel_Correct_GO']

# Mapping of network names to parcel IDs
networkmappings = {
    'vis1': 1, 'vis2': 2, 'smn': 3, 'con': 4, 'dan': 5,
    'lan': 6, 'fpn': 7, 'aud': 8, 'dmn': 9, 'pmulti': 10,
    'vmm': 11, 'ora': 12
}

# All run identifiers in the HDF5 files
allRuns = [
    'test1_901', 'test2_1101', 'test3_1301', 'test4_1501',
    'test5_1701', 'test6_1901', 'test7_2101', 'test8_2301'
]

# Load network assignment labels for each parcel
networkdef = np.loadtxt(os.path.join(networkpartition_dir, 'cortex_parcel_network_assignments.txt'))


def create_allBetas(subj, networkmappings):
    """
    Load and combine beta activations for instruction and GO trials.

    Parameters
    ----------
    subj : str
        Subject identifier (used to construct HDF5 filename).
    networkmappings : dict
        Mapping of network names to parcel indices.

    Returns
    -------
    allBetas_filt : ndarray or None
        Filtered betas (instruction + go) for input networks, shape (T, P).
    allBetas_all : ndarray or None
        Unfiltered GO betas, shape (T, P).
    motor_responses : ndarray or None
        Motor response labels for each trial.
    """
    # Lists to collect data across runs and miniblocks
    allBetas_filt = []
    allBetas_all = []
    motor_responses = []

    # Define rule (FP) and stimulus (VIS1+VIS2) networks
    rule_networks = [networkmappings['fpn']]
    stim_networks = [networkmappings['vis1'], networkmappings['vis2']]
    input_regions = rule_networks + stim_networks

    # Instruction and GO conditions of interest
    instruction_conditions = ['Practice_Instruction', 'Novel_Instruction']
    go_conditions = ['Practice_Correct_GO', 'Novel_Correct_GO']

    # Construct path to the subject's HDF5 file
    h5_path = actual_data_dir + subj + actual_suffix

    try:
        with h5py.File(h5_path, 'r') as h5f:
            # Iterate over each run in the HDF5 group
            for run in allRuns:
                if run not in h5f:
                    logging.warning(f"Run {run} missing for {subj}")
                    continue

                # Iterate over miniblocks within the run
                for miniblock_key in h5f[run]:
                    mb = h5f[run][miniblock_key]

                    # Skip if no motor_response data
                    if 'motor_response' not in mb:
                        continue
                    
                    # Decode conditions and motor responses
                    conditions = [c.decode('utf-8') for c in mb['condition']]
                    betas = mb['betas'][:]
                    motor_resp = [r.decode('utf-8') for r in mb['motor_response']]

                    # Find and zero out non-rule parcels in instruction trial
                    try:
                        instr_idx = next(i for i, cond in enumerate(conditions) if cond in instruction_conditions)
                        instructionBetas = betas[instr_idx].astype(np.float32, copy=False)
                        instructionBetas[~np.isin(networkdef, rule_networks)] = 0
                    except StopIteration:
                        # No instruction trial found, skip
                        continue

                    # Process each GO trial
                    for i, cond in enumerate(conditions):
                        if cond in go_conditions:
                            goBetas = betas[i].astype(np.float32, copy=False)
                            # Append unfiltered betas
                            allBetas_all.append(goBetas)

                            # Zero out non-stimulus parcels
                            goBetas[~np.isin(networkdef, stim_networks)] = 0
                            # Combine instruction + go and zero out non-input parcels
                            combinedBetas = instructionBetas + goBetas
                            combinedBetas[~np.isin(networkdef, input_regions)] = 0

                            allBetas_filt.append(combinedBetas)
                            motor_responses.append(motor_resp[i])

    except Exception as e:
        logging.error(f"Failed to process subject {subj}: {e}")
        return None, None, None

    # Stack lists into arrays before returning
    if allBetas_filt:
        allBetas_filt = np.vstack(allBetas_filt).astype(np.float32)
        allBetas_all = np.vstack(allBetas_all).astype(np.float32)
        motor_responses = np.array(motor_responses)
        return allBetas_filt, allBetas_all, motor_responses
    else:
        logging.info("No valid betas found.")
        return None, None, None


def create_allBetasShuffled(subj, networkmappings):
    """
    Same as create_allBetas, but shuffles beta values within each trial for permutation testing.

    Parameters
    ----------
    subj : str
        Subject identifier.
    networkmappings : dict
        Mapping of network names to parcel IDs.

    Returns
    -------
    allBetas_filt : ndarray or None
        Filtered shuffled betas, shape (T, P).
    allBetas_all : ndarray or None
        Unfiltered shuffled GO betas, shape (T, P).
    motor_responses : ndarray or None
        Motor response labels.
    """
    allBetas_filt = []
    allBetas_all = []
    motor_responses = []

    rule_networks = [networkmappings['fpn']]
    stim_networks = [networkmappings['vis1'], networkmappings['vis2']]
    input_regions = rule_networks + stim_networks

    instruction_conditions = ['Practice_Instruction', 'Novel_Instruction']
    go_conditions = ['Practice_Correct_GO', 'Novel_Correct_GO']

    h5_path = actual_data_dir + subj + actual_suffix

    try:
        with h5py.File(h5_path, 'r') as h5f:
            for run in allRuns:
                if run not in h5f:
                    logging.warning(f"Run {run} missing for {subj}")
                    continue

                for miniblock_key in h5f[run]:
                    mb = h5f[run][miniblock_key]

                    if 'motor_response' not in mb:
                        continue
                    
                    conditions = [c.decode('utf-8') for c in mb['condition']]
                    betas = mb['betas'][:]
                    # Shuffle betas across each channel/trial for permutation
                    shuffled_betas = np.apply_along_axis(np.random.permutation, axis=1, arr=betas)
                    motor_resp = [r.decode('utf-8') for r in mb['motor_response']]

                    try:
                        instr_idx = next(i for i, cond in enumerate(conditions) if cond in instruction_conditions)
                        instructionBetas = shuffled_betas[instr_idx].astype(np.float32, copy=False)
                        instructionBetas[~np.isin(networkdef, rule_networks)] = 0
                    except StopIteration:
                        continue

                    for i, cond in enumerate(conditions):
                        if cond in go_conditions:
                            goBetas = shuffled_betas[i].astype(np.float32, copy=False)
                            allBetas_all.append(goBetas)

                            goBetas[~np.isin(networkdef, stim_networks)] = 0
                            combinedBetas = instructionBetas + goBetas
                            combinedBetas[~np.isin(networkdef, input_regions)] = 0

                            allBetas_filt.append(combinedBetas)
                            motor_responses.append(motor_resp[i])

    except Exception as e:
        logging.error(f"Failed to process subject {subj}: {e}")
        return None, None, None

    if allBetas_filt:
        allBetas_filt = np.vstack(allBetas_filt).astype(np.float32)
        allBetas_all = np.vstack(allBetas_all).astype(np.float32)
        motor_responses = np.array(motor_responses)
        return allBetas_filt, allBetas_all, motor_responses
    else:
        logging.info("No valid betas found.")
        return None, None, None


def load_and_print_pickle(file_path):
    """
    Load a Python object from a pickle file and return it.

    Parameters
    ----------
    file_path : str
        Full path to the .pkl file.

    Returns
    -------
    data : object
        Python object stored in the pickle.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        return data


def loadActualFC(subj):
    """
    Load precision-based FC matrix for a subject produced by graphical Lasso.

    Parameters
    ----------
    subj : str
        Subject identifier.

    Returns
    -------
    glassoFC : ndarray
        FC matrix of shape (nregions, nregions).
    """
    glasso_path = os.path.join(glasso_dir, subj + glasso_suffix)
    glassoFC = np.load(glasso_path)
    return glassoFC


def actflowcalc(actVect, fcMat):
    """
    Compute activity flow: weighted sum of activity from other nodes.

    Parameters
    ----------
    actVect : ndarray
        Activation vector of shape (P,).  
    fcMat : ndarray
        Functional connectivity matrix, shape (P, P).

    Returns
    -------
    pred : ndarray
        Predicted activation vector, shape (P,).
    """
    # Zero out self-connections
    np.fill_diagonal(fcMat, 0)
    # Matrix multiplication for weighted summation
    return fcMat @ actVect


def transfer_function(activity, transfer_func='linear', threshold=0, a=1):
    """
    Apply a non-linear transfer function to activity.

    Parameters
    ----------
    activity : ndarray
        Input activity vector.
    transfer_func : str
        Type of function: 'linear', 'relu', 'sigmoid', 'logit'.
    threshold : float
        Threshold for ReLU.
    a : float
        Scaling parameter for logit.

    Returns
    -------
    ndarray
        Transformed activity vector.
    """
    if transfer_func == 'linear':
        return activity
    elif transfer_func == 'relu':
        return np.maximum(activity, 0)
    elif transfer_func == 'sigmoid':
        return expit(activity)
    elif transfer_func == 'logit':
        # Avoid log(0) by clipping
        epsilon = 1e-6
        activity = np.clip(activity, epsilon, 1 - epsilon)
        return (1 / a) * np.log(activity / (1 - activity))
    else:
        raise ValueError(f"Unsupported transfer function: {transfer_func}")


def determine_optimal_steps(n_jobs, trainBetas_filt_list, trainBetas_all_list, FC_matrix, threshold, max_steps):
    """
    Determine convergence step count per trial and aggregate over trials.

    Parameters
    ----------
    n_jobs : int
        Number of parallel jobs.
    trainBetas_filt_list : list of ndarrays
        Filtered betas for training trials.
    trainBetas_all_list : list of ndarrays
        Unfiltered betas for training trials.
    FC_matrix : ndarray
        Functional connectivity matrix.
    threshold : float
        Minimum R change threshold for convergence.
    max_steps : int
        Maximum allowed steps.

    Returns
    -------
    optimal_steps : int
        Median step count across trials, rounded up.
    """
    def process_trial_1d(trainBetas_filt, trainBetas_all):
        combinedBetas = trainBetas_filt.copy()
        initial_r = None
        for step in range(1, max_steps + 1):
            # Predict next activation vector and apply linear transfer
            predBetas = actflowcalc(combinedBetas, FC_matrix)
            predBetas_transformed = transfer_function(predBetas, transfer_func='linear')
            # Normalize predictions
            scaler = StandardScaler()
            predBetas_scaled = scaler.fit_transform(predBetas_transformed.reshape(-1, 1)).squeeze()
            # Compute correlation with true activations
            new_r = np.corrcoef(trainBetas_all, predBetas_scaled)[0, 1]
            if initial_r is not None:
                change = new_r - initial_r
                # Stop if change is negative or below threshold
                if change < 0 or change < threshold:
                    return step
            initial_r = new_r
            combinedBetas = predBetas_scaled
        return max_steps

    # Run trials in parallel\    steps_list = Parallel(n_jobs=n_jobs)(
        delayed(process_trial_1d)(filt, all_) for filt, all_ in zip(trainBetas_filt_list, trainBetas_all_list)
    )
    # Median step count for robust estimate
    optimal_steps = int(np.ceil(np.median(steps_list)))
    return optimal_steps


def apply_steps_to_trial(trialBetas_filt, trialBetas_all, FC_matrix, steps):
    """
    Apply actflow model for a fixed number of steps on a single trial.

    Parameters
    ----------
    trialBetas_filt : ndarray
        Filtered input betas.
    trialBetas_all : ndarray
        Ground-truth betas.
    FC_matrix : ndarray
        Functional connectivity.
    steps : int
        Number of iterative steps to apply.

    Returns
    -------
    predBetas_scaled : ndarray
        Final predicted betas after normalization.
    """
    predBetas = trialBetas_filt.copy()
    for _ in range(steps):
        # Linear summation and transfer
        predBetas = actflowcalc(predBetas, FC_matrix)
        predBetas = transfer_function(predBetas, transfer_func='linear')
        # Normalize each step's output
        scaler = StandardScaler()
        predBetas_scaled = scaler.fit_transform(predBetas.reshape(-1, 1)).squeeze()
    return predBetas_scaled


def process_trial(args):
    """
    Wrapper to process a single trial in parallel.

    Parameters
    ----------
    args : tuple
        (trialBetas_filt, trialBetas_all, response, FC_matrix, steps)

    Returns
    -------
    stepwise_predBetas : ndarray
        Predicted betas after applying steps.
    """
    start_time = time.time()
    trialBetas_filt, trialBetas_all, response, FC_matrix, steps = args
    stepwise_predBetas = apply_steps_to_trial(trialBetas_filt, trialBetas_all, FC_matrix, steps)
    elapsed_time = time.time() - start_time
    return stepwise_predBetas


def cross_validated_actflow_per_trial_50_50(
    n_jobs, allBetas_filt, allBetas_all, motor_responses, FC_matrix, threshold, max_steps
):
    """
    Perform two-fold cross-validation (50/50 split) across trials.

    Parameters
    ----------
    n_jobs : int
        Number of parallel jobs.
    allBetas_filt : ndarray
        Filtered betas for all trials.
    allBetas_all : ndarray
        Ground-truth betas.
    motor_responses : ndarray
        Labels for each trial.
    FC_matrix : ndarray
        Functional connectivity.
    threshold : float
        Convergence threshold.
    max_steps : int
        Maximum iterative steps.

    Returns
    -------
    predicted_betas : ndarray
        Predicted betas for all trials after cross-validation.
    """
    # Convert inputs to NumPy arrays
    allBetas_filt = np.array(allBetas_filt)
    allBetas_all = np.array(allBetas_all)
    motor_responses = np.array(motor_responses)

    num_trials = allBetas_all.shape[0]
    # Random shuffle and split indices
    indices = np.random.permutation(num_trials)
    split_point = num_trials // 2
    indices_A = indices[:split_point]
    indices_B = indices[split_point:]

    # Data for fold A and fold B
    betas_filt_A, betas_filt_B = allBetas_filt[indices_A], allBetas_filt[indices_B]
    betas_all_A, betas_all_B = allBetas_all[indices_A], allBetas_all[indices_B]

    # Determine optimal steps on opposite folds
    optimal_steps_A = determine_optimal_steps(n_jobs, betas_filt_B, betas_all_B, FC_matrix, threshold, max_steps)
    optimal_steps_B = determine_optimal_steps(n_jobs, betas_filt_A, betas_all_A, FC_matrix, threshold, max_steps)
    final_steps = max(optimal_steps_A, optimal_steps_B)

    # Apply model to all trials with standardized steps
    args_list = [
        (trial_filt, trial_all, resp, FC_matrix, final_steps)
        for trial_filt, trial_all, resp in zip(allBetas_filt, allBetas_all, motor_responses)
    ]
    results = Parallel(n_jobs=n_jobs)(delayed(process_trial)(args) for args in args_list)
    predicted_betas = np.array(results)
    return predicted_betas





























# #load libraries
# import sys
# import os
# import h5py
# import numpy as np
# from scipy import stats
# import statsmodels.api as sm
# import nibabel as nib
# import copy
# import pandas as pd
# import pickle
# from sklearn.metrics import r2_score
# from joblib import Parallel, delayed
# from sklearn.preprocessing import StandardScaler
# import logging
# import time
# import psutil
# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# # Constants and paths (update these paths according to your environment)
# nregions = 360
# target_parcels = list(range(1, 361))

# networkpartition_dir = '/projects/f_mc1689_1/AnalysisTools/ActflowToolbox/dependencies/ColeAnticevicNetPartition/'
# dlabelfile = '/projects/f_mc1689_1/AnalysisTools/ActflowToolbox/dependencies/ColeAnticevicNetPartition/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii'

# glasso_dir = '/projects/f_mc1689_1/MeiranNext/data/results/ArunResults/GlassoOutputTask/'
# glasso_suffix = '-graphLasso_opt-R2.npy'

# output_dir='/projects/f_mc1689_1/MeiranNext/data/results/ArunResults/ActivationPermTestFoundIV/'

# actual_data_dir = '/projects/f_mc1689_1/MeiranNext/data/results/ArunResults/ActualBetasParcelwiseH5/'
# actual_suffix = '_actualBetas.h5'

# pythonScriptDir = "/projects/f_mc1689_1/MeiranNext/docs/scripts/ArunScripts/ActflowTesting/FoundIVRegionwiseTaskFCPermTest/generated_scripts"
# ActflowFunc_dir = '/projects/f_mc1689_1/MeiranNext/docs/scripts/ArunScripts/ActflowTesting/FoundIVRegionwiseTaskFCPermTest/'

# relavantConds = ['Practice_Instruction', 'Novel_Instruction', 'Practice_Correct_GO', 'Novel_Correct_GO']

# networkmappings = {
#     'vis1': 1, 'vis2': 2, 'smn': 3, 'con': 4, 'dan': 5,
#     'lan': 6, 'fpn': 7, 'aud': 8, 'dmn': 9, 'pmulti': 10,
#     'vmm': 11, 'ora': 12
# }

# allRuns = [
#     'test1_901', 'test2_1101', 'test3_1301', 'test4_1501',
#     'test5_1701', 'test6_1901', 'test7_2101', 'test8_2301'
# ]


# # Load network definitions and labels
# networkdef = np.loadtxt(os.path.join(networkpartition_dir, 'cortex_parcel_network_assignments.txt'))


# def create_allBetas(subj, networkmappings):
#     # logging.info(f"Creating combined betas for subject {subj}")

#     allBetas_filt = []
#     allBetas_all = []
#     motor_responses = []

#     # Define network groups
#     rule_networks = [networkmappings['fpn']]
#     stim_networks = [networkmappings['vis1'], networkmappings['vis2']]
#     input_regions = rule_networks + stim_networks

#     # Conditions
#     instruction_conditions = ['Practice_Instruction', 'Novel_Instruction']
#     go_conditions = ['Practice_Correct_GO', 'Novel_Correct_GO']

#     # Path to new HDF5 file
#     h5_path = actual_data_dir + subj + actual_suffix 

#     try:
#         with h5py.File(h5_path, 'r') as h5f:
#             for run in allRuns:
#                 if run not in h5f:
#                     logging.warning(f"Run {run} missing for {subj}")
#                     continue

#                 for miniblock_key in h5f[run]:
#                     mb = h5f[run][miniblock_key]

#                     # Skip miniblocks with no motor_response field
#                     if 'motor_response' not in mb:
#                         continue
                        
#                     conditions = [c.decode('utf-8') for c in mb['condition']]
#                     betas = mb['betas'][:]
#                     motor_resp = [r.decode('utf-8') for r in mb['motor_response']] 

#                     # Find instruction trial
#                     try:
#                         instr_idx = next(i for i, cond in enumerate(conditions) if cond in instruction_conditions)
#                         instructionBetas = betas[instr_idx].astype(np.float32, copy=False)
#                         instructionBetas[~np.isin(networkdef, rule_networks)] = 0
#                     except StopIteration:
#                         continue  # Skip miniblock if no instruction trial found

#                     # Process GO trials
#                     for i, cond in enumerate(conditions):
#                         if cond in go_conditions:
#                             goBetas = betas[i].astype(np.float32, copy=False)
#                             allBetas_all.append(goBetas)

#                             goBetas[~np.isin(networkdef, stim_networks)] = 0
#                             combinedBetas = instructionBetas + goBetas
#                             combinedBetas[~np.isin(networkdef, input_regions)] = 0

#                             allBetas_filt.append(combinedBetas)
#                             motor_responses.append(motor_resp[i])

#     except Exception as e:
#         logging.error(f"Failed to process subject {subj}: {e}")
#         return None, None, None

#     if allBetas_filt:
#         allBetas_filt = np.vstack(allBetas_filt).astype(np.float32)
#         allBetas_all = np.vstack(allBetas_all).astype(np.float32)
#         motor_responses = np.array(motor_responses)
#         return allBetas_filt, allBetas_all, motor_responses
#     else:
#         logging.info("No valid betas found.")
#         return None, None, None

    
# def create_allBetasShuffled(subj, networkmappings):
#     # logging.info(f"Creating combined betas for subject {subj}")

#     allBetas_filt = []
#     allBetas_all = []
#     motor_responses = []

#     # Define network groups
#     rule_networks = [networkmappings['fpn']]
#     stim_networks = [networkmappings['vis1'], networkmappings['vis2']]
#     input_regions = rule_networks + stim_networks

#     # Conditions
#     instruction_conditions = ['Practice_Instruction', 'Novel_Instruction']
#     go_conditions = ['Practice_Correct_GO', 'Novel_Correct_GO']

#     # Path to new HDF5 file
#     h5_path = actual_data_dir + subj + actual_suffix 

#     try:
#         with h5py.File(h5_path, 'r') as h5f:
#             for run in allRuns:
#                 if run not in h5f:
#                     logging.warning(f"Run {run} missing for {subj}")
#                     continue

#                 for miniblock_key in h5f[run]:
#                     mb = h5f[run][miniblock_key]

#                     # Skip miniblocks with no motor_response field
#                     if 'motor_response' not in mb:
#                         continue
                    
#                     conditions = [c.decode('utf-8') for c in mb['condition']]
#                     betas = mb['betas'][:]
                    
#                     shuffled_betas = np.apply_along_axis(np.random.permutation, axis=1, arr=betas)
                    
#                     motor_resp = [r.decode('utf-8') for r in mb['motor_response']]

#                     # Find instruction trial
#                     try:
#                         instr_idx = next(i for i, cond in enumerate(conditions) if cond in instruction_conditions)
#                         instructionBetas = shuffled_betas[instr_idx].astype(np.float32, copy=False)
#                         instructionBetas[~np.isin(networkdef, rule_networks)] = 0
#                     except StopIteration:
#                         continue  # Skip miniblock if no instruction trial found

#                     # Process GO trials
#                     for i, cond in enumerate(conditions):
#                         if cond in go_conditions:
#                             goBetas = shuffled_betas[i].astype(np.float32, copy=False)
#                             allBetas_all.append(goBetas)

#                             goBetas[~np.isin(networkdef, stim_networks)] = 0
#                             combinedBetas = instructionBetas + goBetas
#                             combinedBetas[~np.isin(networkdef, input_regions)] = 0

#                             allBetas_filt.append(combinedBetas)
#                             motor_responses.append(motor_resp[i])

#     except Exception as e:
#         logging.error(f"Failed to process subject {subj}: {e}")
#         return None, None, None

#     if allBetas_filt:
#         allBetas_filt = np.vstack(allBetas_filt).astype(np.float32)
#         allBetas_all = np.vstack(allBetas_all).astype(np.float32)
#         motor_responses = np.array(motor_responses)
#         return allBetas_filt, allBetas_all, motor_responses
#     else:
#         logging.info("No valid betas found.")
#         return None, None, None
   
    
    
# def load_and_print_pickle(file_path):
#     with open(file_path, 'rb') as file:
#         data = pickle.load(file)
#         return data



# def loadActualFC(subj):
#     glasso_path = os.path.join(glasso_dir, subj + glasso_suffix)
#     glassoFC = np.load(glasso_path)
#     return glassoFC


# def actflowcalc(actVect, fcMat):
#     np.fill_diagonal(fcMat, 0)  # Ensure no self-connections
#     return fcMat @ actVect  # Matrix multiplication


# def transfer_function(activity, transfer_func='linear', threshold=0, a=1):
#     """
#     Define input transfer function.
    
#     Parameters:
#     - activity: NumPy array
#     - transfer_func: str, one of 'linear', 'relu', 'sigmoid', 'logit'
#     - threshold: float, used in 'relu'
#     - a: float, used in 'logit'
    
#     Returns:
#     - Transformed activity vector.
#     """
#     if transfer_func == 'linear':
#         return activity
#     elif transfer_func == 'relu':
#         return np.maximum(activity, 0)
#     elif transfer_func == 'sigmoid':
#         return expit(activity)
#     elif transfer_func == 'logit':
#         # To avoid division by zero or log(0), clip activity
#         epsilon = 1e-6
#         activity = np.clip(activity, epsilon, 1 - epsilon)
#         return (1 / a) * np.log(activity / (1 - activity))
#     else:
#         raise ValueError(f"Unsupported transfer function: {transfer_func}")



# def determine_optimal_steps(n_jobs, trainBetas_filt_list, trainBetas_all_list, FC_matrix, threshold, max_steps):
#     # logging.info("Determining optimal steps...")

#     def process_trial_1d(trainBetas_filt, trainBetas_all):
#         combinedBetas = trainBetas_filt.copy()
#         initial_r = None
#         for step in range(1, max_steps + 1):
#             predBetas = actflowcalc(combinedBetas, FC_matrix)
#             # Apply transfer function after linear summation
#             predBetas_transformed = transfer_function(predBetas, transfer_func='linear')
            
#             #apply normalization 
#             scaler = StandardScaler()
#             predBetas_scaled = scaler.fit_transform(predBetas_transformed.reshape(-1, 1)).squeeze()
            
#             new_r = np.corrcoef(trainBetas_all, predBetas_scaled)[0, 1]
            
#             if initial_r is not None:  
#                 change = new_r - initial_r
#                 if change < 0 or change < threshold:
#                     return step 
#             initial_r = new_r
#             combinedBetas = predBetas_scaled
#         return max_steps  # If maximum steps reached without convergence

#     # Parallel processing of trials using default 'loky' backend
#     steps_list = Parallel(n_jobs=n_jobs)(
#         delayed(process_trial_1d)(trainBetas_filt, trainBetas_all) for trainBetas_filt, trainBetas_all in zip(trainBetas_filt_list, trainBetas_all_list)
#     )
    
#     # Aggregate steps to determine optimal number of steps
#     optimal_steps = int(np.ceil(np.median(steps_list)))
#     # logging.info(f"Optimal number of steps determined from training trials: {optimal_steps}")
#     return optimal_steps



# def apply_steps_to_trial(trialBetas_filt, trialBetas_all, FC_matrix, steps):
#     """
#     Applies the actflow model to a single trial using the specified number of steps.
#     Returns the predicted betas for the trial.
#     """

#     predBetas = trialBetas_filt.copy() 
#     # stepwise_predBetas = []  # Initialize list to store betas for each step
    
#     for _ in range(steps):
#         # Apply linear summation
#         predBetas = actflowcalc(predBetas, FC_matrix)
#         # Apply transfer function after linear summation
#         predBetas = transfer_function(predBetas, transfer_func='linear')
        
#         #apply normalization 
#         scaler = StandardScaler()
#         predBetas_scaled = scaler.fit_transform(predBetas.reshape(-1, 1)).squeeze()
            
#     return predBetas_scaled  # Convert list to array of shape (num_steps, nNodes)


# def process_trial(args):
#     start_time = time.time()
#     # logging.info(f"Process {os.getpid()} started processing a trial")
#     trialBetas_filt, trialBetas_all, response, FC_matrix, steps = args
#     stepwise_predBetas = apply_steps_to_trial(trialBetas_filt, trialBetas_all, FC_matrix, steps)
#     elapsed_time = time.time() - start_time
#     # logging.info(f"Process {os.getpid()} finished processing a trial in {elapsed_time:.2f} seconds")
#     return stepwise_predBetas



# def cross_validated_actflow_per_trial_50_50(
#     n_jobs, allBetas_filt, allBetas_all, motor_responses, FC_matrix, threshold, max_steps
# ):
#     """
#     Performs two-fold cross-validation with a 50/50 split.
#     Each half is used to determine the optimal number of steps for the other half.
#     """    
#     # Convert to NumPy arrays
#     allBetas_filt = np.array(allBetas_filt)
#     allBetas_all = np.array(allBetas_all)
#     motor_responses = np.array(motor_responses)
    
#     num_trials = allBetas_all.shape[0]

#     # Shuffle and split the data
#     indices = np.random.permutation(num_trials)
#     split_point = num_trials // 2
#     indices_A = indices[:split_point]
#     indices_B = indices[split_point:]

#     # Create folds
#     betas_filt_A = allBetas_filt[indices_A]
#     betas_filt_B = allBetas_filt[indices_B]
    
#     betas_all_A = allBetas_all[indices_A]
#     betas_all_B = allBetas_all[indices_B]
    
    
#     responses_A = motor_responses[indices_A]
#     responses_B = motor_responses[indices_B]

#     # First fold
#     # logging.info("First fold: Determining optimal steps using Fold B")
#     optimal_steps_A = determine_optimal_steps(n_jobs, betas_filt_B, betas_all_B, FC_matrix, threshold, max_steps)
#     # logging.info(f"Optimal steps for Fold A: {optimal_steps_A}")
    
#     # Second fold
#     # logging.info("Second fold: Determining optimal steps using Fold A")
#     optimal_steps_B = determine_optimal_steps(n_jobs, betas_filt_A, betas_all_A, FC_matrix, threshold, max_steps)
#     # logging.info(f"Optimal steps for Fold B: {optimal_steps_B}")
    
#     # Determine the final step count for both folds
#     final_steps = max(optimal_steps_A, optimal_steps_B)
#     # logging.info(f"Standardizing number of steps to {final_steps}")

#     # Apply the model to betas_A
#     # logging.info("Applying model to whole data")
#     args_list = [
#         (trialBetas_filt, trialBetas_all, response, FC_matrix, final_steps)
#         for trialBetas_filt, trialBetas_all, response in zip(allBetas_filt, allBetas_all, motor_responses)
#     ]
#     results = Parallel(n_jobs=n_jobs)(
#         delayed(process_trial)(args) for args in args_list
#     )
#     predicted_betas = results 
    

#     predicted_betas = np.array(predicted_betas)

#     return predicted_betas


