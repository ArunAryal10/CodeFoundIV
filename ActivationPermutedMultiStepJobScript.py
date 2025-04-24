import os

# # List of subjects
subjNums = [ 'sub-3', 'sub-4', 'sub-5', 'sub-6', 'sub-7', 'sub-8', 'sub-10', 
         'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20',
           'sub-21', 'sub-22', 'sub-23', 'sub-24', 'sub-25', 'sub-26', 'sub-28', 'sub-29', 'sub-30',
          'sub-31', 'sub-32', 'sub-33', 'sub-34', 'sub-35',  'sub-36', 'sub-37', 'sub-38', 'sub-39', 'sub-40'] 

# subjNums = ['sub-10']

# Constants and paths (update these paths according to your environment)
output_dir='/projects/f_mc1689_1/MeiranNext/data/results/ArunResults/ActivationPermTestFoundIV/'

pythonScriptDir="/projects/f_mc1689_1/MeiranNext/docs/scripts/ArunScripts/ActflowTesting/FoundIVRegionwiseTaskFCPermTest/generated_scripts"
ActflowFunc_dir='/projects/f_mc1689_1/MeiranNext/docs/scripts/ArunScripts/ActflowTesting/FoundIVRegionwiseTaskFCPermTest/'

# glasso_path = '/projects/f_mc1689_1/MeiranNext/docs/scripts/ArunScripts/ActflowTesting/MultiStepActflowRegionwiseTaskFC/task_fc_avg.npy'

for subj in subjNums:
    output_subj_path = os.path.join(output_dir, subj)
    os.makedirs(output_subj_path, exist_ok=True)

    subj_dir = os.path.join(pythonScriptDir, subj)
    os.makedirs(subj_dir, exist_ok=True)

    # Generate a per-subject Python script
    python_script_path = os.path.join(subj_dir, f"{subj}_actflow_script.py")
    with open(python_script_path, "w") as file_python:
        # Import statements and environment setup
        file_python.write("import os\n")
        file_python.write("import sys\n")
        file_python.write("import h5py\n")
        file_python.write("import numpy as np\n")
        file_python.write("import nibabel as nib\n")
        file_python.write("import copy\n")
        file_python.write("import pandas as pd\n")
        file_python.write("from scipy import stats\n")
        file_python.write("import statsmodels.api as sm\n")
        file_python.write("import pickle\n")
        file_python.write("from sklearn.metrics import r2_score, mean_absolute_error\n")
        file_python.write("from joblib import Parallel, delayed\n")
        file_python.write("import logging\n")
        file_python.write("import time\n")
        file_python.write("import psutil\n")
        file_python.write("\n")
        # Add the directory containing MultiStepActflow.py to sys.path
        file_python.write(f"sys.path.append('{ActflowFunc_dir}')\n")
        # file_python.write(f"sys.path.append('{glasso_path}')\n")
        file_python.write("\n")
        
        # Import necessary functions
        # file_python.write("from NoTempMultiStepActflow1StepTrialInst import *\n") 
        file_python.write("from ActivationPermutedMultiStepActflow import *\n")
        # file_python.write("from NoTempMultiStepActflow2Step import *\n")
        # file_python.write("from MultiStepActflow import *\n")
        file_python.write("\n")
        
        # Set variables
        file_python.write(f"subj = '{subj}'\n")
        # file_python.write(f"glasso_path = '{glasso_path}'\n")
        file_python.write(f"output_subj_path = '{output_subj_path}'\n")
        file_python.write("\n")
        # Hard-code n_jobs
        file_python.write("n_jobs = 5  # Adjust as needed based on available CPUs\n")
        file_python.write("\n")
        
        # Now, process the subject
        file_python.write("def main():\n")
        file_python.write("    # Number of permutations\n")
        file_python.write("    nIters = 1000\n")
        file_python.write("\n")
        file_python.write("    allBetas_filt, allBetas_all, motor_responses = create_allBetas(subj, networkmappings)\n")
        file_python.write("\n")
        file_python.write("    if allBetas_filt is None or len(allBetas_filt) == 0:\n")
        file_python.write("        print(f'No valid betas found for subject {subj}')\n")
        file_python.write("        return\n")
        file_python.write("\n")
        file_python.write("    nRegions, nTrials = allBetas_all.shape[1], allBetas_all.shape[0]\n")
        file_python.write("    pred_betas_all_perms = np.zeros((nTrials, nRegions, nIters), dtype=np.float32)\n")
        file_python.write("\n")
        file_python.write("    FC_non_shuffled_matrix = loadActualFC(subj)\n")
        file_python.write("    pred_betas = cross_validated_actflow_per_trial_50_50(\n")
        file_python.write("    n_jobs=n_jobs,\n")
        file_python.write("    allBetas_filt=allBetas_filt,\n")
        file_python.write("    allBetas_all=allBetas_all,\n")
        file_python.write("    motor_responses=motor_responses,\n")
        file_python.write("    FC_matrix=FC_non_shuffled_matrix,\n")
        file_python.write("    threshold=0.01,\n")
        file_python.write("    max_steps=10\n")
        file_python.write("    )\n")
        
        file_python.write("    # Run permutations\n")
        file_python.write("    for i in range(nIters):\n")
        file_python.write("        allBetas_filt, allBetas_all, motor_responses = create_allBetasShuffled(subj, networkmappings)\n")
        file_python.write("        predicted_betas = cross_validated_actflow_per_trial_50_50(\n")
        file_python.write("        n_jobs=n_jobs,\n")
        file_python.write("        allBetas_filt=allBetas_filt,\n")
        file_python.write("        allBetas_all=allBetas_all,\n")
        file_python.write("        motor_responses=motor_responses,\n")
        file_python.write("        FC_matrix=FC_non_shuffled_matrix,\n")
        file_python.write("        threshold=0.01,\n")
        file_python.write("        max_steps=10\n")
        file_python.write("        )\n")
        file_python.write("        pred_betas_all_perms[:, :, i] = predicted_betas\n")
        file_python.write("\n")
        file_python.write("    # Save all permutations to HDF5\n")
        file_python.write("    output_file = os.path.join(output_subj_path, f'{subj}_PermPredBetas.h5')\n")
        file_python.write("    with h5py.File(output_file, 'w') as h5f:\n")
        file_python.write("        h5f.create_dataset('NonPermutedBetas', data = pred_betas)\n")
        file_python.write("        h5f.create_dataset('PermutedBetas', data=pred_betas_all_perms)\n")
        file_python.write("    print(f'Permutation predictions saved for subject {subj}', flush=True)\n")
        file_python.write("\n")
        file_python.write("if __name__ == '__main__':\n")
        file_python.write("    main()\n")

    # Make the Python script executable
    os.chmod(python_script_path, 0o755)

    # Generate the SLURM job script for each subject
    slurm_script_path = os.path.join(subj_dir, f"{subj}_slurm_script.sh")
    with open(slurm_script_path, "w") as file_slurm:
        file_slurm.write("#!/bin/bash\n")
        file_slurm.write(f"#SBATCH --partition=price\n")
        file_slurm.write(f"#SBATCH --job-name=Actflow_{subj}\n")
        file_slurm.write("#SBATCH --requeue\n")
        file_slurm.write("#SBATCH --time=00:30:00\n")  # Adjust time as needed
        file_slurm.write("#SBATCH --nodes=1\n")
        file_slurm.write("#SBATCH --ntasks=1\n")
        file_slurm.write("#SBATCH --cpus-per-task=10\n")  # Set to 45 CPUs as per your request
        file_slurm.write("#SBATCH --mem=10000\n")          # Memory allocation (250 GB)
        file_slurm.write(f"#SBATCH --output={subj_dir}/{subj}.out\n")
        file_slurm.write(f"#SBATCH --error={subj_dir}/{subj}.err\n")
        file_slurm.write("#SBATCH --export=ALL\n")
        file_slurm.write("\n")
        # Redirect output and error streams
        file_slurm.write("exec > >(tee -i " + pythonScriptDir + "/${SLURM_JOB_ID}.out)\n")
        file_slurm.write("exec 2>&1\n")
        file_slurm.write("\n")
        # Activate the environment before running the generated Python script
        file_slurm.write("source /projects/f_mc1689_1/AnalysisTools/anaconda3b/bin/activate\n")
        file_slurm.write(f"time python {python_script_path}\n")

    # Make the SLURM script executable
    os.chmod(slurm_script_path, 0o755)

    # Submit the SLURM job
    os.system(f"sbatch {slurm_script_path}")


