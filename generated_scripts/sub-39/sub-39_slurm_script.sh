#!/bin/bash
#SBATCH --partition=price
#SBATCH --job-name=Actflow_sub-39
#SBATCH --requeue
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=10000
#SBATCH --output=/projects/f_mc1689_1/MeiranNext/docs/scripts/ArunScripts/ActflowTesting/FoundIVRegionwiseTaskFCPermTest/generated_scripts/sub-39/sub-39.out
#SBATCH --error=/projects/f_mc1689_1/MeiranNext/docs/scripts/ArunScripts/ActflowTesting/FoundIVRegionwiseTaskFCPermTest/generated_scripts/sub-39/sub-39.err
#SBATCH --export=ALL

exec > >(tee -i /projects/f_mc1689_1/MeiranNext/docs/scripts/ArunScripts/ActflowTesting/FoundIVRegionwiseTaskFCPermTest/generated_scripts/${SLURM_JOB_ID}.out)
exec 2>&1

source /projects/f_mc1689_1/AnalysisTools/anaconda3b/bin/activate
time python /projects/f_mc1689_1/MeiranNext/docs/scripts/ArunScripts/ActflowTesting/FoundIVRegionwiseTaskFCPermTest/generated_scripts/sub-39/sub-39_actflow_script.py
