#!/bin/bash --login
# Script to run on the  slurm cluster to train and test the model
######################
# slurm settings
######################
#SBATCH -J test_slurm # job name
#SBATCH -t 3:00:00 # Sets the time in hh:mm:ss to 10 hours
#SBATCH --mem=10G # sets the amount of memory (MB is the default if you don't put any letter)
#SBATCH --nodes=1 # this should be kept to 1 even if more than 1 cores is requested
#SBATCH --ntasks=1 # this should be kept to 1 even if more than 1 cores is requested
#SBATCH --cpus-per-task=2 # this specifies the number of cores, in this case 2
#SBATCH -q normal # queues available. Check the max. time and resources of each with: 'sacctmgr show qos format=name%10,maxwall%14,maxtrespu%20,maxtres%20,maxjobspu%10'
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH -o log_files/stdout_%x.out # file where to write stdout. %x refers to the job name. Other useful ones are %j for job id
#SBATCH -e log_files/stderr_%x.err # file where to write stderr.
# make bash behave more robustly
# in the cluster course they recommended to add this to the bash script to ensure the exit codes are properly handled by bash
set -e
set -o pipefail
####################
#script starts
####################
cd "$(dirname "$0")"
# Manually source Conda

conda init 
which conda
# Activate Conda environment
conda activate test
python batch128lr01epoch10.py
