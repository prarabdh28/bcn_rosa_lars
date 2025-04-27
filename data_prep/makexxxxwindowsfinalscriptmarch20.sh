#!/bin/bash --login
#SBATCH -J bigwig_to_bed_intersect  # Job name
#SBATCH -t 4:00:00                  # Time limit (hh:mm:ss)
#SBATCH --mem=30G                   # Memory allocation
#SBATCH --nodes=1                    # Number of nodes (keep 1)
#SBATCH --ntasks=1                   # Number of tasks (keep 1)
#SBATCH --cpus-per-task=2            # Number of CPU cores
#SBATCH -q normal                    # Queue name

#SBATCH -o log_files1/stdout_%x.out   # STDOUT log file
#SBATCH -e log_files1/stderr_%x.err   # STDERR log file

# Ensure robust bash behavior
set -e
set -o pipefail

# Set working directory
cd /users/romartinez/pshivhare/chipseq_buenrostro/all_bed_peak_regions_withouthsc

# Create log directory if it doesn't exist
mkdir -p log_files1

# Load necessary modules
module load BEDTools

./process_windows.sh 250
./process_windows.sh 450
./process_windows.sh 950
./process_windows.sh 500
./process_windows.sh 1000
 
