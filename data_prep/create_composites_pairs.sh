#!/bin/bash --login
#SBATCH -J bigwig_to_bed_intersect  # Job name
#SBATCH -t 2:00:00                  # Time limit (hh:mm:ss)
#SBATCH --mem=40G                   # Memory allocation
#SBATCH --nodes=1                    # Number of nodes (keep 1)
#SBATCH --ntasks=1                   # Number of tasks (keep 1)
#SBATCH --cpus-per-task=2            # Number of CPU cores
#SBATCH -q normal                    # Queue name

#SBATCH -o log_files/stdout_%x.out   # STDOUT log file
#SBATCH -e log_files/stderr_%x.err   # STDERR log file

# Ensure robust bash behavior
set -e
set -o pipefail

# Set working directory
cd /users/romartinez/pshivhare/chipseq_buenrostro/all_bed_peak_regions_withouthsc

# Create log directory if it doesn't exist
mkdir -p log_files

# Load necessary modules
module load BEDTools

# Define cell states
cell_states=("CMP" "GMP" "MEP")

# Define transcription factors
tf_list=("FLI1" "PU1" "GATA2" "RUNX1" "LYL1" "TAL1")

# Create all TF pairs
tf_pairs=()
for ((i=0; i<${#tf_list[@]}; i++)); do
    for ((j=i+1; j<${#tf_list[@]}; j++)); do
        tf_pairs+=("${tf_list[i]}_${tf_list[j]}")
    done
done

# Loop through cell states and TF pairs
for cell in "${cell_states[@]}"; do
    for pair in "${tf_pairs[@]}"; do
        # Extract individual TFs
        tf1=$(echo "$pair" | cut -d'_' -f1)
        tf2=$(echo "$pair" | cut -d'_' -f2)

        # Define input file names
        file1="split50size_normchroms_GSE231422_Coverage_${tf1}_${cell}_merge_clip_50peaks.bed"
        file2="split50size_normchroms_GSE231422_Coverage_${tf2}_${cell}_merge_clip_50peaks.bed"

        # Define output file name
        output_file="composite_${tf1}_${tf2}_${cell}.bed"

        # Check if both files exist
        if [[ -f "$file1" && -f "$file2" ]]; then
            echo "Processing intersection for $tf1 and $tf2 in $cell..."

            # Ensure files are sorted correctly before bedtools
            sort -k1,1 -k2,2n "$file1" -o "$file1"
            sort -k1,1 -k2,2n "$file2" -o "$file2"

            # Perform bedtools intersect with TAB-delimited formatting and integer coordinates
            bedtools intersect -a "$file1" -b "$file2" -wa -wb | awk -v OFS="\t" '{
                print $1, int($2), int($3), $4, $5, $9, $10;
            }' > "$output_file"

            echo "Created: $output_file"
        else
            echo "Skipping: Missing file(s) for $tf1 and $tf2 in $cell."
        fi
    done
done

echo "All composite files created!"
