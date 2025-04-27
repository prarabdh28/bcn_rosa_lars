#!/bin/bash


module load BEDTools

# Input files
INPUT_FILE="selected_peaksrounded50_pu1_gmp.bed"  # Filtered signal BED file
PEAK_FILE="GSE231422_PU1_GMP_peaks.narrowPeak"  # Peak file with summit in column 10

# Output file
OUTPUT_FILE="feb23_pu1_gmp_peaksselected_wrounding_250bp_windows_with_sequence.bed"

# Temporary files
TEMP_BED="temp_bedfile.bed"
PEAK_ALIGNED_BED="peak_aligned_bedfile.bed"
NON_PEAK_BED="non_peak_bedfile.bed"
AGGREGATED_BED="aggregated_bedfile.bed"
FASTA_OUTPUT="temp_fasta_output.fa"
CLEAN_FASTA="clean_fasta_output.txt"
TEMP_SIGNAL_OUTPUT="temp_signal_output.txt"

# Path to reference genome
HG38_FASTA="./hg38.fa"

# Step 1: Process Peak File to Generate 250 bp Windows with Summit in the Middle Bin
awk '{
    summit = $2 + $10;  # Calculate absolute summit position
    summit_bin = int(summit / 50) * 50;  # Align to the nearest 50 bp bin
    start = summit_bin - 100;  # Extend to get 5 bins total (250 bp window)
    end = start + 250;
    if (start < 0) start = 0;
    print $1, start, end;
}' OFS="\t" "$PEAK_FILE" | sort -k1,1 -k2,2n | uniq > "$PEAK_ALIGNED_BED"

# Step 2: Create 250 bp Windows for Non-Peak Regions (Standard Method)
awk '{
    start = int($2 / 50) * 50;  # Align start to 50 bp bins
    start = start - (start % 250);  # Adjust to nearest 250 bp boundary
    end = start + 250;
    print $1, start, end;
}' OFS="\t" "$INPUT_FILE" | sort -k1,1 -k2,2n | uniq > "$TEMP_BED"

# Step 3: Remove Overlapping Windows from the Non-Peak List
bedtools intersect -v -a "$TEMP_BED" -b "$PEAK_ALIGNED_BED" > "$NON_PEAK_BED"

# Step 4: Combine Peak-Aligned Windows and Non-Peak Windows
cat "$PEAK_ALIGNED_BED" "$NON_PEAK_BED" | sort -k1,1 -k2,2n | uniq > "$AGGREGATED_BED"

# Step 5: Aggregate Signal Using Selected BED File Instead of BigWig
bedtools map -a "$AGGREGATED_BED" -b "$INPUT_FILE" -c 5 -o sum > "$TEMP_SIGNAL_OUTPUT"

# Format signal output into BED format
awk '{print $1, $2, $3, $4}' OFS="\t" "$TEMP_SIGNAL_OUTPUT" > "$AGGREGATED_BED"

# Step 6: Retrieve FASTA sequences
bedtools getfasta -fi "$HG38_FASTA" -bed "$AGGREGATED_BED" -fo "$FASTA_OUTPUT"

# Step 7: Clean FASTA headers and align sequences
awk 'NR % 2 == 0 {print}' "$FASTA_OUTPUT" > "$CLEAN_FASTA"

# Combine BED and sequences
paste <(cut -f1-3 "$AGGREGATED_BED") "$CLEAN_FASTA" > "$OUTPUT_FILE"

# Cleanup
#rm -f "$TEMP_BED" "$PEAK_ALIGNED_BED" "$NON_PEAK_BED" "$AGGREGATED_BED" "$FASTA_OUTPUT" "$CLEAN_FASTA" "$TEMP_SIGNAL_OUTPUT"

echo "Output written to $OUTPUT_FILE"
