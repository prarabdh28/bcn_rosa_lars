#!/bin/bash
# Usage: ./process_windows.sh <window_size>
# Example: ./process_windows.sh 450
#          ./process_windows.sh 950

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <window_size>"
    exit 1
fi

WINDOW_SIZE=$1
HG38_FASTA="./hg38.fa"  # Reference genome FASTA file (assumed to be indexed)

# Loop over all BED files starting with "modified"
for INPUT_FILE in modified*.bed; do
    echo "Processing $INPUT_FILE with window size ${WINDOW_SIZE}..."
    
    # Create output file name with a prefix indicating the window size
    OUTPUT_FILE="${WINDOW_SIZE}bp_${INPUT_FILE}"
    
    # Define temporary files
    TEMP_BED="temp_bedfile.bed"
    AGGREGATED_BED="aggregated_bedfile.bed"
    FASTA_OUTPUT="temp_fasta_output.fa"
    CLEAN_FASTA="clean_fasta_output.txt"

    # Step 1: Map each coordinate to its WINDOW_SIZE window.
    # This rounds the start down to the nearest multiple of WINDOW_SIZE.
    awk -v ws="$WINDOW_SIZE" '{
        start = int($2 / ws) * ws;
        end = start + ws;
        print $1, start, end, $4, $5;
    }' OFS="\t" "$INPUT_FILE" > "$TEMP_BED"

    # Step 2: Aggregate the signal for both columns (columns 4 and 5) for each window.
    awk '{
        key = $1"\t"$2"\t"$3;
        sig1[key] += $4;
        sig2[key] += $5;
    } END {
        for (k in sig1) {
            print k "\t" sig1[k] "\t" sig2[k];
        }
    }' "$TEMP_BED" | sort -k1,1 -k2,2n > "$AGGREGATED_BED"

    # Step 3: Retrieve FASTA sequences for each window using Bedtools.
    bedtools getfasta -fi "$HG38_FASTA" -bed "$AGGREGATED_BED" -fo "$FASTA_OUTPUT"

    # Step 4: Clean up FASTA headers.
    # Assumes that the FASTA file is in the usual two-line format (header then sequence).
    awk 'NR % 2 == 0 {print}' "$FASTA_OUTPUT" > "$CLEAN_FASTA"

    # Combine the aggregated BED file (columns 1-5) with the corresponding sequence.
    paste <(cut -f1-5 "$AGGREGATED_BED") "$CLEAN_FASTA" > "$OUTPUT_FILE"

    # Cleanup temporary files
    rm -f "$TEMP_BED" "$AGGREGATED_BED" "$FASTA_OUTPUT" "$CLEAN_FASTA"

    echo "Output written to $OUTPUT_FILE"
done
