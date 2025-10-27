#!/usr/bin/env bash   

# Folder containing the .flv files
INPUT_DIR="/mnt/smirk/crema-d-mirror/VideoFlash"
# Output directory
OUTPUT_DIR="/mnt/smirk/results"
# Model checkpoint
CHECKPOINT="/mnt/smirk/pretrained_models/SMIRK_em1.pt"

find "$INPUT_DIR" -maxdepth 1 -type f -name '*.flv' -print0 | \
  shuf -z | \
  while IFS= read -r -d '' file; do
      echo "Processing $file..."
      python demo_video.py \
          --input_path "$file" \
          --out_path "$OUTPUT_DIR" \
          --checkpoint "$CHECKPOINT" \
          --crop
done
