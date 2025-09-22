#!/usr/bin/env bash

# ------------------------------------------------------------
# Usage: ./concat_subfiles.sh <parent_dir> <output_file>
#
# Example:
#   ./concat_subfiles.sh /home/user/documents all_texts.txt
# ------------------------------------------------------------

set -euo pipefail   # safer scripting

# ---------- Argument handling ----------
if [[ $# -ne 2 ]]; then
    echo "Error: Exactly two arguments required."
    echo "Usage: $0 <parent_directory> <output_file>"
    exit 1
fi

PARENT_DIR=$1
OUTPUT_FILE=$2

# Verify the parent directory exists and is readable
if [[ ! -d "$PARENT_DIR" ]]; then
    echo "Error: Directory '$PARENT_DIR' does not exist."
    exit 1
fi

if [[ ! -r "$PARENT_DIR" ]]; then
    echo "Error: Directory '$PARENT_DIR' is not readable."
    exit 1
fi

# Optional: clear previous output file (or you could append)
> "$OUTPUT_FILE"

# ---------- Core logic ----------
# Find all regular files under the parent directory (excluding the parent itself)
# -type f   : regular files only
# -print0   : null‑terminated names to safely handle spaces/newlines
find "$PARENT_DIR" -type f -print0 |
while IFS= read -r -d '' file; do
    # Append a header (optional) so you know which file contributed which part
    echo "===== $file =====" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"   # ensure a newline between files
done

echo "All files under '$PARENT_DIR' have been concatenated into '$OUTPUT_FILE'."
