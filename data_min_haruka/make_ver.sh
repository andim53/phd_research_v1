#!/bin/bash

ENV_NAME="mp_env"
OUTPUT_FILE="environment_versions.md"

# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate environment
conda activate "$ENV_NAME" || {
    echo "Failed to activate environment: $ENV_NAME"
    exit 1
}

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')

ASE_VERSION=$(pip show ase 2>/dev/null | awk -F': ' '/^Version:/ {print $2}')
ASE_VERSION=${ASE_VERSION:-"Not installed"}

MP_API_VERSION=$(pip show mp-api 2>/dev/null | awk -F': ' '/^Version:/ {print $2}')
MP_API_VERSION=${MP_API_VERSION:-"Not installed"}

cat > "$OUTPUT_FILE" <<EOF
# Environment Version Report

**Conda Environment:** \`$ENV_NAME\`

| Package | Version |
|----------|----------|
| Python | $PYTHON_VERSION |
| ASE | $ASE_VERSION |
| mp-api | $MP_API_VERSION |

Generated on: $(date)
EOF

echo "Saved report to $OUTPUT_FILE"
