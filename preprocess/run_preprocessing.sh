#!/bin/bash
# =============================================================================
# ViBES Motion Vector Preprocessing Script
# =============================================================================
# This script converts raw SMPL-X data to motion vector formats.
#
# Usage:
#   ./preprocess/run_preprocessing.sh [OPTIONS]
#
# Options:
#   --format FORMAT    Only process specific format: lower, upper_lower, genmo, all (default: all)
#   --dataset DATASET  Only process specific dataset: beat2, amass, all (default: all)
#   --dry-run          Print commands without executing
#   --max-files N      Limit files per directory (for testing)
#
# Examples:
#   ./preprocess/run_preprocessing.sh                          # Run everything
#   ./preprocess/run_preprocessing.sh --format lower           # Only Lower format
#   ./preprocess/run_preprocessing.sh --dataset beat2          # Only BEAT2
#   ./preprocess/run_preprocessing.sh --max-files 10 --dry-run # Test run
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration - Modify these paths as needed
# =============================================================================
VIBES_ROOT="/path/to/ViBES"
SMPLX_PATH="${VIBES_ROOT}/model_files/smplx_models"
INDEX_PATH="${VIBES_ROOT}/preprocess/index.csv"

# Dataset paths
BEAT2_ROOT="/path/to/BEAT2/beat_english_v2.0.0"
AMASS_ORIGINAL="/path/to/AMASS_original_smplx"
AMASS_PROCESSED="/path/to/AMASS"

# =============================================================================
# Parse arguments
# =============================================================================
FORMAT="all"
DATASET="all"
DRY_RUN=false
MAX_FILES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --max-files)
            MAX_FILES="--max_files $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Helper functions
# =============================================================================
run_cmd() {
    echo ""
    echo "=========================================="
    echo "Running: $1"
    echo "=========================================="
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] $2"
    else
        eval "$2"
    fi
}

# =============================================================================
# Check prerequisites
# =============================================================================
echo "=============================================="
echo "ViBES Motion Vector Preprocessing"
echo "=============================================="
echo "Format: $FORMAT"
echo "Dataset: $DATASET"
echo "Dry run: $DRY_RUN"
echo "Max files: ${MAX_FILES:-unlimited}"
echo ""

cd "$VIBES_ROOT"

# Check SMPLX model
if [ ! -d "$SMPLX_PATH" ]; then
    echo "ERROR: SMPLX model not found at $SMPLX_PATH"
    echo "Please run: ./scripts/build_resources.sh"
    exit 1
fi

# =============================================================================
# Lower Format (71D)
# =============================================================================
if [ "$FORMAT" = "lower" ] || [ "$FORMAT" = "all" ]; then

    # BEAT2 → Lower
    if [ "$DATASET" = "beat2" ] || [ "$DATASET" = "all" ]; then
        run_cmd "BEAT2 → Lower (71D)" \
            "python preprocess/dataset_process_beat2_lower.py \
                --input_dir ${BEAT2_ROOT} \
                --output_dir ${BEAT2_ROOT}/beat2_lower_25 \
                --smplx_path ${SMPLX_PATH} \
                --subdirs smplxflame_25 smplxflame_25_mirror \
                $MAX_FILES"
    fi

    # AMASS → Lower
    if [ "$DATASET" = "amass" ] || [ "$DATASET" = "all" ]; then
        run_cmd "AMASS → Lower (71D)" \
            "python preprocess/dataset_process_amass_lower.py \
                --dataset_path_original ${AMASS_ORIGINAL} \
                --dataset_path_processed ${AMASS_PROCESSED} \
                --smplx_path ${SMPLX_PATH} \
                --index_path ${INDEX_PATH} \
                --ex_fps 25"
    fi
fi

# =============================================================================
# Upper+Lower Format (149D)
# =============================================================================
if [ "$FORMAT" = "upper_lower" ] || [ "$FORMAT" = "all" ]; then

    # BEAT2 → Upper+Lower
    if [ "$DATASET" = "beat2" ] || [ "$DATASET" = "all" ]; then
        run_cmd "BEAT2 → Upper+Lower (149D)" \
            "python preprocess/dataset_process_beat2_parts.py \
                --input_dir ${BEAT2_ROOT} \
                --output_dir ${BEAT2_ROOT}/beat2_parts_25 \
                --smplx_path ${SMPLX_PATH} \
                --subdirs smplxflame_25 smplxflame_25_mirror \
                $MAX_FILES"
    fi

    # AMASS → Upper+Lower
    if [ "$DATASET" = "amass" ] || [ "$DATASET" = "all" ]; then
        run_cmd "AMASS → Upper+Lower (149D)" \
            "python preprocess/dataset_process_amass_parts.py \
                --dataset_path_original ${AMASS_ORIGINAL} \
                --dataset_path_processed ${AMASS_PROCESSED} \
                --smplx_path ${SMPLX_PATH} \
                --index_path ${INDEX_PATH} \
                --ex_fps 25"
    fi
fi

# =============================================================================
# GENMO Format (145D) - Note: BEAT2 script has hardcoded paths
# =============================================================================
if [ "$FORMAT" = "genmo" ] || [ "$FORMAT" = "all" ]; then

    # BEAT2 → GENMO (hardcoded paths in script)
    if [ "$DATASET" = "beat2" ] || [ "$DATASET" = "all" ]; then
        run_cmd "BEAT2 → GENMO (145D)" \
            "python preprocess/dataset_process_beat2_genmo.py"
    fi

    # AMASS → GENMO
    if [ "$DATASET" = "amass" ] || [ "$DATASET" = "all" ]; then
        run_cmd "AMASS → GENMO (145D)" \
            "python preprocess/dataset_process_amass_genmo.py \
                --dataset_path_original ${AMASS_ORIGINAL} \
                --dataset_path_processed ${AMASS_PROCESSED} \
                --index_path ${INDEX_PATH} \
                --ex_fps 25"
    fi
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "Preprocessing Complete!"
echo "=============================================="
echo ""
echo "Output directories:"
if [ "$FORMAT" = "lower" ] || [ "$FORMAT" = "all" ]; then
    echo "  Lower:       ${BEAT2_ROOT}/beat2_lower_25"
    echo "               ${AMASS_PROCESSED}/amass_lower_25"
fi
if [ "$FORMAT" = "upper_lower" ] || [ "$FORMAT" = "all" ]; then
    echo "  Upper+Lower: ${BEAT2_ROOT}/beat2_parts_25"
    echo "               ${AMASS_PROCESSED}/amass_parts_25"
fi
if [ "$FORMAT" = "genmo" ] || [ "$FORMAT" = "all" ]; then
    echo "  GENMO:       ${BEAT2_ROOT}/beat2_genmo_25"
    echo "               ${AMASS_PROCESSED}/amass_genmo_25"
fi
