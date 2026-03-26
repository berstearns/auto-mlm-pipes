#!/usr/bin/env bash
# Train all MLM encoder models on EFCAMDAT sequentially.
# Each run finishes before the next starts.
#
# Usage:
#   bash scripts/train-all-mlm.sh
#   bash scripts/train-all-mlm.sh --dry-run    # print commands without running
#   bash scripts/train-all-mlm.sh --resume      # skip configs whose output_dir/final/ exists

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"
CONFIG_DIR="configs/efcamdat/encoder"
LOG_DIR="logs/mlm"
DRY_RUN=false
RESUME=false

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --resume)  RESUME=true ;;
    esac
done

mkdir -p "$LOG_DIR"

CONFIGS=(
    "$CONFIG_DIR/albert-base-v2-mlm.yaml"
    "$CONFIG_DIR/albert-large-v2-mlm.yaml"
    "$CONFIG_DIR/albert-xlarge-v2-mlm.yaml"
    "$CONFIG_DIR/albert-xxlarge-v2-mlm.yaml"
    "$CONFIG_DIR/bert-base-mlm.yaml"
    "$CONFIG_DIR/bert-large-mlm.yaml"
    "$CONFIG_DIR/debertav3-xsmall-mlm.yaml"
    "$CONFIG_DIR/debertav3-small-mlm.yaml"
    "$CONFIG_DIR/debertav3-base-mlm.yaml"
    "$CONFIG_DIR/debertav3-large-mlm.yaml"
    "$CONFIG_DIR/modernbert-base-mlm.yaml"
    "$CONFIG_DIR/modernbert-large-mlm.yaml"
    "$CONFIG_DIR/nomic-bert-mlm.yaml"
    "$CONFIG_DIR/roberta-base-mlm.yaml"
    "$CONFIG_DIR/roberta-large-mlm.yaml"
)

TOTAL=${#CONFIGS[@]}
PASSED=0
FAILED=0
SKIPPED=0

echo "============================================"
echo " MLM Training Queue: $TOTAL models"
echo " Resume mode: $RESUME"
echo " Dry run: $DRY_RUN"
echo "============================================"
echo ""

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    name="$(basename "$cfg" .yaml)"
    idx=$((i + 1))
    logfile="$LOG_DIR/${name}.log"

    echo "[$idx/$TOTAL] $name"

    # --resume: skip if final model already exists
    if $RESUME; then
        # Peek into the config to guess the output dir pattern
        out_dir=$(grep -oP 'output_dir:\s*\K.*' "$cfg" 2>/dev/null || true)
        if [ -z "$out_dir" ]; then
            # Auto-generated output_dir — check for any matching dir
            match=$(find outputs/ -maxdepth 1 -type d -name "encoder-*${name%-mlm}*" 2>/dev/null | head -1)
            if [ -n "$match" ] && [ -d "$match/final" ]; then
                echo "  SKIP (found $match/final)"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
        elif [ -d "$out_dir/final" ]; then
            echo "  SKIP (found $out_dir/final)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
    fi

    cmd="$PYTHON -m pipelines.train_encoder --config $cfg"

    if $DRY_RUN; then
        echo "  [dry-run] $cmd"
        continue
    fi

    echo "  Config: $cfg"
    echo "  Log:    $logfile"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"

    if $cmd 2>&1 | tee "$logfile"; then
        echo "  DONE ($(date '+%H:%M:%S'))"
        PASSED=$((PASSED + 1))
    else
        echo "  FAILED (exit $?) — see $logfile"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

echo "============================================"
echo " Results: $PASSED passed, $FAILED failed, $SKIPPED skipped / $TOTAL total"
echo "============================================"

exit $FAILED
