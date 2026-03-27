#!/bin/bash
#===============================================================================
# run-multilingual-mlm-efcamdat.sh — Train 4 multilingual MLM encoders on EFCAMDAT
#===============================================================================
# Trains: infoxlm-large, multilingual-e5-large, rembert, xlm-roberta-large
# Includes OOM watchdog: monitors GPU memory during training and halves
# batch size on OOM, then retries automatically.
#
# Usage:
#   bash /workspace/auto-mlm-pipes/deploy/run-multilingual-mlm-efcamdat.sh
#   bash /workspace/auto-mlm-pipes/deploy/run-multilingual-mlm-efcamdat.sh --skip-fetch
#   bash /workspace/auto-mlm-pipes/deploy/run-multilingual-mlm-efcamdat.sh --resume
#===============================================================================

set -uo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'

REPO_DIR="/workspace/auto-mlm-pipes"
DATA_DIR="/workspace/data"
PYENV_ROOT="/root/.pyenv"
CONFIG_DIR="$REPO_DIR/configs/efcamdat/encoder"
LOG_DIR="$REPO_DIR/logs/mlm"

GDRIVE_DATA="i:phd-experimental-data/cefr-classification/data/splits"

SKIP_FETCH=""
RESUME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-fetch)    SKIP_FETCH="1"; shift ;;
        --resume)        RESUME="1"; shift ;;
        -h|--help)
            echo "Usage: $0 [--skip-fetch] [--resume]"
            exit 0
            ;;
        *) shift ;;
    esac
done

export PYENV_ROOT
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
eval "$(pyenv init -)" 2>/dev/null || true

echo -e "${CYAN}=== auto-mlm-pipes: Multilingual MLM Models (4) ===${NC}"
echo "  Repo:    $REPO_DIR"
echo "  Data:    $DATA_DIR"
echo "  Python:  $(python3 --version)"
echo "  GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Start:   $(date)"
echo ""

if [[ ! -d "$REPO_DIR" ]]; then
    echo -e "${RED}FATAL: Repo not found at $REPO_DIR. Run project-setup-mlm first.${NC}"
    exit 1
fi

# Pull latest code
echo -e "${CYAN}[0/3] Pulling latest code...${NC}"
git -C "$REPO_DIR" pull --ff-only 2>&1 | tail -3
echo "  HEAD: $(git -C "$REPO_DIR" log --oneline -1)"
echo ""

#===============================================================================
# Phase 1: Fetch data
#===============================================================================
DATA_FILE="$DATA_DIR/norm-EFCAMDAT-ALL-CONCAT.csv"

if [[ -z "$SKIP_FETCH" ]]; then
    echo -e "${CYAN}[1/3] Fetching EFCAMDAT data...${NC}"

    if ! command -v rclone &>/dev/null; then
        echo -e "${RED}FATAL: rclone not installed.${NC}"
        exit 1
    fi

    mkdir -p "$DATA_DIR"
    if [[ -f "$DATA_FILE" ]]; then
        echo -e "  ${GREEN}✓${NC} norm-EFCAMDAT-ALL-CONCAT.csv (cached)"
    else
        rclone copy "$GDRIVE_DATA/norm-EFCAMDAT-ALL-CONCAT.csv" "$DATA_DIR/" 2>&1
        echo -e "  ${GREEN}✓${NC} norm-EFCAMDAT-ALL-CONCAT.csv (downloaded)"
    fi
    echo ""
else
    echo -e "${YELLOW}[1/3] Skipping fetch (--skip-fetch)${NC}"
    echo ""
fi

#===============================================================================
# Phase 2: Train with OOM watchdog
#===============================================================================
echo -e "${CYAN}[2/3] Running multilingual MLM configs (4 models)...${NC}"
echo "  Start: $(date)"
echo ""

cd "$REPO_DIR"
mkdir -p "$LOG_DIR"

CONFIGS=(
    "$CONFIG_DIR/infoxlm-large-mlm.yaml"
    "$CONFIG_DIR/multilingual-e5-large-mlm.yaml"
    "$CONFIG_DIR/rembert-mlm.yaml"
    "$CONFIG_DIR/xlmr-large-mlm.yaml"
)

TOTAL=${#CONFIGS[@]}
PASSED=0
FAILED=0
SKIPPED=0

#-------------------------------------------------------------------------------
# OOM watchdog: run training, if OOM detected halve batch size and retry
#-------------------------------------------------------------------------------
train_with_watchdog() {
    local cfg="$1"
    local name="$2"
    local logfile="$3"
    local max_retries=3
    local attempt=0
    local bs_override=""
    local ga_override=""

    while [[ $attempt -lt $max_retries ]]; do
        attempt=$((attempt + 1))

        # Build command
        local cmd="python -m pipelines.train_encoder --config $cfg"
        if [[ -n "${TRAIN_FILE_OVERRIDE:-}" ]]; then
            cmd="$cmd --train_file $TRAIN_FILE_OVERRIDE"
        fi
        if [[ -n "$bs_override" ]]; then
            cmd="$cmd --per_device_train_batch_size $bs_override --per_device_eval_batch_size $bs_override"
        fi
        if [[ -n "$ga_override" ]]; then
            cmd="$cmd --gradient_accumulation_steps $ga_override"
        fi

        echo -e "  ${CYAN}attempt $attempt/$max_retries${NC}  bs=${bs_override:-config} ga=${ga_override:-config}"
        echo "  cmd: $cmd"

        # Clear GPU cache before each attempt
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

        # Run training, capture output for OOM detection
        if $cmd 2>&1 | tee "$logfile"; then
            return 0  # success
        fi

        # Check if OOM
        if grep -qi "CUDA out of memory\|OutOfMemoryError\|CUDA error: out of memory\|torch.OutOfMemoryError" "$logfile"; then
            echo -e "  ${YELLOW}OOM detected!${NC} Halving batch size..."

            # Read current batch size (from override or config)
            local current_bs="${bs_override:-$(grep -oP 'per_device_train_batch_size:\s*\K\d+' "$cfg")}"
            local current_ga="${ga_override:-$(grep -oP 'gradient_accumulation_steps:\s*\K\d+' "$cfg")}"

            if [[ "$current_bs" -le 1 ]]; then
                echo -e "  ${RED}batch size already 1, cannot reduce further${NC}"
                return 1
            fi

            bs_override=$((current_bs / 2))
            ga_override=$((current_ga * 2))
            echo -e "  ${YELLOW}new batch size: $bs_override, grad accum: $ga_override${NC}"

            # Clean up partial output dir from failed run
            local last_output
            last_output=$(grep -oP 'Output: \K.*' "$logfile" | tail -1)
            if [[ -n "$last_output" && -d "$last_output" ]]; then
                echo -e "  cleaning up partial output: $last_output"
                rm -rf "$last_output"
            fi
        else
            echo -e "  ${RED}non-OOM failure${NC}"
            return 1
        fi
    done

    echo -e "  ${RED}exhausted $max_retries retries${NC}"
    return 1
}

# Override train_file to point to remote data path
export TRAIN_FILE_OVERRIDE="$DATA_FILE"

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    name="$(basename "$cfg" .yaml)"
    idx=$((i + 1))
    logfile="$LOG_DIR/${name}.log"

    echo "============================================"
    echo "[$idx/$TOTAL] $name"
    echo "============================================"

    # --resume: skip if final model already exists
    if [[ -n "$RESUME" ]]; then
        out_dir=$(grep -oP 'output_dir:\s*\K.*' "$cfg" 2>/dev/null || true)
        if [[ -z "$out_dir" ]]; then
            match=$(find outputs/ -maxdepth 1 -type d -name "encoder-*${name%-mlm}*" 2>/dev/null | head -1)
            if [[ -n "$match" ]] && [[ -d "$match/final" ]]; then
                echo -e "  ${GREEN}SKIP${NC} (found $match/final)"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
        elif [[ -d "$out_dir/final" ]]; then
            echo -e "  ${GREEN}SKIP${NC} (found $out_dir/final)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
    fi

    echo "  Config: $cfg"
    echo "  Log:    $logfile"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"

    if train_with_watchdog "$cfg" "$name" "$logfile"; then
        echo -e "  ${GREEN}DONE${NC} ($(date '+%H:%M:%S'))"
        PASSED=$((PASSED + 1))
    else
        echo -e "  ${RED}FAILED${NC} — see $logfile"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

echo ""
echo "============================================"
echo " Results: $PASSED passed, $FAILED failed, $SKIPPED skipped / $TOTAL total"
echo "============================================"
echo -e "${GREEN}=== All done. $(date) ===${NC}"

exit $FAILED
