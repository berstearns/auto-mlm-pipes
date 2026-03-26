#!/bin/bash
#===============================================================================
# run-all-mlm-efcamdat.sh — Train all 15 MLM encoders on EFCAMDAT
#===============================================================================
# Runs ON the remote server (via tmux). Fetches EFCAMDAT data from GDrive,
# then trains all MLM configs sequentially.
#
# Usage (tmux on remote):
#   ssh $SSH_URL "tmux new-window -t ssh_tmux -n mlm-train \
#     'export PYENV_ROOT=/root/.pyenv && \
#      export PATH=\$PYENV_ROOT/bin:\$PYENV_ROOT/shims:\$PATH && \
#      eval \"\$(pyenv init -)\" && \
#      bash /workspace/auto-mlm-pipes/deploy/run-all-mlm-efcamdat.sh; bash'"
#
# Prerequisites:
#   - project-setup-mlm already run
#   - rclone configured with remote 'i:'
#===============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'

REPO_DIR="/workspace/auto-mlm-pipes"
DATA_DIR="/workspace/data"
PYENV_ROOT="/root/.pyenv"

GDRIVE_DATA="i:/_p/artificial-learners/data/splits"
GDRIVE_RESULTS="i:/_p/artificial-learners/mlm-training/outputs"

SKIP_FETCH=""
SYNC_RESULTS=""
RESUME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-fetch)    SKIP_FETCH="1"; shift ;;
        --sync-results)  SYNC_RESULTS="1"; shift ;;
        --resume)        RESUME="--resume"; shift ;;
        -h|--help)
            echo "Usage: $0 [--skip-fetch] [--sync-results] [--resume]"
            exit 0
            ;;
        *) shift ;;
    esac
done

export PYENV_ROOT
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
eval "$(pyenv init -)" 2>/dev/null || true

echo -e "${CYAN}=== auto-mlm-pipes: EFCAMDAT All MLM Models ===${NC}"
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
        echo -e "${RED}FATAL: rclone not installed. Run 'copy-rclone' via orchestrator first.${NC}"
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
# Phase 2: Run all MLM configs
#===============================================================================
echo -e "${CYAN}[2/3] Running all MLM configs (15 models)...${NC}"
echo "  Start: $(date)"
echo ""

cd "$REPO_DIR"

# Override train_file to point to remote data path
export TRAIN_FILE_OVERRIDE="$DATA_FILE"

bash scripts/train-all-mlm.sh $RESUME \
    2>&1 | tee /workspace/mlm-training-all.log

#===============================================================================
# Phase 3: Sync results
#===============================================================================
if [[ -n "$SYNC_RESULTS" ]]; then
    echo ""
    echo -e "${CYAN}[3/3] Syncing results to GDrive...${NC}"
    if [[ -d "$REPO_DIR/outputs" ]]; then
        rclone copy "$REPO_DIR/outputs/" "$GDRIVE_RESULTS/" 2>&1
        echo -e "  ${GREEN}✓${NC} Synced to $GDRIVE_RESULTS"
    fi
fi

echo ""
echo -e "${GREEN}=== All done. $(date) ===${NC}"
echo "  Log: /workspace/mlm-training-all.log"
