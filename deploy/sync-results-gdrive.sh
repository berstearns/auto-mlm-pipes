#!/bin/bash
#===============================================================================
# sync-results-gdrive.sh — Periodic rclone sync of MLM training outputs
#===============================================================================
# Runs in its own tmux window. Every 15 minutes:
#   1. Scans outputs/ for completed models (has final/ + 10 checkpoint-* dirs)
#   2. Syncs each to i:/_p/artificial-learners/models/encoders/mlm/{family}/{size}/
#   3. Verifies upload, removes local copy only if confirmed
#
# Usage (tmux):
#   ssh $SSH_URL "tmux new-window -t ssh_tmux -n mlm-sync \
#     'bash /workspace/auto-mlm-pipes/deploy/sync-results-gdrive.sh; bash'"
#
# Ctrl+C to stop. Re-run to restart.
#===============================================================================

set -uo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'

REPO_DIR="/workspace/auto-mlm-pipes"
OUTPUTS_DIR="$REPO_DIR/outputs"
GDRIVE_BASE="i:/_p/artificial-learners/models/encoders/mlm"
NUM_EPOCHS=10
INTERVAL=900  # 15 minutes

#===============================================================================
# Model slug -> GDrive family/size mapping
#===============================================================================
declare -A MODEL_MAP=(
    ["albert-base-v2"]="albert/base-v2"
    ["albert-large-v2"]="albert/large-v2"
    ["albert-xlarge-v2"]="albert/xlarge-v2"
    ["albert-xxlarge-v2"]="albert/xxlarge-v2"
    ["bert-base-uncased"]="bert/base"
    ["bert-large-uncased"]="bert/large"
    ["deberta-v3-xsmall"]="debertav3/xsmall"
    ["deberta-v3-small"]="debertav3/small"
    ["deberta-v3-base"]="debertav3/base"
    ["deberta-v3-large"]="debertav3/large"
    ["ModernBERT-base"]="modernbert/base"
    ["ModernBERT-large"]="modernbert/large"
    ["nomic-bert-2048"]="nomic-bert/base"
    ["roberta-base"]="roberta/base"
    ["roberta-large"]="roberta/large"
)

get_gdrive_path() {
    local dir_name="$1"
    # Strip "encoder-" prefix and timestamp suffix (e.g. -20260326_221936)
    local slug="${dir_name#encoder-}"
    slug="${slug%-[0-9]*_[0-9]*}"

    local mapped="${MODEL_MAP[$slug]:-}"
    if [[ -n "$mapped" ]]; then
        echo "$GDRIVE_BASE/$mapped"
    fi
}

sync_one() {
    local out_dir="$1"
    local dir_name
    dir_name="$(basename "$out_dir")"

    local gdrive_target
    gdrive_target="$(get_gdrive_path "$dir_name")"

    if [[ -z "$gdrive_target" ]]; then
        echo -e "  ${YELLOW}SKIP${NC} $dir_name — no GDrive mapping found"
        return 1
    fi

    # Count checkpoint dirs
    local ckpt_count
    ckpt_count=$(find "$out_dir" -maxdepth 1 -type d -name "checkpoint-*" | wc -l)

    # Check completeness: need final/ + all epochs
    if [[ ! -d "$out_dir/final" ]]; then
        echo -e "  ${YELLOW}WAIT${NC} $dir_name — training not finished (no final/)"
        return 1
    fi

    if [[ "$ckpt_count" -lt "$NUM_EPOCHS" ]]; then
        echo -e "  ${YELLOW}WAIT${NC} $dir_name — only $ckpt_count/$NUM_EPOCHS checkpoints"
        return 1
    fi

    echo -e "  ${CYAN}SYNC${NC} $dir_name -> $gdrive_target"
    echo "         checkpoints: $ckpt_count/$NUM_EPOCHS + final/"

    # Upload
    if ! rclone copy "$out_dir/" "$gdrive_target/" --verbose 2>&1 | tail -5; then
        echo -e "  ${RED}FAIL${NC} rclone copy failed for $dir_name"
        return 1
    fi

    # Verify: compare local vs remote file counts
    local local_count remote_count
    local_count=$(find "$out_dir" -type f | wc -l)
    remote_count=$(rclone ls "$gdrive_target/" 2>/dev/null | wc -l)

    if [[ "$remote_count" -ge "$local_count" ]]; then
        echo -e "  ${GREEN}OK${NC}   verified ($remote_count/$local_count files) — removing local copy"
        rm -rf "$out_dir"
    else
        echo -e "  ${RED}MISMATCH${NC} local=$local_count remote=$remote_count — keeping local copy"
        echo "         manual check needed: rclone check '$out_dir/' '$gdrive_target/'"
    fi
}

echo -e "${CYAN}=== MLM Results GDrive Sync ===${NC}"
echo "  Outputs:  $OUTPUTS_DIR"
echo "  Target:   $GDRIVE_BASE/{family}/{size}/"
echo "  Interval: ${INTERVAL}s ($(( INTERVAL / 60 ))min)"
echo "  Expects:  $NUM_EPOCHS checkpoints + final/ per model"
echo ""
echo "  Ctrl+C to stop. Re-run to restart."
echo ""

while true; do
    echo -e "${CYAN}--- sync pass: $(date) ---${NC}"

    if [[ ! -d "$OUTPUTS_DIR" ]]; then
        echo "  No outputs/ dir yet, waiting..."
    else
        found=0
        for out_dir in "$OUTPUTS_DIR"/encoder-*; do
            [[ -d "$out_dir" ]] || continue
            found=1
            sync_one "$out_dir"
        done

        if [[ "$found" -eq 0 ]]; then
            echo "  No encoder-* output dirs found"
        fi
    fi

    echo ""
    echo -e "  sleeping ${INTERVAL}s (next: $(date -d "+${INTERVAL} seconds" '+%H:%M:%S' 2>/dev/null || date -v+${INTERVAL}S '+%H:%M:%S' 2>/dev/null || echo '?'))..."
    echo ""
    sleep "$INTERVAL"
done
