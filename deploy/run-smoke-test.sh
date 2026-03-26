#!/bin/bash
#===============================================================================
# run-smoke-test.sh — Minimal MLM smoke test on remote GPU
#===============================================================================
# Trains bert-base MLM for 1 epoch on 20 dummy sentences.
# Uses minimal GPU memory. Output: outputs/smoke-test-mlm/
#===============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; NC='\033[0m'

REPO_DIR="/workspace/auto-mlm-pipes"
PYENV_ROOT="/root/.pyenv"

export PYENV_ROOT
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
eval "$(pyenv init -)" 2>/dev/null || true

echo -e "${CYAN}=== auto-mlm-pipes: Smoke Test ===${NC}"
echo "  Python: $(python3 --version)"
echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

cd "$REPO_DIR"
rm -rf outputs/smoke-test-mlm

echo -e "${CYAN}[1/2] Running MLM smoke test (bert-base, 1 epoch, dummy data)...${NC}"
if python3 -m pipelines.train_encoder --config configs/dummies/smoke-test.yaml 2>&1; then
    echo -e "${GREEN}Pipeline completed${NC}"
else
    echo -e "${RED}Pipeline FAILED${NC}"
    exit 1
fi

echo ""
echo -e "${CYAN}[2/2] Verifying outputs...${NC}"
if [[ -d outputs/smoke-test-mlm ]]; then
    echo -e "  ${GREEN}[PASS]${NC} Output dir exists"
    ls -lh outputs/smoke-test-mlm/ 2>/dev/null
else
    echo -e "  ${RED}[FAIL]${NC} Output dir missing"
    exit 1
fi

echo ""
echo -e "${GREEN}Smoke test OK${NC}"
