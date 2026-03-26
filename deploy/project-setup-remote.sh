#!/bin/bash
#===============================================================================
# project-setup-remote.sh — Setup auto-mlm-pipes on a remote GPU server
#===============================================================================
# Runs ON the remote via SSH pipe (orchestrator handles the connection).
# Installs pyenv + Python 3.10.18, clones repo, installs deps.
#
# Usage (via orchestrator):
#   ./orchestrator.sh --mode ssh --ssh-url URL project-setup-mlm
#
# Usage (direct SSH pipe):
#   ssh -p PORT root@HOST "bash -s" < deploy/project-setup-remote.sh
#===============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; NC='\033[0m'

GITHUB_REPO="https://github.com/berstearns/auto-mlm-pipes.git"
REPO_DIR="/workspace/auto-mlm-pipes"
PYENV_ROOT="/root/.pyenv"
PYTHON_VERSION="3.10.18"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)    shift ;;
        --repo-dir) REPO_DIR="$2"; shift 2 ;;
        *) shift ;;
    esac
done

phase() { echo ""; echo -e "${CYAN}[$1/5] $2${NC}"; echo "---"; }

echo -e "${CYAN}=== auto-mlm-pipes: Remote Setup ===${NC}"
echo "  Repo:    $GITHUB_REPO"
echo "  Dir:     $REPO_DIR"
echo "  Python:  $PYTHON_VERSION (via pyenv)"
echo "  GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# ============================================================
phase 1 "System packages"
# ============================================================
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq \
    git build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev \
    wget curl > /dev/null 2>&1
echo -e "${GREEN}System packages installed${NC}"

# ============================================================
phase 2 "Install Python $PYTHON_VERSION via pyenv"
# ============================================================
export PYENV_ROOT
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
eval "$(pyenv init -)" 2>/dev/null || true

if ! command -v pyenv &>/dev/null; then
    echo "pyenv not found, installing..."
    git clone --depth 1 https://github.com/pyenv/pyenv.git "$PYENV_ROOT"
    export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
    eval "$(pyenv init -)"
fi

if pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
    echo "Python $PYTHON_VERSION already installed"
else
    echo "Installing Python $PYTHON_VERSION..."
    pyenv install "$PYTHON_VERSION"
fi

pyenv global "$PYTHON_VERSION"
echo -e "${GREEN}Python $(python3 --version) via pyenv${NC}"

# ============================================================
phase 3 "Clone / update repo"
# ============================================================
if [[ -d "$REPO_DIR/.git" ]]; then
    echo "Updating existing repo..."
    git -C "$REPO_DIR" fetch --all 2>&1 | tail -2
    git -C "$REPO_DIR" reset --hard origin/main 2>&1 | tail -2
    echo -e "${GREEN}Repo updated${NC}"
else
    echo "Cloning from $GITHUB_REPO..."
    rm -rf "$REPO_DIR"
    git clone "$GITHUB_REPO" "$REPO_DIR" 2>&1 | tail -3
    echo -e "${GREEN}Repo cloned${NC}"
fi
echo "  HEAD: $(git -C "$REPO_DIR" log --oneline -1)"

# ============================================================
phase 4 "Python dependencies"
# ============================================================
pip install --upgrade pip 2>&1 | tail -1
pip install -e "$REPO_DIR" 2>&1 | tail -5
echo -e "${GREEN}Deps installed${NC}"

# ============================================================
phase 5 "Verify imports"
# ============================================================
python3 -c "
import torch
print(f'  torch {torch.__version__}  (CUDA: {torch.cuda.is_available()})')
import transformers; print(f'  transformers {transformers.__version__}')
import datasets; print(f'  datasets {datasets.__version__}')
import accelerate; print(f'  accelerate {accelerate.__version__}')
from pipelines.train_encoder import main; print('  pipelines.train_encoder OK')
from pipelines.config import EncoderConfig; print('  pipelines.config OK')
from pipelines.data_utils import load_data; print('  pipelines.data_utils OK')
"
echo -e "${GREEN}All imports verified${NC}"

echo ""
echo -e "${GREEN}=== Setup Complete ===${NC}"
echo "  Repo:   $REPO_DIR"
echo "  Python: $(python3 --version)"
echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
