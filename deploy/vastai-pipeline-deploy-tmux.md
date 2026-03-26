# auto-mlm-pipes: vastai pipeline deploy via tmux

Deploy and run MLM encoder pre-training on EFCAMDAT on a vastai GPU server.

## Prerequisites

- vastai instance running with rclone configured (`i:` remote)
- EFCAMDAT data at `i:phd-experimental-data/cefr-classification/data/splits/norm-EFCAMDAT-ALL-CONCAT.csv`

## Step 0: Get SSH URL

```bash
cd ~/p/all-my-tiny-projects/vastai
SSH_URL=$(./vastai-ssh get-url)
```

## Step 1: Setup (first time only)

```bash
./orchestrator.sh --mode ssh --ssh-url "$SSH_URL" project-setup-mlm
```

## Step 2: Smoke test

Quick validation: bert-base MLM, 1 epoch, 20 dummy sentences. Minimal GPU usage.

```bash
ssh $SSH_URL "tmux new-window -t ssh_tmux -n mlm-smoke \
  'export PYENV_ROOT=/root/.pyenv && \
   export PATH=\$PYENV_ROOT/bin:\$PYENV_ROOT/shims:\$PATH && \
   eval \"\$(pyenv init -)\" && \
   bash /workspace/auto-mlm-pipes/deploy/run-smoke-test.sh; bash'"
```

Check:
```bash
ssh $SSH_URL "tmux capture-pane -t ssh_tmux:mlm-smoke -p -S -20"
```

## Step 3: Run all 15 MLM models on EFCAMDAT

**Always pull before launch** (bash reads script into memory before git pull):

```bash
# 1. Push local changes
cd ~/p/research-sketches/auto-mlm-pipes
git add -A && git commit -m "update" && git push origin main

# 2. Pull on remote
ssh $SSH_URL "cd /workspace/auto-mlm-pipes && \
  export PYENV_ROOT=/root/.pyenv && \
  export PATH=\$PYENV_ROOT/bin:\$PYENV_ROOT/shims:\$PATH && \
  eval \"\$(pyenv init -)\" && \
  git pull --ff-only"

# 3. Launch in tmux
ssh $SSH_URL "tmux new-window -t ssh_tmux -n mlm-train \
  'export PYENV_ROOT=/root/.pyenv && \
   export PATH=\$PYENV_ROOT/bin:\$PYEX_ROOT/shims:\$PATH && \
   eval \"\$(pyenv init -)\" && \
   bash /workspace/auto-mlm-pipes/deploy/run-all-mlm-efcamdat.sh --sync-results; bash'"
```

With resume (skip models that already have `output_dir/final/`):
```bash
bash /workspace/auto-mlm-pipes/deploy/run-all-mlm-efcamdat.sh --resume --sync-results
```

## Monitoring

```bash
ssh $SSH_URL "tmux list-windows -t ssh_tmux"
ssh $SSH_URL "tmux capture-pane -t ssh_tmux:mlm-train -p -S -30"
ssh $SSH_URL "tail -50 /workspace/mlm-training-all.log"
ssh $SSH_URL -t "tmux attach -t ssh_tmux:mlm-train"
```

## Models trained (15 MLM configs)

| Family | Models |
|--------|--------|
| ALBERT | base-v2, large-v2, xlarge-v2, xxlarge-v2 |
| BERT | base-uncased, large-uncased |
| DeBERTaV3 | xsmall, small, base, large |
| ModernBERT | base, large |
| NomicBERT | nomic-embed-text-v1 |
| RoBERTa | base, large |

## Files

| File | Purpose |
|------|---------|
| `deploy/project-setup-remote.sh` | Pyenv + deps setup (run once) |
| `deploy/run-smoke-test.sh` | Quick GPU validation |
| `deploy/run-all-mlm-efcamdat.sh` | Full 15-model MLM training |
| `configs/dummies/smoke-test.yaml` | Smoke config (1 epoch, dummy data) |
| `scripts/train-all-mlm.sh` | Sequential training orchestrator |

## Disk space

Check before launching (~1GB per model output, ~15GB total):
```bash
ssh $SSH_URL "df -h /workspace"
```
