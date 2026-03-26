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

# 3. Launch training in tmux
ssh $SSH_URL "tmux new-window -t ssh_tmux -n mlm-train \
  'export PYENV_ROOT=/root/.pyenv && \
   export PATH=\$PYENV_ROOT/bin:\$PYENV_ROOT/shims:\$PATH && \
   eval \"\$(pyenv init -)\" && \
   bash /workspace/auto-mlm-pipes/deploy/run-all-mlm-efcamdat.sh; bash'"
```

With resume (skip models that already have `output_dir/final/`):
```bash
bash /workspace/auto-mlm-pipes/deploy/run-all-mlm-efcamdat.sh --resume
```

## Step 4: Start GDrive sync (separate tmux window)

Syncs completed models to GDrive every 15 minutes. Runs independently from training.

```bash
ssh $SSH_URL "tmux new-window -t ssh_tmux -n mlm-sync \
  'bash /workspace/auto-mlm-pipes/deploy/sync-results-gdrive.sh; bash'"
```

### How the sync works

- Runs in a **loop every 15 minutes** in its own tmux window (`mlm-sync`)
- For each `outputs/encoder-*` directory:
  1. Checks if training is **complete**: `final/` dir exists + 10 `checkpoint-*` dirs (all epochs)
  2. Uploads to `i:/_p/artificial-learners/models/encoders/mlm/{family}/{size}/`
  3. **Verifies** remote file count matches local file count
  4. If verified: **removes local copy** to free disk space
  5. If mismatch: **keeps local copy** for manual handling
- **Ctrl+C** to stop, re-run the tmux command to restart

### GDrive target structure

```
i:/_p/artificial-learners/models/encoders/mlm/
  albert/base-v2/
  albert/large-v2/
  albert/xlarge-v2/
  albert/xxlarge-v2/
  bert/base/
  bert/large/
  debertav3/xsmall/
  debertav3/small/
  debertav3/base/
  debertav3/large/
  modernbert/base/
  modernbert/large/
  nomic-bert/base/
  roberta/base/
  roberta/large/
```

Each model dir contains:
- `checkpoint-*/` (all 10 epoch checkpoints)
- `final/` (best model)
- `resolved_config.yaml`, `train_metrics.json`, `eval_results.json`

### Sync monitoring

```bash
ssh $SSH_URL "tmux capture-pane -t ssh_tmux:mlm-sync -p -S -30"
```

## Monitoring

```bash
# List all tmux windows
ssh $SSH_URL "tmux list-windows -t ssh_tmux"

# Training progress
ssh $SSH_URL "tmux capture-pane -t ssh_tmux:mlm-train -p -S -30"
ssh $SSH_URL "tail -50 /workspace/mlm-training-all.log"
ssh $SSH_URL -t "tmux attach -t ssh_tmux:mlm-train"

# Sync progress
ssh $SSH_URL "tmux capture-pane -t ssh_tmux:mlm-sync -p -S -30"

# Disk space (all epochs saved = ~5-10GB per model)
ssh $SSH_URL "df -h /workspace"
ssh $SSH_URL "du -sh /workspace/auto-mlm-pipes/outputs/*"
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

## Training config notes

- **All epochs saved**: `save_total_limit` is unset (keeps all 10 epoch checkpoints)
- **Batch sizes**: tuned for 32GB VRAM (RTX 5090)
- **Sync**: handled separately via `sync-results-gdrive.sh` (not part of training script)

## Files

| File | Purpose |
|------|---------|
| `deploy/project-setup-remote.sh` | Pyenv + deps setup (run once) |
| `deploy/run-smoke-test.sh` | Quick GPU validation |
| `deploy/run-all-mlm-efcamdat.sh` | Full 15-model MLM training |
| `deploy/sync-results-gdrive.sh` | Periodic GDrive sync (every 15min) |
| `configs/dummies/smoke-test.yaml` | Smoke config (1 epoch, dummy data) |
| `scripts/train-all-mlm.sh` | Sequential training orchestrator |

## Disk space

With all epochs saved, each model uses ~5-10GB. The sync script frees space by
removing local copies after verified upload. Monitor disk usage:

```bash
ssh $SSH_URL "df -h /workspace"
```
