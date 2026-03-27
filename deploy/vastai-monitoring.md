# vastai server monitoring & watchdog

Monitoring, debugging, and auto-recovery for MLM training on vastai GPU servers.

## Prerequisites

```bash
cd ~/p/all-my-tiny-projects/vastai
SSH_URL=$(./vastai-ssh get-url)
```

## Quick status checks

### GPU usage

```bash
# One-liner: utilization + memory
ssh $SSH_URL "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"

# Full nvidia-smi
ssh $SSH_URL "nvidia-smi"
```

### Training progress

```bash
# Last N lines from training tmux pane
ssh $SSH_URL "tmux capture-pane -t ssh_tmux:mlm-train -p -S -20"

# Training log file
ssh $SSH_URL "tail -50 /workspace/mlm-training-all.log"

# Attach directly (Ctrl+B then D to detach)
ssh $SSH_URL -t "tmux attach -t ssh_tmux:mlm-train"
```

### Which models are done / in progress / failed

```bash
ssh $SSH_URL 'for d in /workspace/auto-mlm-pipes/outputs/encoder-*; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  ckpts=$(ls -d "$d"/checkpoint-* 2>/dev/null | wc -l)
  if [ -d "$d/final" ]; then
    echo "DONE ($ckpts ckpts): $name"
  elif [ "$ckpts" -gt 0 ]; then
    echo "IN PROGRESS ($ckpts ckpts): $name"
  else
    echo "FAILED (0 ckpts): $name"
  fi
done'
```

### Config check for a specific model

```bash
# Batch size
grep "per_device_train_batch_size" configs/efcamdat/encoder/<model>-mlm.yaml

# Gradient checkpointing
grep "gradient_checkpointing" configs/efcamdat/encoder/<model>-mlm.yaml
```

### Disk space

```bash
ssh $SSH_URL "df -h /workspace"
ssh $SSH_URL "du -sh /workspace/auto-mlm-pipes/outputs/*"
```

### Sync status

```bash
ssh $SSH_URL "tmux capture-pane -t ssh_tmux:mlm-sync -p -S -20"
```

### List all tmux windows

```bash
ssh $SSH_URL "tmux list-windows -t ssh_tmux"
```

## GDrive sync (rclone, every 10 min)

Syncs all outputs to GDrive regardless of completion status.

### Start sync tmux window

```bash
ssh $SSH_URL "tmux new-window -t ssh_tmux -n mlm-sync \
  'while true; do \
     echo \"--- sync: \$(date) ---\"; \
     rclone copy /workspace/auto-mlm-pipes/outputs/ \"i:/_p/artificial-learners/models/encoders/mlm/\" --verbose 2>&1 | tail -5; \
     echo \"sleeping 600s...\"; echo \"\"; \
     sleep 600; \
   done'"
```

### Verify sync worked

```bash
ssh $SSH_URL "rclone lsf 'i:/_p/artificial-learners/models/encoders/mlm/'"
ssh $SSH_URL "rclone ls 'i:/_p/artificial-learners/models/encoders/mlm/' | head -30"
```

### Manual one-shot sync

```bash
ssh $SSH_URL "rclone copy /workspace/auto-mlm-pipes/outputs/ 'i:/_p/artificial-learners/models/encoders/mlm/' --verbose"
```

### Stop / restart sync

- **Ctrl+C** in the `mlm-sync` tmux pane to stop
- Re-run the `tmux new-window` command above to restart

## Watchdog (local, auto-resumes training)

Runs on your **local machine**. Checks the vastai server GPU every 10 minutes.
If GPU is idle (<10%) and no `train_encoder` process is running, it:

1. Cleans empty output dirs (no checkpoints = OOM'd)
2. Pulls latest code on remote
3. Relaunches training with `--resume` (skips models with `final/`)

### Create watchdog script

```bash
cat << 'WATCHDOG' > /tmp/mlm-watchdog.sh
#!/bin/bash
SSH_URL="ssh://root@<IP>:<PORT>"  # update with current vastai SSH URL
INTERVAL=600  # 10 minutes

while true; do
    gpu_pct=$(ssh -o ConnectTimeout=10 "$SSH_URL" \
      "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits" \
      2>/dev/null | tr -d '[:space:]')

    if [[ -z "$gpu_pct" ]]; then
        echo "[$(date '+%H:%M:%S')] SSH failed, retrying next cycle"
        sleep "$INTERVAL"
        continue
    fi

    echo "[$(date '+%H:%M:%S')] GPU: ${gpu_pct}%"

    if [[ "$gpu_pct" -lt 10 ]]; then
        echo "[$(date '+%H:%M:%S')] GPU IDLE (<10%). Investigating..."

        echo "--- last training output ---"
        ssh "$SSH_URL" "tmux capture-pane -t ssh_tmux:9 -p -S -5 2>/dev/null" || true

        echo "--- completed (have final/) ---"
        ssh "$SSH_URL" 'for d in /workspace/auto-mlm-pipes/outputs/encoder-*; do
          [ -d "$d/final" ] && echo "  DONE: $(basename $d)"; done 2>/dev/null' || true

        py_count=$(ssh "$SSH_URL" \
          "pgrep -cf train_encoder 2>/dev/null; exit 0" \
          2>/dev/null | tr -d '[:space:]')
        echo "train_encoder procs: ${py_count:-0}"

        if [[ "${py_count:-0}" -eq 0 ]]; then
            echo "[$(date '+%H:%M:%S')] No training process. Cleaning empty dirs and resuming..."

            # Remove empty output dirs (OOM'd - no checkpoints, no final)
            ssh "$SSH_URL" 'for d in /workspace/auto-mlm-pipes/outputs/encoder-*; do
                [ -d "$d" ] || continue
                [ -d "$d/final" ] && continue
                ckpts=$(ls -d "$d"/checkpoint-* 2>/dev/null | wc -l)
                [ "$ckpts" -eq 0 ] && echo "  rm empty: $(basename $d)" && rm -rf "$d"
            done' 2>/dev/null || true

            # Pull latest code
            ssh "$SSH_URL" "cd /workspace/auto-mlm-pipes && \
              export PYENV_ROOT=/root/.pyenv && \
              export PATH=\$PYENV_ROOT/bin:\$PYENV_ROOT/shims:\$PATH && \
              eval \"\$(pyenv init -)\" && \
              git pull --ff-only" 2>/dev/null || true

            # Relaunch training with --resume
            ssh "$SSH_URL" "tmux new-window -t ssh_tmux -n mlm-train \
              'export PYENV_ROOT=/root/.pyenv && \
               export PATH=\$PYENV_ROOT/bin:\$PYENV_ROOT/shims:\$PATH && \
               eval \"\$(pyenv init -)\" && \
               bash /workspace/auto-mlm-pipes/deploy/run-all-mlm-efcamdat.sh --skip-fetch --resume; bash'" \
              2>/dev/null

            echo "[$(date '+%H:%M:%S')] >>> RESUMED training with --resume <<<"
        else
            echo "[$(date '+%H:%M:%S')] Training alive but GPU idle - probably loading/saving model"
        fi
    fi

    sleep "$INTERVAL"
done
WATCHDOG
chmod +x /tmp/mlm-watchdog.sh
```

### Start watchdog

```bash
nohup bash /tmp/mlm-watchdog.sh > /tmp/mlm-watchdog.log 2>&1 &
echo $! > /tmp/mlm-watchdog.pid
```

### Monitor watchdog

```bash
tail -f /tmp/mlm-watchdog.log
```

### Stop watchdog

```bash
kill $(cat /tmp/mlm-watchdog.pid)
```

## Troubleshooting

### Model OOM'd

Symptoms: output dir exists but has 0 checkpoints and no `final/`.

Fix: reduce batch size in the config, push, and let the watchdog resume.

```bash
# Check which models OOM'd
ssh $SSH_URL 'for d in /workspace/auto-mlm-pipes/outputs/encoder-*; do
  [ -d "$d" ] || continue
  [ -d "$d/final" ] && continue
  ckpts=$(ls -d "$d"/checkpoint-* 2>/dev/null | wc -l)
  [ "$ckpts" -eq 0 ] && echo "OOM: $(basename $d)"
done'

# Edit config locally, e.g.:
#   per_device_train_batch_size: 4
#   gradient_checkpointing: true

# Push and watchdog will pick it up
cd ~/p/research-sketches/auto-mlm-pipes
git add -A && git commit -m "reduce batch size for OOM model" && git push origin main
```

### Training stuck / process died

```bash
# Check if training process is alive
ssh $SSH_URL "pgrep -af train_encoder"

# Check GPU
ssh $SSH_URL "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"

# Manual resume
ssh $SSH_URL "tmux new-window -t ssh_tmux -n mlm-train \
  'export PYENV_ROOT=/root/.pyenv && \
   export PATH=\$PYENV_ROOT/bin:\$PYENV_ROOT/shims:\$PATH && \
   eval \"\$(pyenv init -)\" && \
   bash /workspace/auto-mlm-pipes/deploy/run-all-mlm-efcamdat.sh --skip-fetch --resume; bash'"
```

### Too many tmux windows

```bash
# List all
ssh $SSH_URL "tmux list-windows -t ssh_tmux"

# Kill a specific window by number
ssh $SSH_URL "tmux kill-window -t ssh_tmux:<N>"
```
