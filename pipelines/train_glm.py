#!/usr/bin/env python3
"""
GLM autoregressive blank infilling pipeline.

Implements the GLM training objective: mask contiguous spans, then reconstruct
them autoregressively in shuffled order with 2D positional encoding.

Usage:
    python -m pipelines.train_glm --config configs/glm/dummy-cpu.yaml
"""

import datetime
import json
import logging
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed

from .config import (
    GLMConfig,
    config_to_dict,
    dump_resolved_config,
    parse_args_and_load_config,
)
from .data_utils import load_data
from .logging_utils import MetricLogger, setup_logging_env

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Blank infilling data preparation
# ---------------------------------------------------------------------------

class BlankInfillingCollator:
    """Prepare (corrupted_prefix, infill_targets) for GLM-style training.

    Each example is split into:
      - prefix: original tokens with [MASK] replacing blanked spans
      - infill: shuffled span tokens concatenated, to be generated autoregressively

    The full sequence is: [prefix] [START_INFILL] [span1_tokens] [SEP] [span2_tokens] [SEP] ...
    Labels: -100 for prefix tokens, actual tokens for infill tokens.
    """

    def __init__(self, tokenizer, mask_ratio: float = 0.15, avg_span_length: int = 3,
                 shuffle_spans: bool = True, max_length: int = 512):
        self.tokenizer = tokenizer
        self.mask_ratio = mask_ratio
        self.avg_span_length = avg_span_length
        self.shuffle_spans = shuffle_spans
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id or 0
        self.mask_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else tokenizer.convert_tokens_to_ids("[MASK]")
        self.sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id or 0

    def __call__(self, examples):
        batch_ids = []
        batch_labels = []
        batch_position_ids = []

        for ex in examples:
            text = ex["text"]
            tokens = self.tokenizer.encode(text, max_length=self.max_length, truncation=True)
            ids, labels, pos_ids = self._create_blank_infilling(tokens)
            batch_ids.append(ids)
            batch_labels.append(labels)
            batch_position_ids.append(pos_ids)

        # Pad
        max_len = max(len(ids) for ids in batch_ids)
        padded_ids, padded_labels, padded_pos, attn_masks = [], [], [], []
        for ids, labs, pos in zip(batch_ids, batch_labels, batch_position_ids):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [self.pad_id] * pad_len)
            padded_labels.append(labs + [-100] * pad_len)
            padded_pos.append(pos + [0] * pad_len)
            attn_masks.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "position_ids": torch.tensor(padded_pos, dtype=torch.long),
            "attention_mask": torch.tensor(attn_masks, dtype=torch.long),
        }

    def _create_blank_infilling(self, tokens):
        """Create corrupted prefix + infill targets with 2D positional encoding."""
        length = len(tokens)
        num_to_mask = max(1, int(length * self.mask_ratio))

        # Select spans
        spans = []
        masked_count = 0
        attempts = 0
        while masked_count < num_to_mask and attempts < 100:
            start = random.randint(0, length - 1)
            span_len = max(1, int(np.random.geometric(1.0 / self.avg_span_length)))
            span_len = min(span_len, length - start, num_to_mask - masked_count)
            overlaps = any(s <= start < s + l or s <= start + span_len - 1 < s + l for s, l in spans)
            if not overlaps:
                spans.append((start, span_len))
                masked_count += span_len
            attempts += 1

        spans.sort(key=lambda x: x[0])

        # Optionally shuffle span order for generation
        span_order = list(range(len(spans)))
        if self.shuffle_spans and len(spans) > 1:
            random.shuffle(span_order)

        # Build corrupted prefix: replace each span with [MASK]
        prefix = []
        prefix_positions = []
        pos = 0
        span_idx = 0
        i = 0
        while i < length:
            if span_idx < len(spans) and i == spans[span_idx][0]:
                start, span_len = spans[span_idx]
                prefix.append(self.mask_id)
                prefix_positions.append(pos)
                pos += 1
                i += span_len
                span_idx += 1
            else:
                prefix.append(tokens[i])
                prefix_positions.append(pos)
                pos += 1
                i += 1

        # Build infill sequence: [span1_tokens SEP span2_tokens SEP ...]
        infill = []
        infill_labels = []
        infill_positions = []

        for order_idx, si in enumerate(span_order):
            start, span_len = spans[si]
            span_tokens = tokens[start:start + span_len]
            for j, tok in enumerate(span_tokens):
                infill.append(tok)
                infill_labels.append(tok)
                # 2D position: (position_in_corrupted_text, intra_span_position)
                # Simplified: use span start position + intra offset
                infill_positions.append(start + j)
            infill.append(self.sep_id)
            infill_labels.append(self.sep_id)
            infill_positions.append(start + span_len)

        # Combine: prefix (no labels) + infill (with labels)
        # For autoregressive: shift labels by 1
        full_ids = prefix + infill
        full_labels = [-100] * len(prefix)
        # Infill labels are shifted: predict next token
        for j in range(len(infill)):
            if j + 1 < len(infill):
                full_labels.append(infill[j + 1])
            else:
                full_labels.append(-100)

        full_positions = prefix_positions + infill_positions

        # Truncate to max_length
        full_ids = full_ids[:self.max_length]
        full_labels = full_labels[:self.max_length]
        full_positions = full_positions[:self.max_length]

        return full_ids, full_labels, full_positions


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, grad_accum=1, logging_steps=10):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += outputs.loss.item()
        num_batches += 1

        if (step + 1) % logging_steps == 0:
            avg = total_loss / num_batches
            logger.info(f"  step {step + 1}: loss={avg:.4f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(cfg: GLMConfig):
    set_seed(cfg.seed)
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    dump_resolved_config(cfg, output_dir)

    logger.info(f"Model: {cfg.model.model}")
    logger.info(f"Mask ratio: {cfg.mask_ratio}, Avg span: {cfg.avg_span_length}, Shuffle: {cfg.shuffle_spans}")
    logger.info(f"Output: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Tokenizer ---
    tok_name = cfg.model.tokenizer or cfg.model.model
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=cfg.model.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({"sep_token": "[SEP]"})

    # --- Model ---
    attn_impl = {"attn_implementation": "flash_attention_2"} if cfg.model.flash_attention else {}
    dtype = torch.bfloat16 if cfg.training.bf16 else (torch.float16 if cfg.training.fp16 else torch.float32)

    if cfg.model.from_scratch:
        model_config = AutoConfig.from_pretrained(cfg.model.model, trust_remote_code=cfg.model.trust_remote_code)
        model = AutoModelForCausalLM.from_config(model_config, **attn_impl, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.model, torch_dtype=dtype, trust_remote_code=cfg.model.trust_remote_code, **attn_impl
        )

    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # --- Data ---
    raw_datasets = load_data(cfg)
    has_validation = len(raw_datasets["validation"]) > 0

    collator = BlankInfillingCollator(
        tokenizer, cfg.mask_ratio, cfg.avg_span_length, cfg.shuffle_spans, cfg.max_length
    )
    train_loader = DataLoader(
        raw_datasets["train"], batch_size=cfg.training.per_device_train_batch_size,
        shuffle=True, collate_fn=collator, num_workers=cfg.training.dataloader_num_workers,
    )
    val_loader = DataLoader(
        raw_datasets["validation"], batch_size=cfg.training.per_device_eval_batch_size,
        shuffle=False, collate_fn=collator,
    ) if has_validation else None

    logger.info(f"Train examples: {len(raw_datasets['train']):,}")

    # --- Optimizer + scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay
    )
    total_steps = (len(train_loader) // cfg.training.gradient_accumulation_steps) * cfg.training.num_train_epochs
    warmup_steps = int(total_steps * cfg.training.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.training.learning_rate, total_steps=max(total_steps, 1),
        pct_start=warmup_steps / max(total_steps, 1),
    )

    # --- Logging ---
    report_to = setup_logging_env(cfg)
    flat = config_to_dict(cfg)
    metric_logger = MetricLogger(report_to, flat)
    metric_logger.init()

    # --- Train ---
    best_val_loss = float("inf")
    for epoch in range(1, cfg.training.num_train_epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
            cfg.training.gradient_accumulation_steps, cfg.logging.logging_steps
        )
        logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}")
        metrics = {"train/loss": train_loss, "epoch": epoch}

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device)
            logger.info(f"Epoch {epoch}: val_loss={val_loss:.4f}")
            metrics["val/loss"] = val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_dir = os.path.join(output_dir, "best")
                os.makedirs(best_dir, exist_ok=True)
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)

        metric_logger.log(metrics, step=epoch)

    # --- Save final ---
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    metric_logger.finish()
    logger.info(f"Training complete. Model saved to: {final_dir}")


def main():
    cfg = parse_args_and_load_config(GLMConfig, "GLM autoregressive blank infilling pipeline")
    run(cfg)


if __name__ == "__main__":
    main()
