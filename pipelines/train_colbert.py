#!/usr/bin/env python3
"""
ColBERT late-interaction training pipeline.

Trains per-token multi-vector representations with MaxSim contrastive loss.
Supports ColBERT, ColBERTv2, GTE-ModernColBERT, Liquid ColBERT backbones.

Usage:
    python -m pipelines.train_colbert --config configs/colbert/dummy-cpu.yaml
"""

import datetime
import json
import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, set_seed

from .config import (
    ColBERTConfig,
    config_to_dict,
    dump_resolved_config,
    parse_args_and_load_config,
)
from .data_utils import load_pairs_data
from .logging_utils import MetricLogger, setup_logging_env

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# ColBERT Encoder
# ---------------------------------------------------------------------------

class ColBERTEncoder(nn.Module):
    """Wraps a backbone + linear projection for per-token embeddings."""

    def __init__(self, backbone: nn.Module, dim: int = 128):
        super().__init__()
        self.backbone = backbone
        hidden_size = backbone.config.hidden_size
        self.projection = nn.Linear(hidden_size, dim)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        projected = self.projection(hidden)  # (batch, seq_len, dim)
        # L2 normalize per-token embeddings
        projected = F.normalize(projected, p=2, dim=-1)
        return projected


# ---------------------------------------------------------------------------
# MaxSim loss
# ---------------------------------------------------------------------------

def maxsim_score(query_embs, doc_embs, query_mask=None, doc_mask=None):
    """Compute MaxSim score between query and document embeddings.

    query_embs: (batch_q, q_len, dim)
    doc_embs: (batch_d, d_len, dim)
    Returns: (batch_q, batch_d) similarity matrix
    """
    # (batch_q, q_len, 1, dim) x (1, 1, batch_d * d_len, dim)
    # Efficient: (batch_q, q_len, dim) @ (batch_d, d_len, dim).T per pair
    batch_q = query_embs.size(0)
    batch_d = doc_embs.size(0)
    q_len = query_embs.size(1)
    d_len = doc_embs.size(1)
    dim = query_embs.size(2)

    # Reshape for batch matmul: (batch_q, q_len, dim) vs (batch_d, d_len, dim)
    # -> (batch_q, batch_d, q_len, d_len)
    scores = torch.einsum("aqd,bpd->abqp", query_embs, doc_embs)

    # Mask document padding
    if doc_mask is not None:
        doc_mask_expanded = doc_mask.unsqueeze(0).unsqueeze(2)  # (1, batch_d, 1, d_len)
        scores = scores.masked_fill(~doc_mask_expanded.bool(), float("-inf"))

    # MaxSim: max over document tokens for each query token
    max_scores = scores.max(dim=-1).values  # (batch_q, batch_d, q_len)

    # Mask query padding
    if query_mask is not None:
        query_mask_expanded = query_mask.unsqueeze(1)  # (batch_q, 1, q_len)
        max_scores = max_scores.masked_fill(~query_mask_expanded.bool(), 0.0)

    # Sum over query tokens
    return max_scores.sum(dim=-1)  # (batch_q, batch_d)


class ColBERTLoss(nn.Module):
    """Contrastive loss with MaxSim scoring and in-batch negatives."""

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, query_embs, pos_embs, query_mask=None, pos_mask=None):
        """
        query_embs: (B, q_len, dim) — encoded queries
        pos_embs: (B, d_len, dim) — encoded positive documents
        Uses in-batch negatives: all other documents in batch are negatives.
        """
        # Compute all-pairs MaxSim scores: (B, B)
        scores = maxsim_score(query_embs, pos_embs, query_mask, pos_mask)
        scores = scores / self.temperature

        # Labels: diagonal (each query matches its own positive)
        labels = torch.arange(scores.size(0), device=scores.device)
        loss = F.cross_entropy(scores, labels)
        return loss


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

class ColBERTCollator:
    """Tokenize query-document pairs for ColBERT training."""

    def __init__(self, tokenizer, query_maxlen: int = 32, doc_maxlen: int = 256):
        self.tokenizer = tokenizer
        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen

    def __call__(self, examples):
        queries = [ex["query"] for ex in examples]
        positives = [ex["positive"] for ex in examples]

        query_enc = self.tokenizer(
            queries, max_length=self.query_maxlen, truncation=True, padding=True, return_tensors="pt"
        )
        pos_enc = self.tokenizer(
            positives, max_length=self.doc_maxlen, truncation=True, padding=True, return_tensors="pt"
        )

        return {
            "query_input_ids": query_enc["input_ids"],
            "query_attention_mask": query_enc["attention_mask"],
            "pos_input_ids": pos_enc["input_ids"],
            "pos_attention_mask": pos_enc["attention_mask"],
        }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, loss_fn, optimizer, scheduler, device, epoch, logging_steps=10):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        query_ids = batch["query_input_ids"].to(device)
        query_mask = batch["query_attention_mask"].to(device)
        pos_ids = batch["pos_input_ids"].to(device)
        pos_mask = batch["pos_attention_mask"].to(device)

        query_embs = model(query_ids, query_mask)
        pos_embs = model(pos_ids, pos_mask)

        loss = loss_fn(query_embs, pos_embs, query_mask, pos_mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

        if (step + 1) % logging_steps == 0:
            avg = total_loss / num_batches
            logger.info(f"  step {step + 1}: loss={avg:.4f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        query_ids = batch["query_input_ids"].to(device)
        query_mask = batch["query_attention_mask"].to(device)
        pos_ids = batch["pos_input_ids"].to(device)
        pos_mask = batch["pos_attention_mask"].to(device)

        query_embs = model(query_ids, query_mask)
        pos_embs = model(pos_ids, pos_mask)
        loss = loss_fn(query_embs, pos_embs, query_mask, pos_mask)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(cfg: ColBERTConfig):
    set_seed(cfg.seed)
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    dump_resolved_config(cfg, output_dir)

    logger.info(f"Model: {cfg.model.model}")
    logger.info(f"Projection dim: {cfg.dim}")
    logger.info(f"Output: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Tokenizer ---
    tok_name = cfg.model.tokenizer or cfg.model.model
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=cfg.model.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Model ---
    attn_impl = {"attn_implementation": "flash_attention_2"} if cfg.model.flash_attention else {}
    dtype = torch.bfloat16 if cfg.training.bf16 else (torch.float16 if cfg.training.fp16 else torch.float32)

    if cfg.model.from_scratch:
        model_config = AutoModel.from_pretrained(cfg.model.model, trust_remote_code=True).config
        backbone = AutoModel.from_config(model_config, **attn_impl, trust_remote_code=True)
    else:
        backbone = AutoModel.from_pretrained(
            cfg.model.model, torch_dtype=dtype, trust_remote_code=cfg.model.trust_remote_code, **attn_impl
        )

    model = ColBERTEncoder(backbone, dim=cfg.dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # --- Data ---
    raw_datasets = load_pairs_data(cfg)
    has_validation = len(raw_datasets["validation"]) > 0

    collator = ColBERTCollator(tokenizer, cfg.query_maxlen, cfg.doc_maxlen)
    train_loader = DataLoader(
        raw_datasets["train"], batch_size=cfg.training.per_device_train_batch_size,
        shuffle=True, collate_fn=collator, num_workers=cfg.training.dataloader_num_workers,
    )
    val_loader = DataLoader(
        raw_datasets["validation"], batch_size=cfg.training.per_device_eval_batch_size,
        shuffle=False, collate_fn=collator,
    ) if has_validation else None

    logger.info(f"Train pairs: {len(raw_datasets['train']):,}")

    # --- Loss, optimizer, scheduler ---
    loss_fn = ColBERTLoss(temperature=cfg.temperature)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay
    )
    total_steps = len(train_loader) * cfg.training.num_train_epochs
    warmup_steps = int(total_steps * cfg.training.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.training.learning_rate, total_steps=total_steps,
        pct_start=warmup_steps / total_steps if total_steps > 0 else 0.1,
    )

    # --- Logging ---
    report_to = setup_logging_env(cfg)
    flat = config_to_dict(cfg)
    metric_logger = MetricLogger(report_to, flat)
    metric_logger.init()

    # --- Train ---
    best_val_loss = float("inf")
    for epoch in range(1, cfg.training.num_train_epochs + 1):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, scheduler, device, epoch, cfg.logging.logging_steps)
        logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}")

        metrics = {"train/loss": train_loss, "epoch": epoch}

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, loss_fn, device)
            logger.info(f"Epoch {epoch}: val_loss={val_loss:.4f}")
            metrics["val/loss"] = val_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_dir = os.path.join(output_dir, "best")
                os.makedirs(best_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(best_dir, "model.pt"))
                tokenizer.save_pretrained(best_dir)

        metric_logger.log(metrics, step=epoch)

    # --- Save final ---
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_dir, "model.pt"))
    tokenizer.save_pretrained(final_dir)
    # Save backbone separately for downstream use
    model.backbone.save_pretrained(os.path.join(final_dir, "backbone"))

    metric_logger.finish()
    logger.info(f"Training complete. Model saved to: {final_dir}")


def main():
    cfg = parse_args_and_load_config(ColBERTConfig, "ColBERT late-interaction training pipeline")
    run(cfg)


if __name__ == "__main__":
    main()
