#!/usr/bin/env python3
"""
SPLADE sparse expansion training pipeline.

Uses MLM head weights to produce vocab-sized sparse representations with FLOPS
regularization. Trains on query-document pairs with contrastive loss.

Usage:
    python -m pipelines.train_splade --config configs/splade/dummy-cpu.yaml
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
from transformers import AutoModelForMaskedLM, AutoTokenizer, set_seed

from .config import (
    SPLADEConfig,
    config_to_dict,
    dump_resolved_config,
    parse_args_and_load_config,
)
from .data_utils import load_pairs_data
from .logging_utils import MetricLogger, setup_logging_env

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# SPLADE Encoder
# ---------------------------------------------------------------------------

class SPLADEEncoder(nn.Module):
    """Wraps AutoModelForMaskedLM to produce sparse vocab-level activations.

    Forward: input_ids -> MLM logits -> ReLU + log1p -> max-pool over tokens -> sparse vector
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch, seq_len, vocab_size)

        # Sparse activation: log(1 + ReLU(logits))
        sparse = torch.log1p(F.relu(logits))

        # Max-pool over sequence dimension (one sparse vector per document)
        if attention_mask is not None:
            # Mask padding tokens
            mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            sparse = sparse * mask

        sparse = sparse.max(dim=1).values  # (batch, vocab_size)
        return sparse


# ---------------------------------------------------------------------------
# FLOPS regularizer
# ---------------------------------------------------------------------------

class FLOPSRegularizer:
    """FLOPS-based sparsity regularization.

    FLOPS = mean(sum(activations)) over batch, penalizes dense representations.
    """

    def __call__(self, sparse_vectors):
        """sparse_vectors: (batch, vocab_size)"""
        # Mean activation per vocab dimension across batch
        mean_activations = sparse_vectors.mean(dim=0)  # (vocab_size,)
        # FLOPS: sum of squared mean activations (encourages sparsity)
        flops = (mean_activations ** 2).sum()
        return flops


# ---------------------------------------------------------------------------
# SPLADE Loss
# ---------------------------------------------------------------------------

class SPLADELoss(nn.Module):
    """Combined contrastive + FLOPS regularization loss for SPLADE."""

    def __init__(self, lambda_q: float = 0.0006, lambda_d: float = 0.0008,
                 temperature: float = 0.05):
        super().__init__()
        self.lambda_q = lambda_q
        self.lambda_d = lambda_d
        self.temperature = temperature
        self.flops_reg = FLOPSRegularizer()

    def forward(self, query_sparse, pos_sparse):
        """
        query_sparse: (B, vocab_size) — sparse query representations
        pos_sparse: (B, vocab_size) — sparse positive document representations
        In-batch negatives: all other docs are negatives.
        """
        # Dot product similarity: (B, B)
        scores = torch.matmul(query_sparse, pos_sparse.t()) / self.temperature

        # Contrastive loss: diagonal are positives
        labels = torch.arange(scores.size(0), device=scores.device)
        contrastive_loss = F.cross_entropy(scores, labels)

        # FLOPS regularization
        flops_q = self.flops_reg(query_sparse)
        flops_d = self.flops_reg(pos_sparse)
        reg_loss = self.lambda_q * flops_q + self.lambda_d * flops_d

        return contrastive_loss + reg_loss, contrastive_loss.item(), reg_loss.item()


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

class SPLADECollator:
    """Tokenize query-document pairs for SPLADE training."""

    def __init__(self, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        queries = [ex["query"] for ex in examples]
        positives = [ex["positive"] for ex in examples]

        query_enc = self.tokenizer(
            queries, max_length=self.max_length, truncation=True, padding=True, return_tensors="pt"
        )
        pos_enc = self.tokenizer(
            positives, max_length=self.max_length, truncation=True, padding=True, return_tensors="pt"
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

def train_epoch(model, dataloader, loss_fn, optimizer, scheduler, device, epoch,
                grad_accum=1, logging_steps=10):
    model.train()
    total_loss = 0.0
    total_contrastive = 0.0
    total_reg = 0.0
    num_batches = 0

    for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        q_ids = batch["query_input_ids"].to(device)
        q_mask = batch["query_attention_mask"].to(device)
        p_ids = batch["pos_input_ids"].to(device)
        p_mask = batch["pos_attention_mask"].to(device)

        query_sparse = model(q_ids, q_mask)
        pos_sparse = model(p_ids, p_mask)

        loss, c_loss, r_loss = loss_fn(query_sparse, pos_sparse)
        (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        total_contrastive += c_loss
        total_reg += r_loss
        num_batches += 1

        if (step + 1) % logging_steps == 0:
            avg = total_loss / num_batches
            logger.info(f"  step {step + 1}: loss={avg:.4f} (contrastive={total_contrastive / num_batches:.4f}, reg={total_reg / num_batches:.4f})")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        q_ids = batch["query_input_ids"].to(device)
        q_mask = batch["query_attention_mask"].to(device)
        p_ids = batch["pos_input_ids"].to(device)
        p_mask = batch["pos_attention_mask"].to(device)

        query_sparse = model(q_ids, q_mask)
        pos_sparse = model(p_ids, p_mask)
        loss, _, _ = loss_fn(query_sparse, pos_sparse)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(cfg: SPLADEConfig):
    set_seed(cfg.seed)
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    dump_resolved_config(cfg, output_dir)

    logger.info(f"Model: {cfg.model.model}")
    logger.info(f"Lambda Q: {cfg.lambda_q}, Lambda D: {cfg.lambda_d}")
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

    backbone = AutoModelForMaskedLM.from_pretrained(
        cfg.model.model, torch_dtype=dtype, trust_remote_code=cfg.model.trust_remote_code, **attn_impl
    )
    model = SPLADEEncoder(backbone).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # --- Data ---
    raw_datasets = load_pairs_data(cfg)
    has_validation = len(raw_datasets["validation"]) > 0

    collator = SPLADECollator(tokenizer, cfg.max_length)
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
    loss_fn = SPLADELoss(lambda_q=cfg.lambda_q, lambda_d=cfg.lambda_d, temperature=cfg.temperature)
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
            model, train_loader, loss_fn, optimizer, scheduler, device, epoch,
            cfg.training.gradient_accumulation_steps, cfg.logging.logging_steps
        )
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
    model.backbone.save_pretrained(os.path.join(final_dir, "backbone"))
    tokenizer.save_pretrained(final_dir)

    metric_logger.finish()
    logger.info(f"Training complete. Model saved to: {final_dir}")


def main():
    cfg = parse_args_and_load_config(SPLADEConfig, "SPLADE sparse expansion training pipeline")
    run(cfg)


if __name__ == "__main__":
    main()
