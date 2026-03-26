#!/usr/bin/env python3
"""
RetroMAE asymmetric masked auto-encoder pipeline.

Pre-trains retrieval-oriented encoders using asymmetric masking:
  - Encoder: full BERT-scale, low masking (15-30%)
  - Decoder: 1-layer transformer, high masking (50-70%)
The decoder reconstructs from [CLS] embedding + heavily-masked tokens.

Usage:
    python -m pipelines.train_retromae --config configs/retromae/dummy-cpu.yaml
"""

import datetime
import json
import logging
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, set_seed

from .config import (
    RetroMAEConfig,
    config_to_dict,
    dump_resolved_config,
    parse_args_and_load_config,
)
from .data_utils import load_data
from .logging_utils import MetricLogger, setup_logging_env

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# RetroMAE Decoder
# ---------------------------------------------------------------------------

class RetroMAEDecoder(nn.Module):
    """Lightweight 1-layer transformer decoder for RetroMAE.

    Takes [CLS] embedding from encoder + heavily-masked token embeddings,
    and reconstructs the original tokens.
    """

    def __init__(self, hidden_size: int, vocab_size: int, num_layers: int = 1,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(hidden_size, vocab_size)

    def forward(self, decoder_input, encoder_cls, attention_mask=None):
        """
        decoder_input: (batch, seq_len, hidden) — masked token embeddings
        encoder_cls: (batch, 1, hidden) — [CLS] from encoder
        Returns: logits (batch, seq_len, vocab_size)
        """
        # Use encoder [CLS] as memory for cross-attention
        memory = encoder_cls  # (batch, 1, hidden)
        output = self.transformer(decoder_input, memory)  # (batch, seq_len, hidden)
        logits = self.output_projection(output)  # (batch, seq_len, vocab_size)
        return logits


# ---------------------------------------------------------------------------
# RetroMAE Model
# ---------------------------------------------------------------------------

class RetroMAEModel(nn.Module):
    """Full RetroMAE: encoder + lightweight decoder."""

    def __init__(self, encoder, decoder, embedding_layer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding_layer = embedding_layer  # shared word embeddings

    def forward(self, encoder_input_ids, encoder_attention_mask,
                decoder_input_ids, decoder_attention_mask, labels):
        """
        encoder_input_ids: lightly masked input
        decoder_input_ids: heavily masked input
        labels: original token ids (reconstruction target)
        """
        # Encoder forward
        encoder_outputs = self.encoder(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
        )
        cls_embedding = encoder_outputs.last_hidden_state[:, 0:1, :]  # (batch, 1, hidden)

        # Decoder forward
        decoder_embeddings = self.embedding_layer(decoder_input_ids)  # (batch, seq_len, hidden)
        decoder_logits = self.decoder(decoder_embeddings, cls_embedding, decoder_attention_mask)

        # Reconstruction loss: cross-entropy on all non-padding positions
        loss = F.cross_entropy(
            decoder_logits.view(-1, decoder_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        return loss, decoder_logits


# ---------------------------------------------------------------------------
# Asymmetric mask collator
# ---------------------------------------------------------------------------

class AsymmetricMaskCollator:
    """Produce (encoder_input, decoder_input) with different mask ratios.

    encoder: low masking (15-30%) — produces good [CLS] embedding
    decoder: high masking (50-70%) — must rely on [CLS] to reconstruct
    """

    def __init__(self, tokenizer, encoder_mask_ratio: float = 0.15,
                 decoder_mask_ratio: float = 0.50, max_length: int = 512):
        self.tokenizer = tokenizer
        self.encoder_mask_ratio = encoder_mask_ratio
        self.decoder_mask_ratio = decoder_mask_ratio
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id or 0
        self.mask_id = tokenizer.mask_token_id or tokenizer.convert_tokens_to_ids("[MASK]")

    def __call__(self, examples):
        texts = [ex["text"] for ex in examples]
        encodings = self.tokenizer(
            texts, max_length=self.max_length, truncation=True, padding=True, return_tensors="pt"
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        batch_size, seq_len = input_ids.shape

        # Create encoder inputs (low masking)
        encoder_ids = input_ids.clone()
        for i in range(batch_size):
            length = attention_mask[i].sum().item()
            maskable = list(range(1, length - 1))  # skip [CLS] and [SEP]
            num_mask = max(1, int(len(maskable) * self.encoder_mask_ratio))
            positions = random.sample(maskable, min(num_mask, len(maskable)))
            for pos in positions:
                encoder_ids[i, pos] = self.mask_id

        # Create decoder inputs (high masking)
        decoder_ids = input_ids.clone()
        for i in range(batch_size):
            length = attention_mask[i].sum().item()
            maskable = list(range(1, length - 1))
            num_mask = max(1, int(len(maskable) * self.decoder_mask_ratio))
            positions = random.sample(maskable, min(num_mask, len(maskable)))
            for pos in positions:
                decoder_ids[i, pos] = self.mask_id

        # Labels: original tokens (ignore padding)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "encoder_input_ids": encoder_ids,
            "encoder_attention_mask": attention_mask,
            "decoder_input_ids": decoder_ids,
            "decoder_attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch,
                grad_accum=1, logging_steps=10):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        enc_ids = batch["encoder_input_ids"].to(device)
        enc_mask = batch["encoder_attention_mask"].to(device)
        dec_ids = batch["decoder_input_ids"].to(device)
        dec_mask = batch["decoder_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        loss, _ = model(enc_ids, enc_mask, dec_ids, dec_mask, labels)
        (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
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
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        enc_ids = batch["encoder_input_ids"].to(device)
        enc_mask = batch["encoder_attention_mask"].to(device)
        dec_ids = batch["decoder_input_ids"].to(device)
        dec_mask = batch["decoder_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        loss, _ = model(enc_ids, enc_mask, dec_ids, dec_mask, labels)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(cfg: RetroMAEConfig):
    set_seed(cfg.seed)
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    dump_resolved_config(cfg, output_dir)

    logger.info(f"Model: {cfg.model.model}")
    logger.info(f"Encoder mask: {cfg.encoder_mask_ratio}, Decoder mask: {cfg.decoder_mask_ratio}")
    logger.info(f"Decoder layers: {cfg.decoder_layers}")
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

    encoder = AutoModel.from_pretrained(
        cfg.model.model, torch_dtype=dtype, trust_remote_code=cfg.model.trust_remote_code, **attn_impl
    )

    hidden_size = encoder.config.hidden_size
    decoder_hidden = cfg.decoder_hidden_dim if cfg.decoder_hidden_dim > 0 else hidden_size
    vocab_size = encoder.config.vocab_size

    decoder = RetroMAEDecoder(
        hidden_size=decoder_hidden,
        vocab_size=vocab_size,
        num_layers=cfg.decoder_layers,
        num_heads=max(1, decoder_hidden // 64),
    )

    # Share word embeddings between encoder and decoder
    embedding_layer = encoder.get_input_embeddings()

    model = RetroMAEModel(encoder, decoder, embedding_layer).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    enc_params = sum(p.numel() for p in encoder.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    logger.info(f"Total params: {n_params:,} (encoder: {enc_params:,}, decoder: {dec_params:,})")

    # --- Data ---
    raw_datasets = load_data(cfg)
    has_validation = len(raw_datasets["validation"]) > 0

    collator = AsymmetricMaskCollator(
        tokenizer, cfg.encoder_mask_ratio, cfg.decoder_mask_ratio, cfg.max_length
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
                encoder.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)

        metric_logger.log(metrics, step=epoch)

    # --- Save final ---
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    # Save encoder (the useful part for downstream)
    encoder.save_pretrained(os.path.join(final_dir, "encoder"))
    # Save full model state
    torch.save(model.state_dict(), os.path.join(final_dir, "retromae_full.pt"))
    tokenizer.save_pretrained(final_dir)

    metric_logger.finish()
    logger.info(f"Training complete. Encoder saved to: {final_dir}/encoder")


def main():
    cfg = parse_args_and_load_config(RetroMAEConfig, "RetroMAE asymmetric masked auto-encoder pipeline")
    run(cfg)


if __name__ == "__main__":
    main()
