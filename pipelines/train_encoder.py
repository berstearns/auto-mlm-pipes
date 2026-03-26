#!/usr/bin/env python3
"""
Core encoder pre-training pipeline: MLM, WWM, RTD, MNTP, span corruption.

Supports all AutoModelForMaskedLM-compatible models:
  ModernBERT, NomicBERT, NeoBERT, DeBERTaV3, BERT, RoBERTa, ALBERT, M2-BERT, etc.

Usage:
    python -m pipelines.train_encoder --config configs/encoder/dummy-cpu.yaml
    python -m pipelines.train_encoder --config configs/encoder/bert-mlm.yaml --learning_rate 3e-5
"""

import datetime
import json
import logging
import math
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .config import (
    EncoderConfig,
    build_parser,
    config_to_dict,
    dump_resolved_config,
    load_config,
    parse_args_and_load_config,
)
from .data_utils import load_data
from .logging_utils import MetricLogger, setup_logging_env

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Tokenization for encoder pre-training
# ---------------------------------------------------------------------------

def tokenize_for_encoder(datasets_dict: dict, tokenizer, max_length: int, num_proc: int = 4) -> dict:
    """Tokenize text for encoder pre-training. Each example is independent (no blocking)."""
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            max_length=max_length,
            truncation=True,
            padding=False,
            return_special_tokens_mask=True,
        )

    result = {}
    for split, ds in datasets_dict.items():
        if len(ds) == 0:
            result[split] = ds
            continue
        result[split] = ds.map(
            tokenize_fn, batched=True, num_proc=num_proc,
            remove_columns=ds.column_names,
        )
    return result


# ---------------------------------------------------------------------------
# Custom collators
# ---------------------------------------------------------------------------

class SpanCorruptionCollator:
    """T5-style span corruption collator.

    Masks contiguous spans with sentinel tokens. Each span is replaced by one sentinel.
    """

    def __init__(self, tokenizer, mask_probability: float = 0.15, mean_span_length: float = 3.0):
        self.tokenizer = tokenizer
        self.mask_probability = mask_probability
        self.mean_span_length = mean_span_length
        self.pad_token_id = tokenizer.pad_token_id or 0
        self.mask_token_id = tokenizer.mask_token_id or tokenizer.convert_tokens_to_ids("[MASK]")

    def __call__(self, examples):
        batch_input_ids = []
        batch_labels = []

        for ex in examples:
            input_ids = ex["input_ids"]
            special_mask = ex.get("special_tokens_mask", [0] * len(input_ids))
            masked_ids, labels = self._corrupt_spans(input_ids, special_mask)
            batch_input_ids.append(masked_ids)
            batch_labels.append(labels)

        # Pad to max length in batch
        max_len = max(len(ids) for ids in batch_input_ids)
        padded_ids = []
        padded_labels = []
        attention_masks = []
        for ids, labs in zip(batch_input_ids, batch_labels):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [self.pad_token_id] * pad_len)
            padded_labels.append(labs + [-100] * pad_len)
            attention_masks.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

    def _corrupt_spans(self, input_ids, special_mask):
        """Select spans to mask, replace each with a single [MASK] sentinel."""
        length = len(input_ids)
        num_to_mask = max(1, int(length * self.mask_probability))

        # Select span start positions
        maskable = [i for i in range(length) if not special_mask[i]]
        if not maskable:
            return list(input_ids), [-100] * length

        spans = []
        masked_count = 0
        attempts = 0
        while masked_count < num_to_mask and attempts < 100:
            start = random.choice(maskable)
            span_len = max(1, int(np.random.geometric(1.0 / self.mean_span_length)))
            span_len = min(span_len, length - start)
            # Check overlap
            overlaps = any(s <= start < s + l or s <= start + span_len - 1 < s + l for s, l in spans)
            if not overlaps:
                spans.append((start, span_len))
                masked_count += span_len
            attempts += 1

        spans.sort(key=lambda x: x[0])

        # Build corrupted sequence: replace each span with [MASK]
        corrupted = []
        labels = []
        i = 0
        span_idx = 0
        while i < length:
            if span_idx < len(spans) and i == spans[span_idx][0]:
                start, span_len = spans[span_idx]
                corrupted.append(self.mask_token_id)
                labels.append(input_ids[start])  # predict first token of span
                i += span_len
                span_idx += 1
            else:
                corrupted.append(input_ids[i])
                labels.append(-100)
                i += 1

        return corrupted, labels


class MNTPCollator:
    """Masked Next Token Prediction collator (LLM2Vec-style).

    Masks random tokens and sets labels to the NEXT token (shifted by 1).
    """

    def __init__(self, tokenizer, mlm_probability: float = 0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.pad_token_id = tokenizer.pad_token_id or 0
        self.mask_token_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else 0

    def __call__(self, examples):
        batch_input_ids = []
        batch_labels = []

        for ex in examples:
            input_ids = list(ex["input_ids"])
            length = len(input_ids)

            # Create mask positions (exclude last token — no next token)
            maskable = list(range(length - 1))
            num_to_mask = max(1, int(len(maskable) * self.mlm_probability))
            mask_positions = set(random.sample(maskable, min(num_to_mask, len(maskable))))

            masked_ids = list(input_ids)
            labels = [-100] * length

            for pos in mask_positions:
                masked_ids[pos] = self.mask_token_id
                labels[pos] = input_ids[pos + 1]  # predict NEXT token

            batch_input_ids.append(masked_ids)
            batch_labels.append(labels)

        max_len = max(len(ids) for ids in batch_input_ids)
        padded_ids = []
        padded_labels = []
        attention_masks = []
        for ids, labs in zip(batch_input_ids, batch_labels):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [self.pad_token_id] * pad_len)
            padded_labels.append(labs + [-100] * pad_len)
            attention_masks.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# RTD (ELECTRA-style) components
# ---------------------------------------------------------------------------

class ElectraForPreTraining(nn.Module):
    """Wraps generator (small MLM) + discriminator (main encoder) for RTD training."""

    def __init__(self, generator, discriminator, discriminator_weight: float = 50.0):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.discriminator_weight = discriminator_weight
        # Discriminator head: binary classification per token
        hidden_size = discriminator.config.hidden_size
        self.disc_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Step 1: Run generator with MLM
        gen_outputs = self.generator(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        gen_loss = gen_outputs.loss
        gen_logits = gen_outputs.logits

        # Step 2: Create corrupted input by sampling from generator predictions
        with torch.no_grad():
            gen_preds = gen_logits.argmax(dim=-1)
            mask_positions = labels != -100
            corrupted_ids = input_ids.clone()
            corrupted_ids[mask_positions] = gen_preds[mask_positions]
            # Labels for discriminator: 1 where token was replaced, 0 where original
            disc_labels = (corrupted_ids != input_ids).float()

        # Step 3: Run discriminator on corrupted input
        disc_outputs = self.discriminator(input_ids=corrupted_ids, attention_mask=attention_mask)
        disc_hidden = disc_outputs.last_hidden_state
        disc_logits = self.disc_head(disc_hidden).squeeze(-1)

        # Step 4: Discriminator loss (binary cross-entropy on all tokens)
        disc_loss = F.binary_cross_entropy_with_logits(
            disc_logits, disc_labels, reduction="none"
        )
        if attention_mask is not None:
            disc_loss = (disc_loss * attention_mask).sum() / attention_mask.sum()
        else:
            disc_loss = disc_loss.mean()

        total_loss = gen_loss + self.discriminator_weight * disc_loss

        return type(gen_outputs)(
            loss=total_loss,
            logits=gen_logits,
        )


class ElectraTrainer(Trainer):
    """Custom Trainer for ELECTRA-style RTD training."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Objective setup functions
# ---------------------------------------------------------------------------

def setup_mlm(cfg: EncoderConfig, tokenizer):
    """Standard Masked Language Modeling."""
    model = _load_mlm_model(cfg)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=cfg.mask_probability,
    )
    return model, collator, Trainer


def setup_wwm(cfg: EncoderConfig, tokenizer):
    """Whole Word Masking."""
    model = _load_mlm_model(cfg)
    collator = DataCollatorForWholeWordMask(
        tokenizer=tokenizer,
        mlm_probability=cfg.mask_probability,
    )
    return model, collator, Trainer


def setup_rtd(cfg: EncoderConfig, tokenizer):
    """Replaced Token Detection (ELECTRA-style)."""
    # Load discriminator (main model)
    discriminator = _load_base_model(cfg)

    # Load or derive generator (smaller MLM)
    gen_model_name = cfg.generator_model or cfg.model.model
    if cfg.model.from_scratch and not cfg.generator_model:
        # Create smaller config
        disc_config = AutoConfig.from_pretrained(cfg.model.model, trust_remote_code=True)
        gen_config = AutoConfig.from_pretrained(cfg.model.model, trust_remote_code=True)
        gen_config.num_hidden_layers = max(1, int(disc_config.num_hidden_layers * cfg.generator_size_fraction))
        gen_config.intermediate_size = max(256, int(disc_config.intermediate_size * cfg.generator_size_fraction))
        generator = AutoModelForMaskedLM.from_config(gen_config, trust_remote_code=True)
    else:
        generator = AutoModelForMaskedLM.from_pretrained(
            gen_model_name,
            torch_dtype=_get_dtype(cfg),
            trust_remote_code=cfg.model.trust_remote_code,
        )

    model = ElectraForPreTraining(generator, discriminator, cfg.discriminator_weight)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=cfg.mask_probability,
    )
    return model, collator, ElectraTrainer


def setup_mntp(cfg: EncoderConfig, tokenizer):
    """Masked Next Token Prediction (LLM2Vec-style decoder-to-encoder conversion)."""
    model_name = cfg.causal_model or cfg.model.model
    attn_impl = {"attn_implementation": "flash_attention_2"} if cfg.model.flash_attention else {}

    if cfg.model.from_scratch:
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=cfg.model.trust_remote_code)
        model_config.is_decoder = False
        model = AutoModelForCausalLM.from_config(model_config, **attn_impl, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=_get_dtype(cfg),
            trust_remote_code=cfg.model.trust_remote_code,
            **attn_impl,
        )
        model.config.is_decoder = False

    # Ensure mask token exists
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        model.resize_token_embeddings(len(tokenizer))

    collator = MNTPCollator(tokenizer=tokenizer, mlm_probability=cfg.mask_probability)
    return model, collator, Trainer


def setup_span_corruption(cfg: EncoderConfig, tokenizer):
    """T5-style span corruption."""
    model = _load_mlm_model(cfg)
    collator = SpanCorruptionCollator(
        tokenizer=tokenizer,
        mask_probability=cfg.mask_probability,
        mean_span_length=cfg.mean_span_length,
    )
    return model, collator, Trainer


OBJECTIVES = {
    "mlm": setup_mlm,
    "wwm": setup_wwm,
    "rtd": setup_rtd,
    "mntp": setup_mntp,
    "span_corruption": setup_span_corruption,
}


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _get_dtype(cfg: EncoderConfig):
    if cfg.training.bf16:
        return torch.bfloat16
    elif cfg.training.fp16:
        return torch.float16
    return torch.float32


def _load_mlm_model(cfg: EncoderConfig):
    """Load AutoModelForMaskedLM (from scratch or pretrained)."""
    attn_impl = {"attn_implementation": "flash_attention_2"} if cfg.model.flash_attention else {}

    if cfg.model.from_scratch:
        logger.info("Training from scratch")
        model_config = AutoConfig.from_pretrained(
            cfg.model.model, trust_remote_code=cfg.model.trust_remote_code
        )
        return AutoModelForMaskedLM.from_config(model_config, **attn_impl, trust_remote_code=True)
    else:
        logger.info(f"Loading pretrained: {cfg.model.model}")
        return AutoModelForMaskedLM.from_pretrained(
            cfg.model.model,
            torch_dtype=_get_dtype(cfg),
            trust_remote_code=cfg.model.trust_remote_code,
            **attn_impl,
        )


def _load_base_model(cfg: EncoderConfig):
    """Load AutoModel (backbone without head)."""
    attn_impl = {"attn_implementation": "flash_attention_2"} if cfg.model.flash_attention else {}

    if cfg.model.from_scratch:
        model_config = AutoConfig.from_pretrained(
            cfg.model.model, trust_remote_code=cfg.model.trust_remote_code
        )
        return AutoModel.from_config(model_config, **attn_impl, trust_remote_code=True)
    else:
        return AutoModel.from_pretrained(
            cfg.model.model,
            torch_dtype=_get_dtype(cfg),
            trust_remote_code=cfg.model.trust_remote_code,
            **attn_impl,
        )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class TopKCheckpointCallback:
    """Keep only top-K checkpoints by eval loss. Used with Trainer callbacks."""
    pass  # Integrated via Trainer's save_total_limit for now


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_single(cfg: EncoderConfig):
    """Run a single training phase."""
    set_seed(cfg.seed)
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    dump_resolved_config(cfg, output_dir)

    flat = config_to_dict(cfg)
    logger.info(f"Objective: {cfg.objective}")
    logger.info(f"Model: {cfg.model.model}")
    logger.info(f"Output: {output_dir}")

    # --- Tokenizer ---
    tok_name = cfg.model.tokenizer or cfg.model.model
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=cfg.model.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Objective setup ---
    if cfg.objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective: {cfg.objective}. Choose from: {list(OBJECTIVES.keys())}")

    model, collator, trainer_cls = OBJECTIVES[cfg.objective](cfg, tokenizer)

    # Resize embeddings if needed
    if hasattr(model, 'resize_token_embeddings'):
        model.resize_token_embeddings(len(tokenizer))

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # --- Data ---
    raw_datasets = load_data(cfg)
    processed = tokenize_for_encoder(raw_datasets, tokenizer, cfg.max_length, cfg.data.num_proc)
    has_validation = "validation" in processed and len(processed["validation"]) > 0

    logger.info(f"Train examples: {len(processed['train']):,}")
    if has_validation:
        logger.info(f"Val examples:   {len(processed['validation']):,}")

    # --- Logging backend ---
    report_to = setup_logging_env(cfg)

    # --- Training args ---
    eval_strategy = cfg.checkpoint.eval_strategy if has_validation else "no"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        max_steps=cfg.training.max_steps,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        max_grad_norm=cfg.training.max_grad_norm,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        optim=cfg.training.optim,
        logging_steps=cfg.logging.logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=cfg.checkpoint.eval_steps if eval_strategy == "steps" else None,
        save_strategy=cfg.checkpoint.save_strategy,
        save_steps=cfg.checkpoint.save_steps,
        save_total_limit=cfg.checkpoint.save_total_limit,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        report_to=report_to,
        seed=cfg.seed,
        run_name=cfg.logging.wandb_run_name,
        load_best_model_at_end=has_validation,
        metric_for_best_model="eval_loss" if has_validation else None,
    )

    # --- Trainer ---
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=processed["train"],
        eval_dataset=processed["validation"] if has_validation else None,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # --- Train ---
    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=cfg.checkpoint.resume_from_checkpoint)

    # --- Save ---
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Model saved to: {final_dir}")

    # --- Evaluate ---
    if has_validation:
        eval_results = trainer.evaluate()
        logger.info(f"Eval results: {eval_results}")
        with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
            json.dump(eval_results, f, indent=2)

    # --- Save metrics ---
    metrics = train_result.metrics
    with open(os.path.join(output_dir, "train_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Training complete.")
    return train_result


def run_phased(cfg: EncoderConfig):
    """Run multi-phase training (progressive sequence length, etc.)."""
    if not cfg.phases:
        return run_single(cfg)

    base_output = cfg.output_dir
    model_path = cfg.model.model

    for i, phase in enumerate(cfg.phases):
        phase_name = phase.get("name", f"phase{i + 1}")
        logger.info(f"=== Phase {i + 1}/{len(cfg.phases)}: {phase_name} ===")

        # Apply phase overrides
        phase_cfg = _apply_phase_overrides(cfg, phase)
        phase_cfg.model.model = model_path
        phase_cfg.output_dir = os.path.join(base_output, phase_name)

        run_single(phase_cfg)

        # Next phase loads from this phase's output
        model_path = os.path.join(phase_cfg.output_dir, "final")

    logger.info("All phases complete.")


def _apply_phase_overrides(base_cfg: EncoderConfig, phase: dict) -> EncoderConfig:
    """Create a new config with phase-specific overrides applied."""
    flat = config_to_dict(base_cfg)
    for k, v in phase.items():
        if k != "name" and k in flat:
            flat[k] = v
    return load_config(None, flat, EncoderConfig)


def run(cfg: EncoderConfig):
    """Main entry point."""
    if cfg.phases:
        run_phased(cfg)
    else:
        run_single(cfg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    cfg = parse_args_and_load_config(EncoderConfig, "Encoder pre-training pipeline")
    run(cfg)


if __name__ == "__main__":
    main()
