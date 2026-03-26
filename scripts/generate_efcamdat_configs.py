#!/usr/bin/env python3
"""
Generate exhaustive EFCAMDAT configs: every model size × every applicable objective.

Usage:
    python scripts/generate_efcamdat_configs.py
"""

import os
from pathlib import Path

DATA_PATH = "/home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/norm-EFCAMDAT-ALL-CONCAT.csv"
OUT_ROOT = Path("configs/efcamdat")

# ---------------------------------------------------------------------------
# Model catalog — every size per family
# ---------------------------------------------------------------------------

MODELS = {
    # ── BERT ──────────────────────────────────────────────────────────────
    "bert-base": {
        "family": "bert", "hf_id": "bert-base-uncased", "params": "110M",
        "bs": 32, "accum": 1, "lr": 5e-5, "max_len": 512, "mask_prob": 0.15,
        "flash": False, "grad_ckpt": False,
    },
    "bert-large": {
        "family": "bert", "hf_id": "bert-large-uncased", "params": "340M",
        "bs": 16, "accum": 2, "lr": 3e-5, "max_len": 512, "mask_prob": 0.15,
        "flash": False, "grad_ckpt": False,
    },
    # ── RoBERTa ───────────────────────────────────────────────────────────
    "roberta-base": {
        "family": "roberta", "hf_id": "FacebookAI/roberta-base", "params": "125M",
        "bs": 32, "accum": 1, "lr": 5e-5, "max_len": 512, "mask_prob": 0.15,
        "flash": False, "grad_ckpt": False,
    },
    "roberta-large": {
        "family": "roberta", "hf_id": "FacebookAI/roberta-large", "params": "355M",
        "bs": 16, "accum": 2, "lr": 3e-5, "max_len": 512, "mask_prob": 0.15,
        "flash": False, "grad_ckpt": False,
    },
    # ── ModernBERT ────────────────────────────────────────────────────────
    "modernbert-base": {
        "family": "modernbert", "hf_id": "answerdotai/ModernBERT-base", "params": "149M",
        "bs": 16, "accum": 2, "lr": 1e-4, "max_len": 1024, "mask_prob": 0.30,
        "flash": True, "grad_ckpt": False,
    },
    "modernbert-large": {
        "family": "modernbert", "hf_id": "answerdotai/ModernBERT-large", "params": "395M",
        "bs": 8, "accum": 4, "lr": 5e-5, "max_len": 1024, "mask_prob": 0.30,
        "flash": True, "grad_ckpt": True,
    },
    # ── NomicBERT ─────────────────────────────────────────────────────────
    "nomic-bert": {
        "family": "nomic", "hf_id": "nomic-ai/nomic-bert-2048", "params": "137M",
        "bs": 32, "accum": 1, "lr": 5e-5, "max_len": 512, "mask_prob": 0.15,
        "flash": True, "grad_ckpt": False,
    },
    # ── ALBERT ────────────────────────────────────────────────────────────
    "albert-base-v2": {
        "family": "albert", "hf_id": "albert-base-v2", "params": "12M",
        "bs": 64, "accum": 1, "lr": 5e-5, "max_len": 512, "mask_prob": 0.15,
        "flash": False, "grad_ckpt": False,
    },
    "albert-large-v2": {
        "family": "albert", "hf_id": "albert-large-v2", "params": "18M",
        "bs": 64, "accum": 1, "lr": 5e-5, "max_len": 512, "mask_prob": 0.15,
        "flash": False, "grad_ckpt": False,
    },
    "albert-xlarge-v2": {
        "family": "albert", "hf_id": "albert-xlarge-v2", "params": "60M",
        "bs": 32, "accum": 1, "lr": 3e-5, "max_len": 512, "mask_prob": 0.15,
        "flash": False, "grad_ckpt": False,
    },
    "albert-xxlarge-v2": {
        "family": "albert", "hf_id": "albert-xxlarge-v2", "params": "235M",
        "bs": 16, "accum": 2, "lr": 2e-5, "max_len": 512, "mask_prob": 0.15,
        "flash": False, "grad_ckpt": True,
    },
    # ── DeBERTaV3 ─────────────────────────────────────────────────────────
    "debertav3-xsmall": {
        "family": "debertav3", "hf_id": "microsoft/deberta-v3-xsmall", "params": "22M",
        "bs": 64, "accum": 1, "lr": 5e-5, "max_len": 512, "mask_prob": 0.15,
        "flash": False, "grad_ckpt": False,
    },
    "debertav3-small": {
        "family": "debertav3", "hf_id": "microsoft/deberta-v3-small", "params": "44M",
        "bs": 48, "accum": 1, "lr": 5e-5, "max_len": 512, "mask_prob": 0.15,
        "flash": False, "grad_ckpt": False,
    },
    "debertav3-base": {
        "family": "debertav3", "hf_id": "microsoft/deberta-v3-base", "params": "86M",
        "bs": 32, "accum": 1, "lr": 5e-5, "max_len": 512, "mask_prob": 0.15,
        "flash": False, "grad_ckpt": False,
    },
    "debertav3-large": {
        "family": "debertav3", "hf_id": "microsoft/deberta-v3-large", "params": "304M",
        "bs": 8, "accum": 4, "lr": 3e-5, "max_len": 512, "mask_prob": 0.15,
        "flash": False, "grad_ckpt": True,
    },
    # ── ELECTRA (discriminator only — RTD objective) ──────────────────────
    "electra-small": {
        "family": "electra", "hf_id": "google/electra-small-discriminator", "params": "14M",
        "bs": 64, "accum": 1, "lr": 5e-5, "max_len": 512, "mask_prob": 0.15,
        "flash": False, "grad_ckpt": False,
        "generator": "google/electra-small-generator",
    },
    "electra-base": {
        "family": "electra", "hf_id": "google/electra-base-discriminator", "params": "110M",
        "bs": 32, "accum": 1, "lr": 5e-5, "max_len": 512, "mask_prob": 0.15,
        "flash": False, "grad_ckpt": False,
        "generator": "google/electra-base-generator",
    },
    "electra-large": {
        "family": "electra", "hf_id": "google/electra-large-discriminator", "params": "335M",
        "bs": 8, "accum": 4, "lr": 2e-5, "max_len": 512, "mask_prob": 0.15,
        "flash": False, "grad_ckpt": True,
        "generator": "google/electra-large-generator",
    },
}

# Which objectives apply to which families
# ELECTRA discriminators have NO MLM head — only RTD
MLM_FAMILIES = {"bert", "roberta", "modernbert", "nomic", "albert", "debertav3"}
WWM_FAMILIES = {"bert", "roberta", "modernbert", "nomic", "albert", "debertav3"}
RTD_FAMILIES = {"electra", "debertav3", "bert", "roberta", "modernbert"}
SPAN_FAMILIES = {"bert", "roberta", "modernbert", "nomic", "albert", "debertav3"}
RETROMAE_FAMILIES = {"bert", "roberta", "modernbert"}


def make_encoder_yaml(model_key, m, objective, extra=""):
    flash_line = "flash_attention: true" if m["flash"] else ""
    grad_ckpt_line = "gradient_checkpointing: true" if m["grad_ckpt"] else ""

    # For ModernBERT, use its native 30% masking; others use 15%
    mask_prob = m["mask_prob"]

    return f"""\
# {model_key} {objective.upper()} on EFCAMDAT ({m['params']} params)
model: {m['hf_id']}
objective: {objective}
mask_probability: {mask_prob}
max_length: {m['max_len']}
{flash_line}
seed: 42

train_file: {DATA_PATH}
text_column: text
val_split: 0.05

per_device_train_batch_size: {m['bs']}
per_device_eval_batch_size: {m['bs']}
gradient_accumulation_steps: {m['accum']}
num_train_epochs: 10
learning_rate: {m['lr']}
weight_decay: 0.01
warmup_ratio: 0.06
lr_scheduler_type: cosine
bf16: true
{grad_ckpt_line}

eval_strategy: epoch
save_strategy: epoch
save_total_limit: 3
logging_steps: 50

wandb_project: efcamdat-encoder
wandb_run_name: {model_key}-{objective}
{extra}""".strip() + "\n"


def make_rtd_yaml(model_key, m):
    flash_line = "flash_attention: true" if m["flash"] else ""
    grad_ckpt_line = "gradient_checkpointing: true" if m["grad_ckpt"] else ""

    # ELECTRA has explicit generator models; others auto-derive
    if "generator" in m:
        gen_line = f"generator_model: {m['generator']}"
    else:
        gen_line = "generator_size_fraction: 0.25"

    return f"""\
# {model_key} RTD on EFCAMDAT ({m['params']} params)
model: {m['hf_id']}
objective: rtd
mask_probability: {m['mask_prob']}
max_length: {m['max_len']}
{gen_line}
discriminator_weight: 50.0
{flash_line}
seed: 42

train_file: {DATA_PATH}
text_column: text
val_split: 0.05

per_device_train_batch_size: {m['bs']}
per_device_eval_batch_size: {m['bs']}
gradient_accumulation_steps: {m['accum']}
num_train_epochs: 10
learning_rate: {m['lr']}
weight_decay: 0.01
warmup_ratio: 0.06
lr_scheduler_type: cosine
bf16: true
{grad_ckpt_line}

eval_strategy: epoch
save_strategy: epoch
save_total_limit: 3
logging_steps: 50

wandb_project: efcamdat-encoder
wandb_run_name: {model_key}-rtd
""".strip() + "\n"


def make_retromae_yaml(model_key, m):
    flash_line = "flash_attention: true" if m["flash"] else ""
    grad_ckpt_line = "gradient_checkpointing: true" if m["grad_ckpt"] else ""

    return f"""\
# RetroMAE {model_key} on EFCAMDAT ({m['params']} params)
# Asymmetric masking: 15% encoder, 50% decoder
model: {m['hf_id']}
encoder_mask_ratio: 0.15
decoder_mask_ratio: 0.50
decoder_layers: 1
max_length: {m['max_len']}
{flash_line}
seed: 42

train_file: {DATA_PATH}
text_column: text
val_split: 0.05

per_device_train_batch_size: {m['bs']}
per_device_eval_batch_size: {m['bs']}
gradient_accumulation_steps: {m['accum']}
num_train_epochs: 10
learning_rate: {m['lr']}
weight_decay: 0.01
warmup_ratio: 0.06
lr_scheduler_type: cosine
bf16: true
{grad_ckpt_line}

logging_steps: 50
wandb_project: efcamdat-retromae
wandb_run_name: {model_key}-retromae
""".strip() + "\n"


def clean_yaml(text):
    """Remove blank lines from empty conditionals."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        if line.strip() == "":
            # Keep blank lines only if previous wasn't blank
            if cleaned and cleaned[-1].strip() != "":
                cleaned.append(line)
        else:
            cleaned.append(line)
    return "\n".join(cleaned)


def main():
    counts = {"mlm": 0, "wwm": 0, "rtd": 0, "span_corruption": 0, "retromae": 0}

    for model_key, m in MODELS.items():
        family = m["family"]

        # MLM
        if family in MLM_FAMILIES:
            path = OUT_ROOT / "encoder" / f"{model_key}-mlm.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(clean_yaml(make_encoder_yaml(model_key, m, "mlm")))
            counts["mlm"] += 1

        # WWM
        if family in WWM_FAMILIES:
            path = OUT_ROOT / "encoder" / f"{model_key}-wwm.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(clean_yaml(make_encoder_yaml(model_key, m, "wwm")))
            counts["wwm"] += 1

        # RTD
        if family in RTD_FAMILIES:
            path = OUT_ROOT / "encoder" / f"{model_key}-rtd.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(clean_yaml(make_rtd_yaml(model_key, m)))
            counts["rtd"] += 1

        # Span Corruption
        if family in SPAN_FAMILIES:
            extra = "mean_span_length: 3.0"
            path = OUT_ROOT / "encoder" / f"{model_key}-span-corruption.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(clean_yaml(make_encoder_yaml(model_key, m, "span_corruption", extra)))
            counts["span_corruption"] += 1

        # RetroMAE
        if family in RETROMAE_FAMILIES:
            path = OUT_ROOT / "retromae" / f"{model_key}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(clean_yaml(make_retromae_yaml(model_key, m)))
            counts["retromae"] += 1

    total = sum(counts.values())
    print(f"Generated {total} configs:")
    for obj, count in sorted(counts.items()):
        print(f"  {obj}: {count}")
    print(f"\nOutput: {OUT_ROOT}/")


if __name__ == "__main__":
    main()
