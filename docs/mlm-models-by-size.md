# MLM Training Models — sorted by size

All 15 encoder models trained on EFCAMDAT-ALL-CONCAT via `scripts/train-all-mlm.sh`.
VRAM estimates assume max_length=512, bf16, AdamW optimizer.

| # | Config | HuggingFace Model | Params | Batch Size | Est. VRAM |
|---|--------|-------------------|--------|------------|-----------|
| 1 | albert-base-v2-mlm | `albert-base-v2` | 12M | 64 | ~1.5 GB |
| 2 | albert-large-v2-mlm | `albert-large-v2` | 18M | 64 | ~2 GB |
| 3 | debertav3-xsmall-mlm | `microsoft/deberta-v3-xsmall` | 22M | 64 | ~2.5 GB |
| 4 | debertav3-small-mlm | `microsoft/deberta-v3-small` | 44M | 48 | ~3.5 GB |
| 5 | albert-xlarge-v2-mlm | `albert-xlarge-v2` | 60M | 32 | ~4 GB |
| 6 | debertav3-base-mlm | `microsoft/deberta-v3-base` | 86M | 32 | ~6 GB |
| 7 | bert-base-mlm | `bert-base-uncased` | 110M | 32 | ~7 GB |
| 8 | roberta-base-mlm | `FacebookAI/roberta-base` | 125M | 32 | ~8 GB |
| 9 | nomic-bert-mlm | `nomic-ai/nomic-bert-2048` | 137M | 32 | ~9 GB |
| 10 | modernbert-base-mlm | `answerdotai/ModernBERT-base` | 149M | 16 | ~8 GB |
| 11 | albert-xxlarge-v2-mlm | `albert-xxlarge-v2` | 235M | 16 | ~12 GB |
| 12 | debertav3-large-mlm | `microsoft/deberta-v3-large` | 304M | 8 | ~14 GB |
| 13 | bert-large-mlm | `bert-large-uncased` | 340M | 16 | ~18 GB |
| 14 | roberta-large-mlm | `FacebookAI/roberta-large` | 355M | 16 | ~19 GB |
| 15 | modernbert-large-mlm | `answerdotai/ModernBERT-large` | 395M | 8 | ~16 GB |
