# ColBERT EFCAMDAT configs

ColBERT requires query-document paired data (JSONL/CSV with query, positive, negative columns).
The raw EFCAMDAT CSV contains unpaired learner text, so a data preparation step is needed first.

Options:
1. Generate synthetic queries from learner text (e.g., using an LLM)
2. Use CEFR level as supervision to create proficiency-discriminative pairs
3. Use L1 as supervision for L1-discriminative pairs

Once pairs are generated, create configs pointing to the paired JSONL file.
