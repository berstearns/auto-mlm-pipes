"""
Shared data loading utilities for all auto-mlm-pipes pipelines.

Supports: .txt files, .csv files, folders of .txt, HuggingFace datasets,
and paired query-document data (JSONL/CSV) for retrieval pipelines.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-text loaders
# ---------------------------------------------------------------------------

def load_text_file(path: str) -> Dataset:
    """Load .txt file, one document per line."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return Dataset.from_dict({"text": lines})


def load_text_folder(path: str) -> Dataset:
    """Load all .txt files from a directory."""
    p = Path(path)
    texts = []
    for fp in sorted(p.glob("*.txt")):
        with open(fp, "r", encoding="utf-8") as f:
            texts.extend([line.strip() for line in f if line.strip()])
    return Dataset.from_dict({"text": texts})


def load_csv_file(path: str, text_column: str = "text", label_column: Optional[str] = None) -> Dataset:
    """Load a .csv file, extracting text and optional label columns."""
    import pandas as pd
    df = pd.read_csv(path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not in {path}. Available: {list(df.columns)}")

    ds_dict = {"text": df[text_column].dropna().astype(str).tolist()}
    if label_column and label_column in df.columns:
        ds_dict["label"] = df[label_column].tolist()

    return Dataset.from_dict(ds_dict)


def load_data(cfg, seed: int = 42) -> dict:
    """Load dataset from config. Returns {"train": Dataset, "validation": Dataset}.

    cfg can be a dataclass or dict. Accesses: train_file, validation_file,
    dataset_name, dataset_config_name, text_column, val_split, seed.
    """
    # Support both dataclass and dict access
    def _get(key, default=None):
        if hasattr(cfg, 'data'):
            return getattr(cfg.data, key, default)
        elif isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    train_file = _get("train_file")
    validation_file = _get("validation_file")
    dataset_name = _get("dataset_name")
    dataset_config = _get("dataset_config_name")
    val_split = _get("val_split", 0.1)
    text_column = _get("text_column", "text")
    label_column = _get("label_column")

    _seed = getattr(cfg, 'seed', seed) if hasattr(cfg, 'seed') else seed

    if dataset_name:
        raw = load_dataset(dataset_name, dataset_config)
        if "validation" not in raw and "test" not in raw:
            if val_split > 0:
                split = raw["train"].train_test_split(test_size=val_split, seed=_seed)
                return {"train": split["train"], "validation": split["test"]}
            return {"train": raw["train"], "validation": Dataset.from_dict({"text": []})}
        val_key = "validation" if "validation" in raw else "test"
        return {"train": raw["train"], "validation": raw[val_key]}

    if train_file:
        p = Path(train_file)
        if p.is_dir():
            ds = load_text_folder(train_file)
        elif p.suffix == ".csv":
            ds = load_csv_file(train_file, text_column, label_column)
        else:
            ds = load_text_file(train_file)

        if validation_file:
            vp = Path(validation_file)
            if vp.is_dir():
                val_ds = load_text_folder(validation_file)
            elif vp.suffix == ".csv":
                val_ds = load_csv_file(validation_file, text_column, label_column)
            else:
                val_ds = load_text_file(validation_file)
            return {"train": ds, "validation": val_ds}
        else:
            if val_split > 0 and len(ds) > 1:
                split = ds.train_test_split(test_size=val_split, seed=_seed)
                return {"train": split["train"], "validation": split["test"]}
            return {"train": ds, "validation": Dataset.from_dict({"text": []})}

    raise ValueError("Provide --train_file or --dataset_name in config/CLI.")


# ---------------------------------------------------------------------------
# Paired data loaders (ColBERT, SPLADE)
# ---------------------------------------------------------------------------

def load_pairs_jsonl(path: str, query_col: str = "query",
                     positive_col: str = "positive",
                     negative_col: str = "negative") -> Dataset:
    """Load query-document pairs from JSONL. Each line: {query, positive, negative}."""
    queries, positives, negatives = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            queries.append(obj[query_col])
            positives.append(obj[positive_col])
            neg = obj.get(negative_col, "")
            negatives.append(neg if isinstance(neg, str) else json.dumps(neg))

    return Dataset.from_dict({
        "query": queries,
        "positive": positives,
        "negative": negatives,
    })


def load_pairs_csv(path: str, query_col: str = "query",
                   positive_col: str = "positive",
                   negative_col: str = "negative") -> Dataset:
    """Load query-document pairs from CSV."""
    import pandas as pd
    df = pd.read_csv(path)
    for col in [query_col, positive_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not in {path}. Available: {list(df.columns)}")

    ds_dict = {
        "query": df[query_col].astype(str).tolist(),
        "positive": df[positive_col].astype(str).tolist(),
    }
    if negative_col in df.columns:
        ds_dict["negative"] = df[negative_col].astype(str).tolist()
    else:
        ds_dict["negative"] = [""] * len(df)

    return Dataset.from_dict(ds_dict)


def load_pairs_data(cfg, seed: int = 42) -> dict:
    """Load paired data for retrieval pipelines. Returns {"train": Dataset, "validation": Dataset}."""
    def _get(key, default=None):
        if hasattr(cfg, 'data'):
            return getattr(cfg.data, key, default)
        elif isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    train_file = _get("train_file")
    validation_file = _get("validation_file")
    val_split = _get("val_split", 0.1)
    query_col = _get("query_column", "query")
    positive_col = _get("positive_column", "positive")
    negative_col = _get("negative_column", "negative")

    _seed = getattr(cfg, 'seed', seed) if hasattr(cfg, 'seed') else seed

    if not train_file:
        raise ValueError("Paired data requires --train_file (JSONL or CSV).")

    p = Path(train_file)
    if p.suffix in (".jsonl", ".json"):
        ds = load_pairs_jsonl(train_file, query_col, positive_col, negative_col)
    elif p.suffix == ".csv":
        ds = load_pairs_csv(train_file, query_col, positive_col, negative_col)
    else:
        raise ValueError(f"Paired data must be .jsonl or .csv, got: {p.suffix}")

    if validation_file:
        vp = Path(validation_file)
        if vp.suffix in (".jsonl", ".json"):
            val_ds = load_pairs_jsonl(validation_file, query_col, positive_col, negative_col)
        else:
            val_ds = load_pairs_csv(validation_file, query_col, positive_col, negative_col)
        return {"train": ds, "validation": val_ds}
    else:
        if val_split > 0 and len(ds) > 1:
            split = ds.train_test_split(test_size=val_split, seed=_seed)
            return {"train": split["train"], "validation": split["test"]}
        return {"train": ds, "validation": Dataset.from_dict({"query": [], "positive": [], "negative": []})}
