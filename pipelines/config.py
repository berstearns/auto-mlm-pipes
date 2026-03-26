"""
Shared dataclass-based configuration system for all auto-mlm-pipes pipelines.

Config merge order: DEFAULTS (dataclass defaults) < YAML < CLI overrides.
Supports flat YAML keys (backward-compatible with axolotl-style configs).
"""

import argparse
import datetime
import os
import yaml
from dataclasses import dataclass, field, fields, asdict
from typing import Optional, Any, get_type_hints


# ---------------------------------------------------------------------------
# Shared sub-config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Model selection and initialization."""
    model: str = "bert-base-uncased"
    tokenizer: Optional[str] = None          # defaults to model if None
    from_scratch: bool = False
    flash_attention: bool = False
    torch_compile: bool = False
    trust_remote_code: bool = True


@dataclass
class DataConfig:
    """Data source and loading."""
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    dataset_name: Optional[str] = None       # HuggingFace dataset
    dataset_config_name: Optional[str] = None
    text_column: str = "text"
    label_column: Optional[str] = None
    val_split: float = 0.1
    num_proc: int = 4
    # Pair data (ColBERT, SPLADE)
    query_column: str = "query"
    positive_column: str = "positive"
    negative_column: str = "negative"


@dataclass
class TrainingConfig:
    """Core training hyperparameters."""
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    max_steps: int = -1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = False
    optim: str = "adamw_torch"
    dataloader_num_workers: int = 2


@dataclass
class LoggingConfig:
    """Logging and experiment tracking."""
    report_to: str = "wandb"                 # wandb, tensorboard, aim, none
    log_dir: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    logging_steps: int = 10


@dataclass
class CheckpointConfig:
    """Evaluation and checkpoint saving."""
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 5
    top_k_checkpoints: int = 5
    resume_from_checkpoint: Optional[str] = None


# ---------------------------------------------------------------------------
# Pipeline-specific composed configs
# ---------------------------------------------------------------------------

@dataclass
class EncoderConfig:
    """Core encoder pipeline: MLM, WWM, RTD, MNTP, span corruption."""
    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    # Encoder-specific
    objective: str = "mlm"                   # mlm, wwm, rtd, mntp, span_corruption
    mask_probability: float = 0.15
    max_length: int = 512
    seed: int = 42
    output_dir: Optional[str] = None

    # RTD (ELECTRA) specific
    generator_model: Optional[str] = None    # separate generator model, or auto-derive
    generator_size_fraction: float = 0.25    # fraction of discriminator size
    discriminator_weight: float = 50.0       # weight for discriminator loss

    # MNTP (LLM2Vec) specific
    causal_model: Optional[str] = None       # the decoder model to convert

    # Span corruption (T5-style)
    mean_span_length: float = 3.0

    # Multi-phase training
    phases: Optional[list] = None


@dataclass
class ColBERTConfig:
    """ColBERT late-interaction pipeline."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    dim: int = 128                           # projection dimension per token
    doc_maxlen: int = 256
    query_maxlen: int = 32
    nbits: int = 2                           # compression bits (1, 2, 4)
    use_ib_negatives: bool = True            # in-batch negatives
    num_negatives: int = 7                   # hard negatives per query
    temperature: float = 0.05
    similarity: str = "cosine"               # cosine or l2
    seed: int = 42
    output_dir: Optional[str] = None


@dataclass
class GLMConfig:
    """GLM autoregressive blank infilling pipeline."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    mask_ratio: float = 0.15                 # fraction of tokens to blank
    max_length: int = 512
    avg_span_length: int = 3                 # average blank span length
    shuffle_spans: bool = True               # span order permutation
    seed: int = 42
    output_dir: Optional[str] = None


@dataclass
class SPLADEConfig:
    """SPLADE sparse expansion pipeline."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    lambda_d: float = 0.0008                # document sparsity regularization
    lambda_q: float = 0.0006                # query sparsity regularization
    max_length: int = 256
    num_negatives: int = 7
    distillation_weight: float = 1.0         # weight for knowledge distillation loss
    teacher_model: Optional[str] = None      # cross-encoder teacher for distillation
    temperature: float = 0.05
    seed: int = 42
    output_dir: Optional[str] = None


@dataclass
class RetroMAEConfig:
    """RetroMAE asymmetric masked auto-encoder pipeline."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    encoder_mask_ratio: float = 0.15         # low masking for encoder
    decoder_mask_ratio: float = 0.50         # high masking for decoder
    decoder_layers: int = 1                  # lightweight decoder
    decoder_hidden_dim: int = 0              # 0 = same as encoder hidden
    max_length: int = 512
    seed: int = 42
    output_dir: Optional[str] = None


# ---------------------------------------------------------------------------
# Config loading utilities
# ---------------------------------------------------------------------------

def _build_flat_key_map(config_cls):
    """Build a mapping from flat YAML keys to (sub_field_name, key_name).

    E.g., for EncoderConfig:
        "model" -> ("model", "model")       # ModelConfig.model
        "learning_rate" -> ("training", "learning_rate")
        "mask_probability" -> (None, "mask_probability")  # top-level
    """
    flat_map = {}
    for f in fields(config_cls):
        if hasattr(f.default_factory, '__call__') if hasattr(f, 'default_factory') else False:
            # This is a sub-dataclass field
            try:
                sub_cls = f.default_factory()
                for sf in fields(sub_cls):
                    if sf.name not in flat_map:
                        flat_map[sf.name] = (f.name, sf.name)
            except TypeError:
                pass
        else:
            # Top-level field
            flat_map[f.name] = (None, f.name)

    # Also map top-level fields from the config_cls itself (non-sub-dataclass)
    for f in fields(config_cls):
        if f.name not in flat_map:
            flat_map[f.name] = (None, f.name)
        # Override: top-level fields always win in flat mode
        if not any(hasattr(getattr(f, 'default_factory', None), '__call__') for _ in [0]):
            pass

    return flat_map


def _get_field_type(config_cls, field_name, sub_field_name=None):
    """Get the expected type for a field."""
    if sub_field_name:
        for f in fields(config_cls):
            if f.name == field_name:
                sub_cls = f.default_factory()
                for sf in fields(sub_cls):
                    if sf.name == sub_field_name:
                        return sf.type
    else:
        for f in fields(config_cls):
            if f.name == field_name:
                return f.type
    return None


def _coerce_value(value, target_type, field_name=""):
    """Coerce a value to the target type. Handles YAML misparses like '5e-5' as str."""
    if value is None:
        return None

    # Handle Optional[X] -> extract X
    origin = getattr(target_type, '__origin__', None)
    if origin is type(None):
        return None
    # For Optional (Union[X, None]), try the non-None type
    args = getattr(target_type, '__args__', None)
    if args and type(None) in args:
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            target_type = non_none[0]

    if target_type in (str, 'str') or target_type is str:
        return str(value) if value is not None else None
    if target_type in (int, 'int') or target_type is int:
        if isinstance(value, int):
            return value
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return value
    if target_type in (float, 'float') or target_type is float:
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    if target_type in (bool, 'bool') or target_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes')
        return bool(value)

    return value


def _flat_dict_to_nested(flat_dict: dict, config_cls) -> dict:
    """Convert a flat config dict to nested structure matching the config dataclass."""
    flat_map = _build_flat_key_map(config_cls)
    nested = {}
    sub_dicts = {}

    for key, value in flat_dict.items():
        if key in flat_map:
            parent, field_name = flat_map[key]
            if parent is None:
                # Top-level field
                nested[field_name] = value
            else:
                # Sub-dataclass field
                if parent not in sub_dicts:
                    sub_dicts[parent] = {}
                sub_dicts[parent][field_name] = value
        else:
            # Unknown key -> keep at top level
            nested[key] = value

    nested.update(sub_dicts)
    return nested


def _instantiate_config(nested_dict: dict, config_cls):
    """Instantiate a config dataclass from a nested dict, with type coercion."""
    kwargs = {}
    for f in fields(config_cls):
        if f.name in nested_dict:
            val = nested_dict[f.name]
            # Check if this field is a sub-dataclass
            if isinstance(val, dict) and hasattr(f.default_factory, '__call__'):
                sub_cls = type(f.default_factory())
                sub_kwargs = {}
                for sf in fields(sub_cls):
                    if sf.name in val:
                        sub_kwargs[sf.name] = _coerce_value(val[sf.name], sf.type, sf.name)
                kwargs[f.name] = sub_cls(**sub_kwargs)
            else:
                kwargs[f.name] = _coerce_value(val, f.type, f.name)

    return config_cls(**kwargs)


def load_config(config_path: Optional[str], cli_overrides: dict, config_cls=EncoderConfig):
    """Load config from YAML + CLI overrides into a dataclass.

    Merge order: dataclass defaults < YAML file < CLI overrides.
    Supports both flat and nested YAML formats.
    """
    # Start with empty overrides
    yaml_dict = {}
    if config_path:
        with open(config_path) as f:
            yaml_dict = yaml.safe_load(f) or {}

    # Merge CLI overrides into yaml_dict (CLI wins)
    for k, v in cli_overrides.items():
        if v is not None:
            yaml_dict[k] = v

    # Check if YAML is already nested or flat
    is_nested = any(
        isinstance(v, dict) and k in {f.name for f in fields(config_cls)}
        for k, v in yaml_dict.items()
    )

    if not is_nested:
        nested = _flat_dict_to_nested(yaml_dict, config_cls)
    else:
        # Apply CLI flat overrides into nested structure
        flat_overrides = {k: v for k, v in yaml_dict.items() if not isinstance(v, dict)}
        nested_parts = {k: v for k, v in yaml_dict.items() if isinstance(v, dict)}
        if flat_overrides:
            extra_nested = _flat_dict_to_nested(flat_overrides, config_cls)
            # Merge: nested YAML parts + flat-resolved parts
            for k, v in extra_nested.items():
                if isinstance(v, dict) and k in nested_parts:
                    nested_parts[k].update(v)
                else:
                    nested_parts[k] = v
        nested = nested_parts

    return _instantiate_config(nested, config_cls)


def config_to_dict(cfg) -> dict:
    """Flatten a config dataclass to a plain dict (for serialization, Trainer args, etc.)."""
    result = {}
    for f in fields(cfg):
        val = getattr(cfg, f.name)
        if hasattr(val, '__dataclass_fields__'):
            for sf in fields(val):
                result[sf.name] = getattr(val, sf.name)
        else:
            result[f.name] = val
    return result


def dump_resolved_config(cfg, output_dir: str):
    """Save resolved config as YAML to output_dir/resolved_config.yaml."""
    os.makedirs(output_dir, exist_ok=True)
    flat = config_to_dict(cfg)
    # Remove non-serializable values
    clean = {}
    for k, v in flat.items():
        if v is not None and not callable(v):
            clean[k] = v
    path = os.path.join(output_dir, "resolved_config.yaml")
    with open(path, "w") as f:
        yaml.dump(clean, f, default_flow_style=False)
    return path


# ---------------------------------------------------------------------------
# CLI argparse builder
# ---------------------------------------------------------------------------

def build_parser(config_cls=EncoderConfig, description="") -> argparse.ArgumentParser:
    """Build argparse from config dataclass fields. All args default to None for merge."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    seen = set()
    # Gather all fields (sub-dataclass fields + top-level)
    for f in fields(config_cls):
        val = f.default if f.default is not f.default_factory else None
        if hasattr(f, 'default_factory') and f.default_factory is not f.default:
            try:
                sub_instance = f.default_factory()
                if hasattr(sub_instance, '__dataclass_fields__'):
                    for sf in fields(sub_instance):
                        if sf.name not in seen:
                            _add_arg(parser, sf)
                            seen.add(sf.name)
                    continue
            except TypeError:
                pass

        if f.name not in seen:
            _add_arg(parser, f)
            seen.add(f.name)

    return parser


def _add_arg(parser: argparse.ArgumentParser, f):
    """Add a single dataclass field as an argparse argument."""
    name = f"--{f.name}"
    alias = f"--{f.name.replace('_', '-')}"
    names = [name]
    if alias != name:
        names.append(alias)

    ftype = f.type
    # Unwrap Optional
    args_t = getattr(ftype, '__args__', None)
    if args_t and type(None) in args_t:
        non_none = [a for a in args_t if a is not type(None)]
        ftype = non_none[0] if non_none else str

    if ftype is bool or ftype == 'bool':
        parser.add_argument(*names, default=None, type=_str_to_bool,
                            help=f"(bool, default: {f.default})")
    elif ftype is list or getattr(ftype, '__origin__', None) is list:
        parser.add_argument(*names, default=None, nargs='+',
                            help=f"(list, default: {f.default})")
    elif ftype in (int, float, str):
        parser.add_argument(*names, default=None, type=ftype,
                            help=f"({ftype.__name__}, default: {f.default})")
    else:
        parser.add_argument(*names, default=None, type=str,
                            help=f"(default: {f.default})")


def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes'):
        return True
    if v.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v}")


def parse_args_and_load_config(config_cls=EncoderConfig, description=""):
    """Convenience: parse CLI args and load config in one call."""
    parser = build_parser(config_cls, description)
    args = parser.parse_args()
    cli_overrides = {k: v for k, v in vars(args).items() if k != "config"}
    cfg = load_config(args.config, cli_overrides, config_cls)

    # Auto output_dir
    if cfg.output_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_tag = cfg.model.model.split("/")[-1] if hasattr(cfg.model, 'model') else "model"
        pipeline_tag = config_cls.__name__.replace("Config", "").lower()
        cfg = _set_output_dir(cfg, f"outputs/{pipeline_tag}-{model_tag}-{ts}")

    return cfg


def _set_output_dir(cfg, output_dir: str):
    """Set output_dir on a frozen-ish dataclass (dataclasses are mutable by default)."""
    cfg.output_dir = output_dir
    return cfg
