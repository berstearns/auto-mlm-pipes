"""
Swappable logging backend for training pipelines.

Supports: wandb (offline by default), tensorboard, aim, none.
Works with both HF Trainer (via setup_logging_env) and custom loops (via MetricLogger).
"""

import logging
import os
from pathlib import Path

from .config import config_to_dict

logger = logging.getLogger("logging_utils")

VALID_BACKENDS = {"wandb", "tensorboard", "aim", "none"}


def setup_logging_env(cfg) -> str:
    """Configure environment for the chosen logging backend.

    cfg can be a dataclass or dict.
    Returns the effective report_to string for TrainingArguments.
    """
    if isinstance(cfg, dict):
        backend = cfg.get("report_to", "wandb")
        wandb_project = cfg.get("wandb_project")
        output_dir = cfg.get("output_dir", "outputs")
        log_dir = cfg.get("log_dir")
    else:
        flat = config_to_dict(cfg)
        backend = flat.get("report_to", "wandb")
        wandb_project = flat.get("wandb_project")
        output_dir = flat.get("output_dir", "outputs")
        log_dir = flat.get("log_dir")

    if backend not in VALID_BACKENDS:
        raise ValueError(f"Unknown report_to={backend!r}. Choose from: {VALID_BACKENDS}")

    if backend == "wandb":
        if "WANDB_MODE" not in os.environ:
            os.environ["WANDB_MODE"] = "offline"
            logger.info("WANDB_MODE set to 'offline' (no account needed).")
        if wandb_project:
            os.environ["WANDB_PROJECT"] = wandb_project

    elif backend == "tensorboard":
        tb_dir = log_dir or os.path.join(output_dir or "outputs", "tb_logs")
        Path(tb_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"TensorBoard logging to: {tb_dir}")

    elif backend == "aim":
        logger.info("Aim logging enabled.")

    elif backend == "none":
        logger.info("Logging disabled (report_to=none).")

    return backend


class MetricLogger:
    """Lightweight logging adapter for custom training loops.

    Dispatches init/log/finish to the chosen backend.
    """

    def __init__(self, backend: str, cfg):
        self.backend = backend
        self._cfg = config_to_dict(cfg) if not isinstance(cfg, dict) else cfg
        self._run = None

    def init(self):
        if self.backend == "wandb":
            try:
                import wandb
                self._run = wandb.init(
                    project=self._cfg.get("wandb_project", "default"),
                    name=self._cfg.get("wandb_run_name"),
                    config=self._cfg,
                )
            except ImportError:
                logger.warning("wandb not installed, disabling logging.")
                self.backend = "none"

        elif self.backend == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = self._cfg.get("log_dir") or os.path.join(
                    self._cfg.get("output_dir", "outputs"), "tb_logs"
                )
                self._run = SummaryWriter(log_dir=log_dir)
            except ImportError:
                logger.warning("tensorboard not installed, disabling logging.")
                self.backend = "none"

        elif self.backend == "aim":
            try:
                from aim import Run
                self._run = Run()
                self._run["hparams"] = self._cfg
            except ImportError:
                logger.warning("aim not installed, disabling logging.")
                self.backend = "none"

    def log(self, metrics: dict, step: int | None = None):
        if self.backend == "wandb" and self._run is not None:
            import wandb
            wandb.log(metrics, step=step)
        elif self.backend == "tensorboard" and self._run is not None:
            for k, v in metrics.items():
                self._run.add_scalar(k, v, global_step=step)
        elif self.backend == "aim" and self._run is not None:
            for k, v in metrics.items():
                self._run.track(v, name=k, step=step)

    def finish(self):
        if self.backend == "wandb" and self._run is not None:
            import wandb
            wandb.finish()
        elif self.backend == "tensorboard" and self._run is not None:
            self._run.close()
        elif self.backend == "aim" and self._run is not None:
            self._run.close()
        self._run = None
