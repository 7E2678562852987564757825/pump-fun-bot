"""Config loader."""
from __future__ import annotations
import logging
import logging.handlers
from pathlib import Path
from typing import Any
import yaml

_DEFAULT_CFG = Path(__file__).parent / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else _DEFAULT_CFG
    with cfg_path.open() as f:
        return yaml.safe_load(f)


def setup_logging(cfg: dict[str, Any]) -> None:
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO"))
    log_file = log_cfg.get("file", "logs/hl_liq_cascade.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    rfh = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=log_cfg.get("max_bytes", 10_485_760),
        backupCount=log_cfg.get("backup_count", 5),
    )
    handlers.append(rfh)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=handlers,
    )
