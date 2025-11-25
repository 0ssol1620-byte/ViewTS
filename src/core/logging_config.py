#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
logging_config.py
──────────────────────────────────────────────────────────────────────────────
Centralised logging initialiser for ViewTS  –  2025-05-20 edition

Features
========
1. Rotating **text log**  → logs/camera_test_suite.log
2. Colour **console log** → “[cam:SER]” 태그 인식 ANSI 색상
3. **Error-image logger**
   • Root: logs/error_images/<cam_id>/YYYYMMDD/…
   • Side-car JSON meta “*.meta.json” 저장
   • 파일 보존일(기본 30 일) 경과 시 자동 삭제
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict

# ─────────────────────────────── Constants ────────────────────────────────
LOG_ROOT              = Path("logs")
TEXT_LOG_FILE         = LOG_ROOT / "camera_test_suite.log"
IMAGE_LOG_ROOT        = LOG_ROOT / "error_images"
IMAGE_RETENTION_DAYS  = int(os.getenv("IMAGE_LOG_KEEP_DAYS", 30))

LOG_LEVEL_FILE    = logging.INFO
LOG_LEVEL_CONSOLE = logging.DEBUG
FILE_MAX_BYTES    = 3 * 1024 * 1024   # 3 MB
FILE_BACKUP_COUNT = 3

# ─────────────────────────────── Colour helper ─────────────────────────────
ANSI = {
    "RESET":  "\033[0m",
    "GREY":   "\033[90m",
    "RED":    "\033[91m",
    "GREEN":  "\033[92m",
    "YELLOW": "\033[93m",
}

LEVEL_COLOUR = {
    logging.DEBUG:    ANSI["GREY"],
    logging.INFO:     ANSI["GREEN"],
    logging.WARNING:  ANSI["YELLOW"],
    logging.ERROR:    ANSI["RED"],
    logging.CRITICAL: ANSI["RED"],
}

class ColourFormatter(logging.Formatter):
    """Adds ANSI colours and time stamp; keeps original message intact."""
    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        colour = LEVEL_COLOUR.get(record.levelno, "")
        reset  = ANSI["RESET"]
        ts     = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        return f"{colour}{ts} {super().format(record)}{reset}"

# ───────────────────────────── Initialiser ────────────────────────────────
def init_logging() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    # rotating file handler
    fh = RotatingFileHandler(
        TEXT_LOG_FILE,
        maxBytes=FILE_MAX_BYTES,
        backupCount=FILE_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setLevel(LOG_LEVEL_FILE)
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                          "%Y-%m-%d %H:%M:%S")
    )

    # colour console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(LOG_LEVEL_CONSOLE)
    ch.setFormatter(ColourFormatter("%(message)s"))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.addHandler(ch)

    root.info("File logging initialised → %s", TEXT_LOG_FILE)
    root.debug("Console colour formatter active.")
    _purge_old_error_images()

# ───────────────────── Error-image meta + purge helpers ────────────────────
def _error_meta_path(image_path: Path) -> Path:
    return image_path.with_suffix(image_path.suffix + ".meta.json")

def log_error_image(
    *,
    cam_id: str,
    image_path: str | Path,
    step_num: int,
    message: str,
    extra: Dict[str, Any] | None = None,
) -> None:
    """
    Save JSON metadata next to an already-stored error image.
    """
    image_path = Path(image_path)
    meta_path  = _error_meta_path(image_path)

    meta: Dict[str, Any] = {
        "cam_id":    cam_id,
        "timestamp": datetime.utcnow().isoformat(),
        "step":      step_num,
        "image":     str(image_path),
        "message":   message,
    }
    if extra:
        meta.update(extra)

    try:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with meta_path.open("w", encoding="utf-8") as fp:
            json.dump(meta, fp, indent=2)
        logging.getLogger(__name__).info("Error-image meta saved → %s", meta_path)
    except Exception as exc:                                                  # noqa: BLE001
        logging.getLogger(__name__).warning("Meta save failed: %s", exc, exc_info=True)

def _purge_old_error_images() -> None:
    """Remove error images & meta older than IMAGE_RETENTION_DAYS."""
    if IMAGE_RETENTION_DAYS <= 0 or not IMAGE_LOG_ROOT.exists():
        return

    cutoff = datetime.now() - timedelta(days=IMAGE_RETENTION_DAYS)
    removed = 0

    for file in IMAGE_LOG_ROOT.rglob("*"):
        try:
            if file.is_file() and datetime.fromtimestamp(file.stat().st_mtime) < cutoff:
                file.unlink(missing_ok=True)
                removed += 1
        except Exception:
            continue

    # prune empty directories
    for d in sorted(IMAGE_LOG_ROOT.rglob("*"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            shutil.rmtree(d, ignore_errors=True)

    if removed:
        logging.getLogger(__name__).info("Purged %d expired error-image files.", removed)

# ───────────────────────── Auto-initialise on import ───────────────────────
if not logging.getLogger().handlers:
    init_logging()
