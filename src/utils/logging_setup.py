#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/utils/logging_setup.py

애플리케이션의 로깅 시스템을 초기화합니다.
"""

import logging, logging.handlers, os, sys
from typing import Optional

def init_logging(level: int = logging.INFO,
                 log_file: Optional[str] = None,
                 file_level: Optional[int] = None) -> None:
    """로깅 초기화."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)         # 모든 메시지 처리

    # 중복 방지
    if root.handlers:
        root.handlers.clear()

    # ── 콘솔(또는 stderr) 핸들러 ────────────────────────────────
    stream = sys.stderr or sys.stdout    # ★ 명시적 스트림
    console_handler = logging.StreamHandler(stream)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(
        "[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    root.addHandler(console_handler)

    # ── 파일 핸들러 (옵션) ──────────────────────────────────────
    if log_file:
        file_level = file_level or level
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            fh = logging.handlers.RotatingFileHandler(
                filename=log_file,
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8"
            )
            fh.setLevel(file_level)
            fh.setFormatter(logging.Formatter(
                "[%(levelname)s] %(asctime)s - %(name)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
            root.addHandler(fh)
            root.info("File logging → '%s' (%s)", log_file, logging.getLevelName(file_level))
        except Exception as e:
            root.error("Failed to init file logging: %s", e, exc_info=True)

    root.info("Logging system initialized.")
