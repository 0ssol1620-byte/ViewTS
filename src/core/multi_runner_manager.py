#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MultiRunnerManager
~~~~~~~~~~~~~~~~~~
• 카메라별 `SequenceRunner` 여러 개를 병렬로 실행/모니터링
• MainWindow · CLI 등에서 통합 제어/진행률 파이프 역할

API
─────────────────────────────────────────────────────────────
start(sequence, controllers, ctx)        → 병렬 실행
stop(cam_id: str | None = None)          → 단일/전체 중지
is_running(cam_id: str | None = None)    → 단일/전체 상태
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtCore import QObject, pyqtSignal

from src.core.sequence_runner import SequenceRunner, Sequence
from src.core.camera_controller import CameraController
import numpy as np
logger = logging.getLogger(__name__)


class MultiRunnerManager(QObject):
    # 통합 신호 ---------------------------------------------------------------
    log_message         = pyqtSignal(str, str)                 # level, text
    progress_update     = pyqtSignal(str, int, int)            # cam_id, step, total
    sequence_finished   = pyqtSignal(str, str, str)            # cam_id, status, detail

    step_started        = pyqtSignal(str, int, str)            # cam_id, idx, name
    step_result         = pyqtSignal(str, int, str, dict)      # cam_id, idx, name, result
    validation_result   = pyqtSignal(str, int, str, bool, str) # cam_id, idx, rule, passed, detail
    loop_progress       = pyqtSignal(str, int, int)            # cam_id, current, target
    preview_frame = pyqtSignal(str, np.ndarray)
    # -----------------------------------------------------------------------
    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._runners: Dict[str, SequenceRunner] = {}  # cam_id → runner

    # ───────────────────────────── public API
    def start(
            self,
            sequence: Sequence,
            controllers: List[CameraController],
            initial_context: Optional[dict] = None,
            memento_dir: Path | None = None,
    ) -> None:
        """
        controllers 의 각 카메라에 대해 독립 Runner 생성·시작.
        이미 실행 중이면 stop() 으로 정리 후 재시작.
        """
        self.stop()  # ← 기 실행중인 것 정리
        base_ctx = initial_context or {}

        for ctrl in controllers:
            cam_id = ctrl.cam_id
            runner = SequenceRunner(
                sequence=sequence,
                controller=ctrl,
                initial_context={**base_ctx, "cam_id": cam_id},  # ★ per-runner
                memento_dir=memento_dir
                            or (Path(__file__).parents[2] / "logs" / "memento"),
                parent=self,
            )
            self._hook_runner_signals(cam_id, runner)
            self._runners[cam_id] = runner
            runner.start()
            logger.info("SequenceRunner started – %s", cam_id)

    def stop(self, cam_id: str | None = None) -> None:
        """cam_id 지정 시 해당 Runner, None 이면 전체 중지."""
        targets = (
            [cam_id] if cam_id else list(self._runners.keys())
        )

        for cid in targets:
            runner = self._runners.get(cid)
            if runner and runner.is_running():
                runner.stop()

    # 상태 조회 --------------------------------------------------------------
    def is_running(self, cam_id: str | None = None) -> bool:
        if cam_id:
            r = self._runners.get(cam_id)
            return bool(r and r.is_running())
        return any(r.is_running() for r in self._runners.values())

    # ───────────────────────────── internal
    def _hook_runner_signals(self, cam_id: str, r: SequenceRunner) -> None:
        # 로그 / 주요 상태
        r.log_message.connect(self.log_message.emit)
        r.progress_update.connect(  # ★ cam_id와 함께 재발행
            lambda cur, tot, cid=cam_id: self.progress_update.emit(cid, cur, tot)
        )
        r.sequence_finished.connect(
            lambda st, de, cid=cam_id: self.sequence_finished.emit(cid, st, de)
        )

        # 세부 이벤트
        r.step_started.connect(
            lambda idx, name, cid=cam_id: self.step_started.emit(cid, idx, name)
        )
        r.step_result.connect(
            lambda idx, name, res, cid=cam_id: self.step_result.emit(cid, idx, name, res)
        )
        r.validation_result.connect(
            lambda idx, rule, ok, detail, cid=cam_id:
            self.validation_result.emit(cid, idx, rule, ok, detail)
        )
        r.loop_progress_update.connect(
            lambda cur, tgt, cid=cam_id: self.loop_progress.emit(cid, cur, tgt)
        )
        r.test_frame_grabbed.connect(
            lambda cid_inner, frame, _cid=cam_id: self.preview_frame.emit(cid_inner, frame)
        )
        # Runner thread 종료 되면 dict 정리
        r.finished.connect(lambda cid=cam_id: self._runners.pop(cid, None))


