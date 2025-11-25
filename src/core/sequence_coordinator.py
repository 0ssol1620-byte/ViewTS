#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sequence_coordinator.py
────────────────────────────────────────────────────────────────────────
여러 대 카메라에서 개별 SequenceRunner(QThread)를 병렬 실행하고,
첫 Runner 가 실패·중단되면 나머지를 즉시 중단(cascade cancel)한다.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

from src.core.sequence_runner import SequenceRunner
from src.core.controller_pool import controllers        # 풀에 등록된 컨트롤러
from src.core.sequence_types import Sequence            # 유형 alias

logger = logging.getLogger(__name__)


class SequenceCoordinator:
    """
    Runner 인스턴스를 n개 생성 → ThreadPoolExecutor + asyncio.gather 로 병렬 실행.
    """

    def __init__(self,
                 sequence: Sequence,
                 cam_ids: List[str],
                 *,
                 max_workers: int = 8) -> None:

        self.sequence      = sequence
        self.cam_ids       = cam_ids
        self.runners: Dict[str, SequenceRunner] = {}
        self._executor     = ThreadPoolExecutor(max_workers=max_workers)

        # Runner 생성 (아직 start() 안 함)
        for cid in cam_ids:
            ctrl = controllers[cid]
            runner = SequenceRunner(sequence, ctrl)
            self.runners[cid] = runner

    # ------------------------------------------------------------------
    async def _run_runner(self, runner: SequenceRunner) -> None:
        """
        QThread.start() 는 blocking 이 아니므로,
        완료를 Future 로 래핑해 asyncio 와 접합.
        """
        loop = asyncio.get_running_loop()
        done = loop.create_future()

        def _on_finished(status: str, msg: str) -> None:      # Qt → Future
            if not done.done():
                done.set_result((status, msg))

        runner.sequence_finished.connect(_on_finished)
        # start() 자체는 즉시 반환하므로 thread executor 필요 없음
        runner.start()

        status, msg = await done
        if status in ("Failed", "Error"):
            raise RuntimeError(f"{runner.controller.cam_id} → {status}: {msg}")

    # ------------------------------------------------------------------
    async def run_all(self) -> None:
        """
        모든 Runner 동시 실행.
        * 첫 실패 (exception) 시 gather() 가 즉시 취소하도록 return_when=FIRST_EXCEPTION*
        """
        tasks = [
            asyncio.create_task(self._run_runner(r))
            for r in self.runners.values()
        ]

        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_EXCEPTION,
        )

        # 첫 실패 감지되면 pending Runner 중단
        for t in pending:
            t.cancel()

        # 예외 재-raise (첫 번째 것만)
        for t in done:
            exc = t.exception()
            if exc:
                # stop() broadcast 로 나머지 Runner 는 이미 중단 요청됨
                raise exc   # propagate to caller

    # ------------------------------------------------------------------
    @staticmethod
    def run(sequence: Sequence, cam_ids: List[str]) -> None:
        """
        Boiler-plate entrypoint (sync 환경에서도 호출하기 쉽게).
        """
        coord = SequenceCoordinator(sequence, cam_ids)
        asyncio.run(coord.run_all())
