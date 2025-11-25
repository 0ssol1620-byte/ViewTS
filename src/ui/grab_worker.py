#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GrabWorker – Real-time, Low-latency Multi-camera Drainer (speed-tuned)
=====================================================================
❶ “Flush-and-Grab” 은 유지하되 flush 빈도를 300 ms 로 제한
❷ 컨트롤러별 target_width 를 받아 백그라운드에서 **FastTransformation**
   으로 미리 축소된 QImage 를 생성해 UI 스레드는 그리기만 함
"""

from __future__ import annotations
from typing import List, Union, Optional, Any
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QObject, QSize, Qt
from PyQt5.QtGui  import QImage
import numpy as np
import threading, time, logging, math
from contextlib import suppress

try:
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from src.core.camera_controller import CameraController
except Exception:
    CameraController = Any  # type: ignore

logger = logging.getLogger(__name__)


class GrabWorker(QThread):
    """
    하나의 스레드가 다수 CameraController 를 빠르게 ‘배수펌프’ 식으로 비웁니다.
    target_width 에 맞춰 이미지 크기를 줄여 보내 CPU→GPU 트래픽을 크게 절감합니다.
    """
    frame_ready = pyqtSignal(str, QImage)   # cam_id, scaled QImage
    finished    = pyqtSignal()

    # ------------------------------------------------------------------ #
    # 생성/제어
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        controllers: Union[List["CameraController"], "CameraController"],
        *,
        max_fps: int = 30,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._controllers: List["CameraController"] = (
            controllers if isinstance(controllers, list) else [controllers]
        )

        self._frame_interval = 1.0 / max_fps if max_fps > 0 else 0.0
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # 컨트롤러별 타겟 폭 - 기본 640
        self._target_width: dict[str, int] = {
            getattr(c, "cam_id", str(idx)): 640
            for idx, c in enumerate(self._controllers)
        }
    @property
    def controllers(self):
        """legacy access — keep old code working"""
        return self._controllers
    # 외부에서 타겟 폭 업데이트
    @pyqtSlot(str, int)
    def update_target_width(self, cam_id: str, width: int) -> None:
        with self._lock:
            self._target_width[cam_id] = max(64, width)

    # 타일이 resize 될 때 호출해 달라고 연결할 slot
    @pyqtSlot(int)
    def update_target_width_slot(self, width: int) -> None:
        sender = self.sender()
        cam_id = getattr(sender, "cam_id", None)
        if cam_id:
            self.update_target_width(cam_id, width)

    # 컨트롤러 추가/제거
    def add_controller(self, ctrl: "CameraController") -> None:
        with self._lock:
            if ctrl not in self._controllers:
                self._controllers.append(ctrl)
                self._target_width[ctrl.cam_id] = 640

    def remove_controller(self, ctrl: "CameraController") -> None:
        with self._lock:
            with suppress(ValueError):
                self._controllers.remove(ctrl)
            self._target_width.pop(ctrl.cam_id, None)

    def stop(self) -> None:
        self._stop_event.set()
        if self.isRunning():
            if not self.wait(1000):
                self.terminate()
                self.wait()

    def run(self) -> None:
        """
        GrabWorker 실시간 드레인 루프 (최종 안정화판)
          • Quiet 중: 평소엔 pop 스킵, 단 OutputFifo 고수위면 컨트롤러 비상 드레인으로 역압 해소
          • 일반 시: OutputFifo 고수위(≥28)면 선제 드레인으로 포화 방지
          • 빈 큐면 pop 시도 금지(드라이버 -1011 스팸 차단)
          • 주기적 ensure_stream_ready()로 입력 풀 보강
          • UI 전송 주기 제한(FPS), 바운디드 드레인으로 burst 완화
        """
        logger.info("GrabWorker thread started.")
        self._stop_event.clear()

        last_emit_by_cam: dict[str, float] = {}
        last_refill_by_cam: dict[str, float] = {}

        # 튜닝 상수
        POP_TIMEOUT_MS = 20  # 프레임 주기가 긴 경우 false-timeout 완화
        DRAIN_BUDGET = 2  # 한 사이클에서 pop 할 최대 프레임 수
        REFILL_PERIOD_S = 0.05  # ← 0.30 에서 50ms 로 줄여 더 자주 리필
        HWM_OUT_QUIET = 16  # Quiet 중 임계도 조금 낮춤
        HWM_OUT_ACTIVE = 24  # 일반 시 임계도 살짝 낮춤
        EMERGENCY_BUDGET = 12
        ACTIVE_BUDGET = 16

        try:
            while not self._stop_event.is_set():
                with self._lock:
                    ctrls = list(self._controllers)
                    tgt_w = dict(self._target_width)

                if not ctrls:
                    self.msleep(50)
                    continue

                now = time.monotonic()

                for ctrl in ctrls:
                    if self._stop_event.is_set():
                        break

                    # 연결/그래빙 상태 확인
                    try:
                        if not (ctrl.is_connected() and ctrl.is_grabbing()):
                            continue
                    except Exception:
                        continue

                    # Quiet 종료 시점 복구(있으면)
                    with suppress(Exception):
                        if hasattr(ctrl, "_maybe_leave_quiet"):
                            ctrl._maybe_leave_quiet()

                    # Quiet 윈도우 처리 (GrabWorker.run 내부)
                    try:
                        if getattr(ctrl, "_is_quiet", lambda: False)():
                            snap = {}
                            with suppress(Exception):
                                if hasattr(ctrl, "_fifo_snapshot"):
                                    snap = ctrl._fifo_snapshot()
                            out = int(snap.get("out", 0))

                            # 트리거가 켜져 있으면 더 낮은 임계 + 더 큰 예산
                            trig_on = False
                            try:
                                p = getattr(ctrl, "params", None)
                                if p and "TriggerMode" in p.features():
                                    trig_on = (str(p.get("TriggerMode")) != "Off")
                            except Exception:
                                pass

                            if trig_on:
                                with suppress(Exception):
                                    if hasattr(ctrl, "_emergency_backpressure_relief"):
                                        ctrl._emergency_backpressure_relief(hwm=12, budget=48)
                            else:
                                if out >= HWM_OUT_QUIET:
                                    with suppress(Exception):
                                        if hasattr(ctrl, "_emergency_backpressure_relief"):
                                            ctrl._emergency_backpressure_relief(hwm=HWM_OUT_QUIET,
                                                                                budget=EMERGENCY_BUDGET)
                            continue
                    except Exception:
                        pass

                    cam_id = getattr(ctrl, "cam_id", "cam")
                    ui_due = (self._frame_interval <= 0.0) or (
                            now - last_emit_by_cam.get(cam_id, 0.0) >= self._frame_interval
                    )

                    # 주기적 입력 풀 보강
                    if now - last_refill_by_cam.get(cam_id, 0.0) >= REFILL_PERIOD_S:
                        with suppress(Exception):
                            diag = ctrl.ensure_stream_ready(min_input_fifo=1, refill_if_zero=True)
                            logger.debug("[%s] ensure_stream_ready: %s", cam_id, diag)
                        last_refill_by_cam[cam_id] = now

                    # 일반 시에도 출력 FIFO 고수위면 선제 역압 해소
                    try:
                        snap = {}
                        with suppress(Exception):
                            if hasattr(ctrl, "_fifo_snapshot"):
                                snap = ctrl._fifo_snapshot()
                        out = int(snap.get("out", 0))
                        if out >= HWM_OUT_ACTIVE:
                            with suppress(Exception):
                                if hasattr(ctrl, "_emergency_backpressure_relief"):
                                    ctrl._emergency_backpressure_relief(
                                        hwm=HWM_OUT_ACTIVE, budget=ACTIVE_BUDGET
                                    )
                    except Exception:
                        pass

                    # 빈 큐면 pop 금지
                    try:
                        if not ctrl._stream_has_output():
                            continue
                    except Exception:
                        continue

                    # 바운디드 드레인
                    last_frame = None
                    budget = DRAIN_BUDGET

                    while budget > 0 and not self._stop_event.is_set():
                        try:
                            f = ctrl.get_next_frame(timeout_ms=POP_TIMEOUT_MS, count_timeout_error=False)
                        except Exception:
                            f = None

                        if f is None:
                            break

                        last_frame = f
                        budget -= 1

                    # UI 전송
                    if last_frame is not None and ui_due:
                        try:
                            qimg = self._ndarray_to_qimage(last_frame, tgt_w.get(cam_id, 640))
                            if qimg:
                                self.frame_ready.emit(cam_id, qimg)
                                last_emit_by_cam[cam_id] = now
                        except Exception:
                            pass

                self.msleep(1)

        except Exception as e:
            logger.error("GrabWorker main loop error: %s", e, exc_info=True)
        finally:
            with suppress(Exception):
                self.finished.emit()
            logger.info("GrabWorker thread finished.")

    # ------------------------------------------------------------------ #
    # NumPy → QImage (FastTransformation 포함)  (교체 버전: deep copy)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ndarray_to_qimage(arr: np.ndarray, max_w: int) -> Optional[QImage]:
        """
        - 16-bit Gray → Auto-contrast 8-bit
        - 3-채널 16-bit → 상위 8-bit
        - 최종적으로 max_w 로 FastTransformation 다운스케일
        - QImage는 반드시 .copy()로 소유 메모리를 분리 (emit 후 원본 변경 보호)
        """
        try:
            q: Optional[QImage] = None

            if arr.ndim == 2 and arr.dtype == np.uint16:
                mn, mx = int(arr.min()), int(arr.max())
                if mx == mn:
                    arr8 = np.zeros_like(arr, dtype=np.uint8)
                else:
                    arr8 = ((arr - mn) * (255.0 / (mx - mn))).astype(np.uint8)
                h, w = arr8.shape
                q = QImage(arr8.data, w, h, w, QImage.Format_Grayscale8).copy()

            elif arr.ndim == 2 and arr.dtype == np.uint8:
                h, w = arr.shape
                q = QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()

            elif arr.ndim == 3 and arr.shape[2] == 3:
                if arr.dtype == np.uint16:
                    arr = (arr >> 8).astype(np.uint8)
                h, w, _ = arr.shape
                q = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()

            if q and not q.isNull() and q.width() > max_w > 0:
                # copy()로 분리된 메모리 위에서 스케일 (추가 복사 없음)
                q = q.scaledToWidth(max_w, Qt.FastTransformation)  # type: ignore[attr-defined]
            return q

        except Exception as e:
            logger.debug("ndarray→QImage fail: %s", e)
        return None

