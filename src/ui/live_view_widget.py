#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/ui/live_view_widget.py
───────────────────────────────────────────────────────────────────────────────
Live-view composite widget supporting

• set_controller(ctrl)                → 단일-카메라 1-up 모드
• start(ctrls: list[CameraController])→ 다중 카메라 모자이크 (최대 4-up)
• stop()                              → 모든 GrabWorker 안전 종료

타일(Label) 크기 변화 → GrabWorker.update_target_width() 로 개별 전달해
다운스케일 폭을 카메라별로 정확히 유지한다.
"""

import sys
import math
import time
from enum import Enum, auto
from typing import List, Optional

import numpy as np
from PyQt5.QtCore import (
    Qt,
    QTimer,
    pyqtSignal,
    pyqtSlot,
    QRectF,
)
from PyQt5.QtGui import QImage, QPainter, QPixmap, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QDialog,
)
# ─── 기존 import 구역 맨 아래 정도에 추가 ───
from contextlib import suppress

# 외부 GrabWorker (다운스케일·FPS 제한·frame_ready 신호 보유)
from src.ui.grab_worker import GrabWorker
import collections

# ───────────────────────────────────────────────────────────── helpers
class _FullScreenDialog(QDialog):
    """풀-스크린 타일 뷰."""

    def __init__(self, pixmap_provider, title: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setWindowTitle(title)

        self._provider = pixmap_provider
        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addWidget(self._label)
        self.setLayout(layout)
        self.setModal(True)
        self.showFullScreen()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(30)  # ≈33 FPS

    def _refresh(self):
        pm = self._provider()
        if pm is not None:
            self._label.setPixmap(pm.scaled(self._label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Escape, Qt.Key_Space, Qt.Key_Return):
            self.accept()
        else:
            super().keyPressEvent(e)

from PyQt5.QtWidgets import QLabel, QSizePolicy
class LiveFeedTile(QLabel):
    doubleClicked = pyqtSignal()
    widthChanged  = pyqtSignal(int)

    def __init__(self, parent: Optional['QWidget']=None) -> None:
        super().__init__(parent)

        # ── UI 셋업 ─────────────────────────────────────────────
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("""
            QLabel {
                background: #455A64;
                border: 1px solid #607D8B;
                border-radius: 8px;
            }
        """)

        # ── 내부 상태 ──────────────────────────────────────────
        self._font = QFont("Inter", 13, 500)
        self._frame_ts = collections.deque(maxlen=100)
        self._last_fps = 0.0
        self._f_cnt    = 0
        self._err_cnt  = 0

        self._qimg: QImage = QImage()   # ← 이제 QPixmap 대신 QImage 보관

    # ======================================================================
    # GrabWorker / Controller → QImage 저장
    # ======================================================================
    @pyqtSlot(object)
    def update_frame(self, frame_data):
        now = time.time()
        self._frame_ts.append(now)
        if len(self._frame_ts) >= 2:
            dt = self._frame_ts[-1] - self._frame_ts[0]
            if dt > 0:
                self._last_fps = (len(self._frame_ts) - 1) / dt
        self._f_cnt += 1

        qimg: Optional[QImage] = None

        # ---- 1. QImage 바로 전달 ---------------------------
        if isinstance(frame_data, QImage):
            qimg = frame_data

        # ---- 2. NumPy → QImage -----------------------------
        elif isinstance(frame_data, np.ndarray):
            img = frame_data

            # 16-bit Gray → 8-bit Auto-contrast
            if img.ndim == 2 and img.dtype == np.uint16:
                mn, mx = int(img.min()), int(img.max())
                img8 = np.zeros_like(img, dtype=np.uint8) if mx == mn \
                       else ((img - mn) * (255.0 / (mx - mn))).astype(np.uint8)
                h, w = img8.shape
                qimg = QImage(img8.data, w, h, w, QImage.Format_Grayscale8)

            # 8-bit Gray
            elif img.ndim == 2 and img.dtype == np.uint8:
                h, w = img.shape
                qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)

            # 8/16-bit RGB
            elif img.ndim == 3 and img.shape[2] == 3:
                if img.dtype == np.uint16:
                    img = (img >> 8).astype(np.uint8)
                h, w, _ = img.shape
                qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)

        # ---- QImage 보관 & repaint 요청 ---------------------
        if qimg and not qimg.isNull():
            self._qimg = qimg
            self.update()

    # ======================================================================
    # QWidget 이벤트
    # ======================================================================
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.widthChanged.emit(self.width())

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.doubleClicked.emit()
        super().mouseDoubleClickEvent(e)

    def paintEvent(self, _):
        if self._qimg.isNull():
            return

        # ── 이미지 그리기 (FastTransformation) ───────────────
        p = QPainter(self)
        tgt = self.rect()
        scaled = self._qimg.scaled(
            tgt.size(), Qt.KeepAspectRatio, Qt.FastTransformation
        )
        x = (tgt.width()  - scaled.width())  // 2
        y = (tgt.height() - scaled.height()) // 2
        p.drawImage(x, y, scaled)

        # ── FPS / Loss 오버레이 ──────────────────────────────
        loss = self._err_cnt / max(self._f_cnt + self._err_cnt, 1)
        txt  = f"FPS: {self._last_fps:5.1f}  |  Loss: {loss*100:4.1f}%"
        p.setFont(self._font)
        m = p.fontMetrics()
        w = m.horizontalAdvance(txt) + 20
        h = m.height() + 16
        r = QRectF(16, 16, w, h)
        p.fillRect(r, QColor(31, 42, 48, 200))
        p.setPen(QColor("#F0F4F8"))
        p.drawRoundedRect(r, 6, 6)
        p.drawText(r.adjusted(10, 8, -10, -8),
                   Qt.AlignLeft | Qt.AlignVCenter, txt)
        p.end()

    # 제공용
    def pixmap_provider(self):
        return self._qimg


# ───────────────────────────────────────────────────────────── main widget
class _ViewMode(Enum):
    SINGLE = auto()
    MOSAIC = auto()


class LiveViewWidget(QWidget):
    """≤4 타일 & GrabWorker 스레드 관리"""

    MAX_TILES = 4

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # 탭이 늘어나면 최대한 확장
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # ── 그리드 레이아웃 ────────────────────────────────
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(8)

        # ── 내부 상태 ─────────────────────────────────────
        self._tiles: List[LiveFeedTile] = []
        self._workers: List[GrabWorker] = []
        self._ctrls: List = []              # 현재 연결된 Controller
        self._mode = _ViewMode.SINGLE

        # 최소 1개 타일 확보 후 레이아웃 구성
        self._ensure_tile_pool(1)
        self._rebuild_grid(1)

    def _ensure_tile_pool(self, n_needed: int) -> None:
        while len(self._tiles) < n_needed and len(self._tiles) < self.MAX_TILES:
            tile = LiveFeedTile(self)
            self._tiles.append(tile)
    def set_controller(self, ctrl):
        """단일 카메라(1-up) 라이브뷰."""
        self.stop()
        if ctrl is None:
            return
        self._mode = _ViewMode.SINGLE
        self._spawn_tiles(1)
        self._connect_ctrls([ctrl])

    def start(self, ctrls: List, mosaic: bool = True):
        """다중 카메라 모자이크(≤4)."""
        self.stop()
        if not ctrls:
            return
        if len(ctrls) == 1 and not mosaic:
            self.set_controller(ctrls[0])
            return

        self._mode = _ViewMode.MOSAIC
        self._spawn_tiles(len(ctrls))
        self._connect_ctrls(ctrls[: self.MAX_TILES])

    def stop(self):
        """
        [안정화 버전] live-view를 중지하고 모든 관련 리소스를 안전하게 해제합니다.
        객체를 파괴하기 전에 시그널 연결을 먼저 해제하여 경합 상태를 방지합니다.
        """
        # --- [핵심 수정] ---
        # 1. GrabWorker 객체를 파괴하기 전에, 먼저 시그널 연결을 해제합니다.
        if hasattr(self, "_worker") and self._worker:
            # self._worker.frame_ready.disconnect()와 같이 모든 연결을 끊는 것은 복잡하므로,
            # 대신 타일(시그널을 보내는 쪽)의 연결을 끊는 것이 더 간단하고 효과적입니다.
            for tile in self._tiles:
                try:
                    tile.widthChanged.disconnect()
                except TypeError:
                    # 이미 연결이 끊겨있으면 TypeError가 발생할 수 있으므로 무시합니다.
                    pass
        # --- [수정 끝] ---

        # 2. 카메라의 live-view 모드를 중지시킵니다.
        for ctrl in self._ctrls:
            with suppress(Exception):
                ctrl.stop_live_view()
        self._ctrls.clear()

        # 3. 이제 안전하게 GrabWorker 스레드를 중지하고 객체를 파괴합니다.
        if hasattr(self, "_worker") and self._worker:
            self._worker.stop()
            self._worker = None # 이 시점 이후에는 lambda가 호출되어도 안전합니다.

        # 4. 화면을 초기 상태로 되돌립니다.
        for t in self._tiles:
            # pixmap을 직접 조작하는 대신 update_frame에 빈 배열을 보내는 것이 더 일관성 있습니다.
            t.update_frame(np.array([]))
            t.update() # repaint 강제
    # ===================================================================
    # 6) LiveViewWidget._connect_ctrls  ― 시그널 연결 + live_view 시작
    # ===================================================================
    # 선행 수정 이후 _connect_ctrls 그대로 동작 ↴
    def _connect_ctrls(self, ctrls: List):
        """컨트롤러들을 GrabWorker에 등록하고 시그널 연결"""
        self._ctrls = ctrls

        # ① 컨트롤러 live-view ON
        for c in ctrls:
            c.start_live_view()

        # ② GrabWorker 생성 (하나만)
        if hasattr(self, "_worker") and self._worker:
            self._worker.stop()
        self._worker = GrabWorker(ctrls, max_fps=30, parent=self)

        # ③ GrabWorker → 타일 (프레임 신호)
        for tile, ctrl in zip(self._tiles, ctrls):
            # frame_ready(cam_id, QImage)
            self._worker.frame_ready.connect(
                lambda cid, q, t=tile, my=ctrl.cam_id:
                    t.update_frame(q) if cid == my else None,
                Qt.QueuedConnection,
            )
            # 타일 폭 변화 → GrabWorker
            tile.widthChanged.connect(
                lambda w, cid=ctrl.cam_id: self._worker.update_target_width(cid, w),
                Qt.QueuedConnection,
            )

        self._worker.start()
    # ───────── internal helpers
    # tile
    def _spawn_tiles(self, n: int):
        n = max(1, min(self.MAX_TILES, n))
        while len(self._tiles) < n:
            tile = LiveFeedTile()
            tile.doubleClicked.connect(lambda t=tile: self._open_full(t))
            self._tiles.append(tile)
        while len(self._tiles) > n:
            t = self._tiles.pop()
            self._grid.removeWidget(t)
            t.deleteLater()

        self._rebuild_grid(n)

    def _rebuild_grid(self, n_tiles: int) -> None:
        """현재 타일 수(n_tiles)에 맞게 그리드를 다시 짠다."""

        # 1) 기존 위젯 제거 ------------------------------------------------
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

        # 2) 행·열 계산 ----------------------------------------------------
        cols = 2 if n_tiles > 1 else 1
        rows = math.ceil(n_tiles / cols)

        # 3) 타일 배치 & stretch 설정 --------------------------------------
        for idx in range(n_tiles):
            r, c = divmod(idx, cols)
            tile = self._tiles[idx]
            self._grid.addWidget(tile, r, c)

            # 실제 타일이 들어간 행·열에만 늘어남 값을 준다
            self._grid.setRowStretch(r, 1)
            self._grid.setColumnStretch(c, 1)

        # 4) 남는 행·열 stretch 0 으로 초기화 ------------------------------
        for r in range(rows, self._grid.rowCount()):
            self._grid.setRowStretch(r, 0)
        for c in range(cols, self._grid.columnCount()):
            self._grid.setColumnStretch(c, 0)


    # slot: external frameReady style not needed (mapped directly above)

    # popup
    def _open_full(self, tile: LiveFeedTile):
        _FullScreenDialog(tile.pixmap_provider, parent=self).exec_()

    # graceful close
    def closeEvent(self, e):
        self.stop()
        super().closeEvent(e)


# ───────────────────────────────────────────────────────────── demo
if __name__ == "__main__":
    class _DummyController:
        """Mock CameraController."""

        def __init__(self, seed=0):
            np.random.seed(seed)

        def start_acquisition(self):
            pass

        def stop_acquisition(self):
            pass

        def get_latest_frame(self, timeout_ms=50):
            time.sleep(0.03)  # ≈33 FPS
            h, w = 480, 640
            return (np.random.rand(h, w, 3) * 255).astype(np.uint8)

    app = QApplication(sys.argv)
    v = LiveViewWidget()
    v.resize(960, 600)

    ctrls = [_DummyController(i) for i in range(4)]

    def _toggle(state=[False]):
        state[0] = not state[0]
        if state[0]:
            v.set_controller(ctrls[0])
        else:
            v.start(ctrls)

    _toggle()
    timer = QTimer()
    timer.timeout.connect(_toggle)
    timer.start(5000)

    v.show()
    sys.exit(app.exec_())
