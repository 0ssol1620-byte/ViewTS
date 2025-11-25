#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
camera_status_dock.py
-------------------------------------------------
Live-status dock widget (FPS · Loss %) polled every 1 s
"""

from __future__ import annotations
from typing import Optional, Any, Dict, TYPE_CHECKING

from PyQt5.QtCore    import Qt, QTimer
from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QLabel, QFrame
)

# ──────────────────────────────────────────────────────────────
# CameraController stub (디자인-타임·문서화용)
# ──────────────────────────────────────────────────────────────
try:
    from src.core.camera_controller import CameraController            # noqa: E402
except ImportError:                                                    # pragma: no cover
    class CameraController:                                            # type: ignore
        stats: Dict[str, Any] = {"fps": 0.0, "frame_loss": 0.0}
        def is_connected(self) -> bool: return False
        def is_grabbing(self)  -> bool: return False

if TYPE_CHECKING:                      # for static type checkers
    from src.core.camera_controller import CameraController as _CC    # noqa: F401

# ══════════════════════════════════════════════════════════════
class CameraStatusDock(QDockWidget):
    """Right-side dock that shows live FPS / frame-loss."""

    POLL_INTERVAL_MS = 1000  # 1 s

    # ------------------------------------------------------ init
    def __init__(self,
                 controller: Optional[CameraController] = None,        # ★ optional
                 parent: Optional[QWidget] = None) -> None:
        super().__init__("Camera Status", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self._controller: Optional[CameraController] = controller      # ★ 내부 변수

        # --- inner widget -----------------------------------------
        inner = QWidget(self)
        vbox = QVBoxLayout(inner); vbox.setContentsMargins(8, 8, 8, 8); vbox.setSpacing(4)

        self.lbl_connection = QLabel("Camera: –")
        self.lbl_connection.setFrameShape(QFrame.Panel); self.lbl_connection.setFrameShadow(QFrame.Sunken)
        self.lbl_fps   = QLabel("FPS  : –")
        self.lbl_loss  = QLabel("Loss %: –")

        for lbl in (self.lbl_connection, self.lbl_fps, self.lbl_loss):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font: 10pt 'Consolas'; padding:4px;")

        vbox.addWidget(self.lbl_connection); vbox.addWidget(self.lbl_fps); vbox.addWidget(self.lbl_loss); vbox.addStretch(1)
        self.setWidget(inner)

        # --- polling timer ---------------------------------------
        self._timer = QTimer(self)
        self._timer.setInterval(self.POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()
        self._refresh()                       # 첫 업데이트

    # ------------------------------------------------ set_controller ★ NEW
    def set_controller(self, controller: Optional[CameraController]) -> None:
        """
        MainWindow 가 카메라 연결/교체 시 호출한다.
        None 이면 ‘Disconnected’ 상태로 전환된다.
        """
        self._controller = controller
        self._refresh()                       # 즉시 반영

    # ------------------------------------------------ _refresh
    def _refresh(self) -> None:
        """Pull stats from controller and update labels."""
        ctrl = self._controller
        if ctrl is None or not isinstance(ctrl, CameraController):
            self._show_disconnected()
            return

        connected = ctrl.is_connected()
        grabbing  = ctrl.is_grabbing()

        # connection label
        if connected:
            state_txt = "Grabbing" if grabbing else "Idle"
            self.lbl_connection.setText(f"Camera: {state_txt}")
            clr = "#98fb98" if grabbing else "#f0e68c"
        else:
            self._show_disconnected()
            return
        self.lbl_connection.setStyleSheet(
            f"font: 10pt 'Consolas'; padding:4px; background:{clr};")

        # stats
        stats = getattr(ctrl, "stats", {})
        fps  = stats.get("fps", 0.0)
        loss = stats.get("frame_loss", 0.0) * 100.0
        self.lbl_fps.setText (f"FPS  : {fps:6.2f}")
        self.lbl_loss.setText(f"Loss %: {loss:6.2f}")

    # ------------------------------------------------ helper
    def _show_disconnected(self) -> None:
        self.lbl_connection.setText("Camera: Disconnected")
        self.lbl_connection.setStyleSheet(
            "font: 10pt 'Consolas'; padding:4px; background:#f08080;")
        self.lbl_fps.setText ("FPS  : –")
        self.lbl_loss.setText("Loss %: –")
