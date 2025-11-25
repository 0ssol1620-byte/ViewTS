#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/ui/device_manager_dock.py
──────────────────────────────────────────────────────────────────────────────
Realtime *Device Manager* dock monitoring all registered `CameraController`
objects.

What’s new (2025-05-20)
──────────────────────────────────────────────────────────────────────────────
1. **Clickable camera list** ― `QListWidget.itemClicked` → emit `device_selected`
   so `MainWindow` (or any parent) can switch the active-camera instantly.
2. **Status icons** ― coloured LED-style circles showing *Idle* vs *Grabbing*
   state for better UX.

This dock polls each controller **only while visible** (1 s interval).
All polling is exception-safe and uses a tiny cached attribute (`_grabbing`)
to minimise native driver calls.
"""
from __future__ import annotations

import logging
from typing import Dict

from PyQt5.QtCore    import Qt, QTimer, pyqtSignal
from PyQt5.QtGui     import QIcon, QPixmap, QColor
from PyQt5.QtWidgets import QDockWidget, QListWidget, QListWidgetItem

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
def _make_led(col: QColor, size: int = 10) -> QIcon:
    """Generate an in-memory coloured circle icon."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    pm_p = pm.paintEngine()
    if pm_p:  # suppress linter; ensures QPainter lifetime
        pass
    pm.fill(Qt.transparent)
    painter = None
    try:
        from PyQt5.QtGui import QPainter, QBrush
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setBrush(QBrush(col))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, size, size)
    finally:
        if painter:
            painter.end()
    return QIcon(pm)

_ICON_IDLE     = _make_led(QColor("#aaaaaa"))
_ICON_GRABBING = _make_led(QColor("#2ecc71"))   # green


class DeviceManagerDock(QDockWidget):
    """
    Monitor the global `controller_pool.controllers` dict.

    Signals
    -------
    device_selected(str)
        Emitted when a user clicks a camera row.
    """

    device_selected = pyqtSignal(str)
    POLL_MS = 1000

    # ------------------------------------------------------------------ init
    def __init__(self, parent=None) -> None:
        super().__init__("Device Manager", parent)
        self.list = QListWidget()
        self.setWidget(self.list)

        from src.core.controller_pool import controllers as _pool
        self._pool: Dict[str, "CameraController"] = _pool

        self.list.itemClicked.connect(self._on_item_clicked)

        # polling timer (active only when dock visible)
        self._timer = QTimer(self, timeout=self._on_timeout)
        self.visibilityChanged.connect(self._on_visibility_changed)

        self.refresh()  # initial populate

    # ---------------------------------------------------------------- public
    def refresh(self) -> None:
        """Refresh list safely using cached `_grabbing` flags."""
        self.list.clear()
        for cid, ctrl in self._pool.items():
            grabbing = getattr(ctrl, "_grabbing", False)
            item = QListWidgetItem(cid)
            item.setData(Qt.UserRole, cid)
            item.setIcon(_ICON_GRABBING if grabbing else _ICON_IDLE)
            item.setToolTip("Grabbing" if grabbing else "Idle")
            self.list.addItem(item)

    # ---------------------------------------------------------------- slots
    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        """Emit clicked cam_id so MainWindow can switch active controller."""
        cid = item.data(Qt.UserRole)
        if cid:
            logger.debug("DeviceManagerDock clicked %s", cid)
            self.device_selected.emit(cid)

    def _on_visibility_changed(self, visible: bool) -> None:
        if visible:
            self._timer.start(self.POLL_MS)
        else:
            self._timer.stop()

    def _on_timeout(self) -> None:
        """While visible, poll `is_grabbing()` and update cache + UI."""
        changed = False
        for cid, ctrl in list(self._pool.items()):
            try:
                grabbing = bool(ctrl.is_grabbing())  # native call
            except Exception as exc:
                logger.debug("[%s] is_grabbing failed: %s", cid, exc)
                grabbing = False
            if getattr(ctrl, "_grabbing", None) != grabbing:
                ctrl._grabbing = grabbing
                changed = True
        if changed:
            self.refresh()
