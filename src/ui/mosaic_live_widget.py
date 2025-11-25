# src/ui/mosaic_live_widget.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MosaicLiveWidget
────────────────
카메라 ID 목록을 받아 그리드(GridLayout)로 QLabel 타일을 만들고,
update_frame(cam_id, qimg) 로 각 셀에 실시간 영상을 표시한다.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QWidget, QLabel, QGridLayout


class MosaicLiveWidget(QWidget):
    """다중 카메라 라이브-뷰 그리드."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._grid  = QGridLayout(self)
        self._grid.setSpacing(2)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._cells: Dict[str, QLabel] = {}      # cam_id ↦ QLabel

    # ─────────────────────────────────────────── public API
    def set_camera_ids(self, cam_ids: List[str]) -> None:
        """
        현재 표시할 카메라 ID 목록을 설정.
        - 빠진 ID는 셀 제거
        - 새 ID는 셀 생성
        """
        # 1) 제거 대상
        for cid in list(self._cells):
            if cid not in cam_ids:
                self._grid.removeWidget(self._cells[cid])
                self._cells.pop(cid).deleteLater()

        # 2) 추가 대상
        for cid in cam_ids:
            if cid not in self._cells:
                lbl = QLabel(f"{cid}\n(no signal)", self,
                             alignment=Qt.AlignCenter)
                lbl.setStyleSheet("background:black; color:#bbb; font-size:11px;")
                lbl.setMinimumSize(QSize(160, 120))
                lbl.setScaledContents(True)
                self._cells[cid] = lbl

        # 3) 그리드 재배치
        self._relayout()

    def update_frame(self, cam_id: str, img: QImage) -> None:
        """GrabWorker.frame_ready → (cam_id, QImage) 슬롯에서 호출."""
        lbl = self._cells.get(cam_id)
        if not lbl or img.isNull():
            return
        # QLabel.scaledContents 띄운 상태라 그냥 Pixmap 교체
        pix = QPixmap.fromImage(img)
        lbl.setPixmap(pix)

    # ─────────────────────────────────────────── helpers
    def _relayout(self) -> None:
        """셀 개수에 맞춰 그리드 행/열 재계산."""
        for i in reversed(range(self._grid.count())):
            self._grid.itemAt(i).widget().setParent(None)

        n = len(self._cells)
        if n == 0:
            return
        cols = int(math.ceil(math.sqrt(n)))
        for idx, (cid, lbl) in enumerate(self._cells.items()):
            r, c = divmod(idx, cols)
            self._grid.addWidget(lbl, r, c)
