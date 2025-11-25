#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ErrorImagesTab – ErrorImageCaptured 이벤트를 받아 목록/썸네일/프리뷰를 보여주는 탭
다크테마 스타일을 명시해 Gallery 탭에서 흰색 박스가 보이지 않도록 함.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from PyQt5.QtCore import QSize, Qt, QTimer
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QLabel
)

from src.core import events

logger = logging.getLogger(__name__)


class ErrorImagesTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # ── Left: list ───────────────────────────────────────
        self.list = QListWidget(objectName="ErrList")
        self.list.setIconSize(QSize(72, 72))
        self.list.setUniformItemSizes(True)
        self.list.setMinimumWidth(220)

        # ── Right: preview ───────────────────────────────────
        self.preview = QLabel("No file", alignment=Qt.AlignCenter, objectName="ErrPreview")
        self.preview.setMinimumSize(360, 240)
        self.preview.setWordWrap(True)

        # ── Layout ───────────────────────────────────────────
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        lay.addWidget(self.list, 1)
        lay.addWidget(self.preview, 2)

        # ── Dark styles (fixes the white box) ────────────────
        self.setStyleSheet("""
            QListWidget#ErrList {
                background: #2A3339;
                border: 1px solid #3A4A55;
                border-radius: 8px;
                color: #F0F4F8;
                outline: 0;
            }
            QListWidget#ErrList::item { padding: 6px; }
            QListWidget#ErrList::item:selected {
                background: #607D8B; color: #F0F4F8;
            }
            QLabel#ErrPreview {
                background: #2A3339;
                border: 1px solid #3A4A55;
                border-radius: 8px;
                color: #90A4AE;        /* placeholder 텍스트 색 */
            }
        """)

        # Signals
        self.list.currentItemChanged.connect(self._on_curr_changed)
        events.subscribe(events.ErrorImageCaptured, self._on_error_img)

    # ------------------------------------------------------------------
    def _on_error_img(self, ev: events.ErrorImageCaptured) -> None:
        # 1) 경로 해석(절대경로 우선, 없으면 meta 기반 복구)
        path: Path | None = ev.path
        if not path:
            ip = ev.meta.get("image_path", "")
            path = Path(ip) if ip else None
        if path and not path.is_absolute():
            try:
                path = path.resolve()
            except Exception:
                pass

        # 2) UI 스레드에서 갱신
        def _update_ui():
            # 경로가 없어도(또는 파일이 없어도) 항목은 추가해 사용자에게 피드백 제공
            display_name = path.name if isinstance(path, Path) else (ev.meta.get("image_path") or "unknown")
            item = QListWidgetItem(display_name)
            item.setData(Qt.UserRole, (path, ev.meta))

            # 썸네일 시도 (실패해도 아이템은 유지)
            thumb_set = False
            if isinstance(path, Path) and path.exists():
                pix = QPixmap(str(path))
                if not pix.isNull():
                    thumb = pix.scaled(72, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    item.setIcon(QIcon(thumb))
                    thumb_set = True

            self.list.addItem(item)
            if self.list.count() == 1:
                self.list.setCurrentItem(item)

            if not thumb_set and isinstance(path, Path):
                # 프리뷰 텍스트 모드라도 경로는 보이도록
                self.preview.setText(str(path))

        QTimer.singleShot(0, _update_ui)

    # ------------------------------------------------------------------
    def _on_curr_changed(self, item: QListWidgetItem | None) -> None:
        if not item:
            self.preview.setText("No file")
            self.preview.setPixmap(QPixmap())
            return

        path, meta = item.data(Qt.UserRole)
        pix = QPixmap(str(path))
        if pix.isNull():
            # 썸네일을 못 만들면 경로 텍스트만 표기
            self.preview.setText(str(path))
            self.preview.setPixmap(QPixmap())
            return

        self.preview.setText("")  # 텍스트 제거
        self.preview.setPixmap(pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # 프리뷰 영역 리사이즈 시 이미지도 함께 리스케일
    def resizeEvent(self, e):
        super().resizeEvent(e)
        pm = self.preview.pixmap()
        if pm:
            self.preview.setPixmap(pm.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
