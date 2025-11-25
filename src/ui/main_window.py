#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/ui/main_window.py - Simplified and stable version

애플리케이션의 메인 윈도우.
모든 초기화 순서 제어는 main.py에서 담당하므로, 이 파일은
UI 구성과 이벤트 처리에만 집중합니다.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

# --- PyQt5 Imports ---
from PyQt5.QtCore import Qt, QTimer, QSize, QSettings, QByteArray, pyqtSlot
from PyQt5.QtGui import QIcon, QPixmap, QImage, QColor, QBrush, QCloseEvent, QFont, QDesktopServices
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTextEdit, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QFileDialog, QMessageBox, QStatusBar, QToolBar, QAction,
    QSizePolicy, QProgressBar, QDockWidget, QComboBox, QStackedLayout, QMenu, QFrame
)

# --- Core Logic and UI Widget Imports ---
from src.core.sequence_types import Sequence
from src.core.camera_controller import CameraController, CameraConnectionError
from src.core import controller_pool
from src.core.multi_runner_manager import MultiRunnerManager
from src.ui.sequence_editor_widget import SequenceEditorWidget
from src.ui.device_manager_dock import DeviceManagerDock
from src.ui.camera_status_dock import CameraStatusDock
from src.ui.mosaic_live_widget import MosaicLiveWidget
from src.ui.camera_settings_dialog import CameraSettingsDialog
from src.ui.live_view_widget import LiveViewWidget
from src.ui.error_images_tab import ErrorImagesTab
from src.core.events    import ErrorImageCaptured, subscribe, unsubscribe
import numpy as np
logger = logging.getLogger(__name__)

# --- Type Checking Placeholders ---
if TYPE_CHECKING:
    from src.core.sequence_runner import SequenceRunner
    from src.ui.grab_worker import GrabWorker
else:
    SequenceRunner = object
    GrabWorker = object
class _DataclassJSONEncoder(json.JSONEncoder):
    def default(self, o):
        return asdict(o) if is_dataclass(o) else super().default(o)

def get_icon(name: str) -> QIcon:
    base = Path(__file__).parent / "icons"
    for ext in (".svg", ".png"):
        p = base / f"{name}{ext}"
        if p.exists():
            return QIcon(str(p))
    return QIcon()

def _icon(name: str) -> QIcon:
    for ext in (".svg", ".png"):
        p = Path(__file__).parent / "icons" / f"{name}{ext}"
        if p.exists():
            return QIcon(str(p))
    return QIcon()
# ══════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    APP_NAME = "Vieworks Test Suite"
    ORG_NAME = "Vieworks"

    def __init__(self) -> None:
        super().__init__()
        # 커스텀 타이틀 바 설정
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.old_pos = None  # For draggable title bar

        self.title_bar = QWidget(self)
        self.title_bar.setFixedHeight(40)
        self.title_bar.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1F2A30, stop:1 #2D3A42);
                color: #F0F4F8;
                font-size: 15px;
                font-weight: 600;
                padding: 0px 12px;
                border-bottom: 1px solid #455A64;
            }
            QComboBox {
                background: #455A64;
                color: #F0F4F8;
                border: 1px solid #90A4AE;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background: #1F2A30;
                color: #F0F4F8;
                selection-background-color: #607D8B;
            }
        """)
        layout = QHBoxLayout(self.title_bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.title_label = QLabel("Vieworks Test Suite [v1.0]")
        layout.addWidget(self.title_label)

        # Add menus as buttons with dropdowns for luxurious integration
        self.file_menu_btn = QPushButton("&File")
        self.file_menu = QMenu(self)
        self.file_menu_btn.setMenu(self.file_menu)
        layout.addWidget(self.file_menu_btn)

        self.camera_menu_btn = QPushButton("&Camera")
        self.camera_menu = QMenu(self)
        self.camera_menu_btn.setMenu(self.camera_menu)
        layout.addWidget(self.camera_menu_btn)

        self.view_menu_btn = QPushButton("&View")
        self.view_menu = QMenu(self)
        self.view_menu_btn.setMenu(self.view_menu)
        layout.addWidget(self.view_menu_btn)

        self.run_menu_btn = QPushButton("&Run")
        self.run_menu = QMenu(self)
        self.run_menu_btn.setMenu(self.run_menu)
        layout.addWidget(self.run_menu_btn)

        self.help_menu_btn = QPushButton("&Help")
        self.help_menu = QMenu(self)
        self.help_menu_btn.setMenu(self.help_menu)
        layout.addWidget(self.help_menu_btn)

        # Style menu buttons professionally
        menu_btn_style = """
            QPushButton {
                background: transparent;
                color: #F0F4F8;
                padding: 8px 16px;
                font-weight: 600;
                border: none;
            }
            QPushButton:hover {
                background: #455A64;
            }
        """
        for btn in [self.file_menu_btn, self.camera_menu_btn, self.view_menu_btn, self.run_menu_btn, self.help_menu_btn]:
            btn.setStyleSheet(menu_btn_style)

        layout.addStretch()

        self.cmb_active_cam = QComboBox(minimumWidth=240)
        layout.addWidget(QLabel("Active Cam:", styleSheet="color: #B0BEC5; font-size: 13px; padding-right: 12px;"))
        layout.addWidget(self.cmb_active_cam)

        # Minimize button
        self.min_btn = QPushButton("_", self.title_bar)
        self.min_btn.setFixedWidth(40)
        self.min_btn.setStyleSheet("""
            QPushButton {
                background: #F9A825;
                color: #1F2A30;
                border-radius: 4px;
                padding: 4px;
                font-weight: 700;
                font-size: 14px;
            }
            QPushButton:hover {
                background: #FFB74D;
            }
        """)
        self.min_btn.clicked.connect(self.showMinimized)
        layout.addWidget(self.min_btn)

        # Maximize/Restore button
        self.max_btn = QPushButton("□", self.title_bar)
        self.max_btn.setFixedWidth(40)
        self.max_btn.setStyleSheet("""
            QPushButton {
                background: #2E7D32;
                color: #F0F4F8;
                border-radius: 4px;
                padding: 4px;
                font-weight: 700;
                font-size: 14px;
            }
            QPushButton:hover {
                background: #66BB6A;
            }
        """)
        self.max_btn.clicked.connect(self._toggle_maximize)
        layout.addWidget(self.max_btn)

        # Close button
        close_btn = QPushButton("×", self.title_bar)
        close_btn.setFixedWidth(40)
        close_btn.setStyleSheet("""
            QPushButton {
                background: #D32F2F;
                color: #F0F4F8;
                border-radius: 4px;
                padding: 4px;
                font-weight: 700;
                font-size: 14px;
            }
            QPushButton:hover {
                background: #EF5350;
            }
        """)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        # 상태 변수
        self.controller: Optional[CameraController] = None
        self.runner_mgr: Optional[MultiRunnerManager] = None
        self.sequence_runner: Optional[SequenceRunner] = None
        self._grab_worker: Optional[GrabWorker] = None
        self.current_sequence = Sequence(name="Untitled")
        self.current_sequence_path: Optional[str] = None
        self.is_sequence_modified: bool = False
        self._mosaic_enabled: bool = False
        self._seq_running_ui: bool = False
        self._live_view_before_run: bool = False

        # Create docks early
        self.device_manager_dock = DeviceManagerDock(self)
        self.device_manager_dock.setFeatures(
            QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.RightDockWidgetArea, self.device_manager_dock)
        self.device_manager_dock.hide()

        self.camera_status_dock = CameraStatusDock(None, self)
        self.camera_status_dock.setFeatures(
            QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.RightDockWidgetArea, self.camera_status_dock)
        self.camera_status_dock.hide()

        # UI 구성
        self._create_actions()
        self._populate_menus()
        self._build_ui()
        self._create_statusbar()
        self.resize(1440, 900)

        # 설정 및 타이틀
        self._load_settings()
        self._update_window_title()

        # 에러 이미지 이벤트 구독
        subscribe(ErrorImageCaptured, self._on_error_image_event)

    def _populate_menus(self):
        # File menu
        self.file_menu.addActions([self.act_new, self.act_open, self.act_save, self.act_save_as])
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.act_export)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.act_exit)

        # Camera menu
        self.camera_menu.addActions([self.act_refresh, self.act_disconnect_all])
        self.camera_menu.addSeparator()
        self.camera_menu.addActions([self.act_settings, self.act_grab_start, self.act_grab_stop])

        # View menu
        self.view_menu.addActions(
            [self.act_toggle_dev_mgr, self.act_toggle_cam_stat, self.act_mosaic_toggle, self.act_live_all])

        # Run menu
        self.run_menu.addActions([self.act_run, self.act_stop])

        # Help menu
        self.help_menu.addAction(self.act_about)

    def _toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
            self.max_btn.setText("□")
        else:
            self.showMaximized()
            self.max_btn.setText("❐")  # Simple restore symbol (adjust if font doesn't support)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and event.pos().y() < self.title_bar.height():
            title_pos = self.title_bar.mapFromParent(event.pos())
            child = self.title_bar.childAt(title_pos)
            if child is None or child == self.title_label:
                self.old_pos = event.globalPos()

    def mouseMoveEvent(self, event):
        if self.old_pos:
            delta = event.globalPos() - self.old_pos
            self.move(self.pos() + delta)
            self.old_pos = event.globalPos()

    def mouseReleaseEvent(self, event):
        self.old_pos = None

    def mouseDoubleClickEvent(self, event):
        if event.pos().y() < self.title_bar.height():
            self._toggle_maximize()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.title_bar.setGeometry(0, 0, self.width(), 40)

    def _is_sequence_running(self) -> bool:
        """
        MultiRunnerManager / SequenceRunner / UI 플래그 어느 쪽이든
        실행 중이면 True. 각 is_running() 이 None 을 돌려도 bool() 로
        강제 변환해 항상 bool 값을 반환한다.
        """
        return bool(
            getattr(self, "_seq_running_ui", False)
            or (self.runner_mgr and bool(self.runner_mgr.is_running()))
            or (self.sequence_runner and bool(self.sequence_runner.is_running()))
        )

    def _on_error_image_event(self, ev: ErrorImageCaptured) -> None:  # 이벤트 버스 콜백
        # Qt 스레드 안전 – GUI 업데이트는 main-thread 에서
        QTimer.singleShot(0, lambda e=ev: self._add_error_row(e))

    @pyqtSlot(str, dict)
    def _on_error_image_signal(self, path: str, meta: dict) -> None:
        # 1) 절대경로 보정
        p = Path(path) if path else None
        if p and not p.is_absolute():
            try:
                p = Path(meta.get("image_path", str(p))).resolve()
            except Exception:
                pass

        # 2) 로그 테이블 갱신
        ev = ErrorImageCaptured(p if p else None, meta)
        self._add_error_row(ev)

        # 3) 갤러리에도 직접 반영 (이벤트버스가 늦거나 누락되어도 보장)
        try:
            if hasattr(self, "err_gallery") and self.err_gallery:
                # 내부 콜백을 그대로 재사용 (메인스레드)
                self.err_gallery._on_error_img(ev)
        except Exception:
            pass

    @pyqtSlot(str, object)
    def _on_sequence_preview_frame(self, cam_id: str, frame: np.ndarray) -> None:
        """
        [수정됨] SequenceRunner로부터 실시간 프리뷰 프레임을 수신하여 Live View에 표시합니다.
        """
        # ★★★ [수정] 시퀀스 실행 중 프리뷰를 막던 잘못된 조건문을 제거합니다. ★★★
        # if self._is_sequence_running():
        #     return

        # 이제 아래 코드는 시퀀스 실행 중에도 정상적으로 실행됩니다.

        # Live-View 스택이 숨겨져 있다면 다시 보이도록 설정합니다.
        if self.live_stack.currentIndex() == 0:
            self.live_stack.setCurrentIndex(1)

        # ── 모자이크(다중 카메라) 뷰 모드일 경우 ──────────────────
        if self._mosaic_enabled and len(self.live_widget._tiles) > 1:
            try:
                # 프레임을 보낸 카메라(cam_id)에 해당하는 타일을 찾습니다.
                idx = next(
                    i for i, c in enumerate(self.live_widget._ctrls)
                    if c.cam_id == cam_id
                )
                # 해당 타일에 프레임을 업데이트합니다.
                self.live_widget._tiles[idx].update_frame(frame)
            except (StopIteration, IndexError):
                # 매핑에 실패한 경우 (드문 경우), 첫 번째 타일에라도 표시합니다.
                if self.live_widget._tiles:
                    self.live_widget._tiles[0].update_frame(frame)

        # ── 단일 카메라 뷰 모드일 경우 ──────────────────────
        elif self.live_widget._tiles:
            # 첫 번째 (그리고 유일한) 타일에 프레임을 업데이트합니다.
            self.live_widget._tiles[0].update_frame(frame)
    def _add_error_row(self, ev: ErrorImageCaptured) -> None:
        row = self.err_table.rowCount()
        self.err_table.insertRow(row)

        ts_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # --- 테이블 아이템 생성 ---
        item_timestamp = QTableWidgetItem(ts_str)
        item_step = QTableWidgetItem(str(ev.meta.get("step_name", "N/A")))

        # ★★★ [수정] 에러 메시지와 Traceback 정보 추출 ★★★
        error_message = str(ev.meta.get("message", "No message"))
        full_traceback = str(ev.meta.get("traceback", "No traceback available"))

        item_message = QTableWidgetItem(error_message)
        item_path = QTableWidgetItem(str(ev.path) if ev.path else "N/A")

        # ★★★ [수정] 툴팁에 전체 Traceback 정보 설정 ★★★
        # 사용자가 에러 메시지 위에 마우스를 올리면 전체 에러 로그를 볼 수 있습니다.
        tooltip_text = f"Exception Type: {ev.meta.get('exception_type', 'Unknown')}\n\n{full_traceback}"
        item_message.setToolTip(tooltip_text)

        # --- 테이블에 아이템 설정 ---
        self.err_table.setItem(row, 0, item_timestamp)
        self.err_table.setItem(row, 1, item_step)
        # ★★★ [수정] 새로운 컬럼에 에러 메시지 아이템 추가 ★★★
        self.err_table.setItem(row, 2, item_message)
        self.err_table.setItem(row, 3, item_path)

        # 최근 1000개만 보관
        if self.err_table.rowCount() > 1000:
            self.err_table.removeRow(0)

        # 사용자가 파일 경로를 더블클릭할 수 있도록 열 너비 자동 조절
        self.err_table.resizeColumnsToContents()

    def _open_error_image(self, row: int, col: int) -> None:
        # Columns: [0] Timestamp, [1] Step, [2] Error Message, [3] File Path
        path_item = self.err_table.item(row, 3)
        if not path_item:
            return
        path_text = path_item.text().strip()
        if not path_text or path_text.lower() == "n/a":
            return
        try:
            QDesktopServices.openUrl(Path(path_text).as_uri())
        except Exception:
            pass

    def post_init_setup(self):
        logger.info("UI 상태 초기화 및 동기화 시작…")

        self.controller = controller_pool.first_controller()

        if self.controller:
            logger.info("기본 컨트롤러: %s", self.controller.cam_id)
            self.sequence_editor.set_controller(self.controller)
            self.camera_status_dock.set_controller(self.controller)
        else:
            logger.warning("초기 연결된 카메라 없음.")
            self.sequence_editor.set_controller(None)
            self.camera_status_dock.set_controller(None)

        self._refresh_device_list()
        self._update_ui_states()
        logger.info("UI 상태 동기화 완료.")

    def _build_ui(self) -> None:
        # 전역 다크 테마 스타일
        self.setStyleSheet("""
            QWidget { background: #1F2A30; color: #F0F4F8; font-size: 13px; }
            QTabWidget::pane { border: none; background: transparent; }
            QTabBar::tab {
                background: #455A64; color: #F0F4F8;
                border-radius: 6px 6px 0 0; padding: 12px 24px; margin-right: 6px;
            }
            QTabBar::tab:selected { background: #607D8B; color: #F0F4F8; }
            QSplitter::handle { background: #455A64; width: 12px; border-radius: 4px; }
        """)

        # ── 최상위 레이아웃 ─────────────────────────────────────
        cw = QWidget(self)
        self.setCentralWidget(cw)

        main_lay = QVBoxLayout(cw)
        main_lay.setContentsMargins(0, 0, 0, 0)

        main_lay.addWidget(self.title_bar)

        # Content area
        content_widget = QWidget()
        content_lay = QHBoxLayout(content_widget)
        content_lay.setContentsMargins(24, 24, 24, 24)
        main_lay.addWidget(content_widget)

        splitter = QSplitter(Qt.Horizontal, self)
        content_lay.addWidget(splitter)

        # ── 좌측: Sequence Editor ─────────────────────────────
        self.sequence_editor = SequenceEditorWidget(self)
        splitter.addWidget(self.sequence_editor)

        # ── 우측: 탭 스택 영역 ────────────────────────────────
        right_pan = QWidget()
        r_lay = QVBoxLayout(right_pan)
        r_lay.setContentsMargins(12, 12, 12, 12)
        splitter.addWidget(right_pan)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1.5)
        splitter.setSizes([self.width() // 2.5, self.width() * 1.5 // 2.5])

        self.tabs = QTabWidget()
        r_lay.addWidget(self.tabs)

        # ──────────────────────────────────────────────────────
        #  (1) Live View 탭
        # ──────────────────────────────────────────────────────
        live_tab = QWidget()
        live_tab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        lv_lay = QVBoxLayout(live_tab)
        lv_lay.setContentsMargins(0, 0, 0, 0)

        self.live_stack = QStackedLayout()
        lv_lay.addLayout(self.live_stack)

        # ― ① Disconnected Placeholder ―
        placeholder = QLabel("Camera Disconnected", alignment=Qt.AlignCenter)
        placeholder.setMinimumSize(360, 203)
        placeholder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        placeholder.setStyleSheet("""
            background: #455A64; color: #F0F4F8; font-size: 15px; font-weight: 500;
            border-radius: 8px; padding: 24px;
        """)
        self.live_stack.addWidget(placeholder)

        # ― ② Live Feed Widget ―
        self.live_widget = LiveViewWidget()
        self.live_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.live_stack.addWidget(self.live_widget)

        self.live_stack.setCurrentIndex(0)  # 기본은 placeholder
        self.tabs.addTab(live_tab, _icon("live_view"), "Live View")

        # ──────────────────────────────────────────────────────
        #  (2) Logs 탭
        # ──────────────────────────────────────────────────────
        log_tab = QWidget()
        log_lay = QVBoxLayout(log_tab)

        self.log_view = QTextEdit(readOnly=True, lineWrapMode=QTextEdit.NoWrap)
        self.log_view.setFont(QFont("Inter", 13, 500))
        self.log_view.setStyleSheet("""
            background: #455A64; color: #F0F4F8; border: none;
            padding: 12px; line-height: 1.6;
        """)
        log_lay.addWidget(self.log_view)
        self.tabs.addTab(log_tab, _icon("logs"), "Logs")

        # ──────────────────────────────────────────────────────
        #  (3) Results 탭
        # ──────────────────────────────────────────────────────
        self._build_results_tab()

        # ──────────────────────────────────────────────────────
        #  (4) Error Images 탭
        # ──────────────────────────────────────────────────────
        self._build_error_images_tab()

    def _build_results_tab(self) -> None:
        tab = QWidget()
        lay = QVBoxLayout(tab)
        self.results_table = QTableWidget(0, 6, self)
        self.results_table.setHorizontalHeaderLabels(
            ["Step #", "Name", "Action", "Status", "Time (ms)", "Details"]
        )
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setStyleSheet("""
            QTableWidget {
                background: #455A64;
                color: #F0F4F8;
                border: 1px solid #607D8B;
                border-radius: 8px;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #607D8B;
            }
            QTableWidget::item:selected {
                background: #607D8B;
                color: #F0F4F8;
            }
            QHeaderView::section {
                background: #455A64;
                color: #F0F4F8;
                padding: 8px;
                border: none;
            }
        """)
        h = self.results_table.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.ResizeToContents)
        h.setSectionResizeMode(5, QHeaderView.Stretch)
        lay.addWidget(self.results_table)
        self.tabs.addTab(tab, _icon("results"), "Results")

        # src/ui/main_window.py 내의 MainWindow 클래스

    def _build_error_images_tab(self) -> None:
        """
        Error Images 탭 (디자인 버전)
        - 상단 주 탭과 12px 간격
        - 보조 탭(Logs / Gallery)을 '카드' 컨테이너 안에 배치
        - 탭 버튼/테이블을 컴팩트한 사이즈로 스타일링
        """
        outer_tab = QWidget()
        outer_lay = QVBoxLayout(outer_tab)
        outer_lay.setContentsMargins(12, 12, 12, 12)
        outer_lay.setSpacing(0)

        # 주 탭과 살짝 띄우기
        outer_lay.addSpacing(12)

        # ── 카드 컨테이너 ──────────────────────────────────────
        card = QFrame()
        card.setObjectName("ErrCard")
        card.setStyleSheet("""
            QFrame#ErrCard {
                background: #2A3339;
                border: 1px solid #3A4A55;
                border-radius: 10px;
            }
        """)
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(10, 10, 10, 10)
        card_lay.setSpacing(10)

        # ── 보조 탭(Logs / Gallery) ────────────────────────────
        sub_tabs = QTabWidget()
        sub_tabs.setStyleSheet("""
            QTabBar::tab {
                padding: 6px 12px;          /* 컴팩트 */
                font-size: 12px;
                min-height: 26px;
                background: #546E7A;
                color: #F0F4F8;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 4px;
            }
            QTabBar::tab:selected { background: #607D8B; }
            QTabWidget::pane { border: none; }
        """)
        card_lay.addWidget(sub_tabs)

        # ── (A) Logs ───────────────────────────────────────────
        logs_widget = QWidget()
        logs_lay = QVBoxLayout(logs_widget)

        self.err_table = QTableWidget(0, 4, self)
        self.err_table.setHorizontalHeaderLabels(["Timestamp", "Step", "Error Message", "File Path"])
        self.err_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.err_table.verticalHeader().setVisible(False)
        self.err_table.setStyleSheet("""
            QTableWidget {
                background: #455A64;
                color: #F0F4F8;
                border: 1px solid #607D8B;
                border-radius: 8px;
                font-size: 12px;            /* 조금 더 작게 */
            }
            QTableWidget::item {
                padding: 6px;               /* 조금 더 작게 */
                border-bottom: 1px solid #607D8B;
            }
            QTableWidget::item:selected {
                background: #607D8B;
                color: #F0F4F8;
            }
            QHeaderView::section {
                background: #455A64;
                color: #F0F4F8;
                padding: 6px;               /* 조금 더 작게 */
                border: none;
                font-size: 12px;            /* 조금 더 작게 */
            }
        """)
        self.err_table.cellDoubleClicked.connect(self._open_error_image)
        h = self.err_table.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.ResizeToContents)
        h.setSectionResizeMode(3, QHeaderView.Stretch)
        logs_lay.addWidget(self.err_table)
        sub_tabs.addTab(logs_widget, "Logs")

        # ── (B) Gallery ────────────────────────────────────────
        self.err_gallery = ErrorImagesTab(self)
        sub_tabs.addTab(self.err_gallery, "Gallery")

        # 카드 컨테이너를 아우터 레이아웃에 추가
        outer_lay.addWidget(card)

        # 메인 탭에 등록
        self.tabs.addTab(outer_tab, _icon("error_images"), "Error Images")

    def _create_actions(self):
        # File 메뉴 액션
        self.act_new = QAction(get_icon("new"), "&New", self)
        self.act_open = QAction(get_icon("open"), "&Open", self)
        self.act_save = QAction(get_icon("save"), "&Save", self)
        self.act_save_as = QAction("Save &As…", self)
        self.act_exit = QAction("E&xit", self)

        # Camera 메뉴 액션
        self.act_refresh = QAction(get_icon("refresh"), "&Refresh Connections", self)
        self.act_disconnect_all = QAction(get_icon("disconnect"), "Disconnect A&ll", self)
        self.act_settings = QAction(get_icon("settings"), "Camera Settings…", self)
        self.act_grab_start = QAction(get_icon("grab_start"), "Grab Start", self)
        self.act_grab_stop = QAction(get_icon("grab_stop"), "Grab Stop", self)

        # Run 메뉴 액션
        self.act_run = QAction(get_icon("run"), "&Run Sequence", self)
        self.act_stop = QAction(get_icon("stop"), "&Stop", self)

        # 기타 액션들
        self.act_export = QAction(get_icon("export"), "Export Report", self)
        self.act_about = QAction("&About", self)
        self.act_toggle_dev_mgr = QAction("Device Manager", self, checkable=True)
        self.act_toggle_cam_stat = QAction("Camera Status", self, checkable=True)
        self.act_mosaic_toggle = QAction(get_icon("mosaic"), "Mosaic View", self, checkable=True)
        self.act_live_all = QAction(get_icon("live_all"), "Live – All Cams", self)

        # 단축키 및 시그널 연결
        self.act_new.setShortcut("Ctrl+N");
        self.act_new.triggered.connect(self._new_sequence)
        self.act_open.setShortcut("Ctrl+O");
        self.act_open.triggered.connect(self._open_sequence)
        self.act_save.setShortcut("Ctrl+S");
        self.act_save.triggered.connect(self._save_sequence)
        self.act_save_as.triggered.connect(self._save_sequence_as)
        self.act_exit.setShortcut("Ctrl+Q");
        self.act_exit.triggered.connect(self.close)

        self.act_refresh.triggered.connect(self._refresh_connections)
        self.act_disconnect_all.triggered.connect(self._disconnect_all_cameras)

        self.act_settings.triggered.connect(self._show_camera_settings)
        self.act_grab_start.triggered.connect(self._start_grab)
        self.act_grab_stop.triggered.connect(self._stop_grab)

        self.act_run.setShortcut("F5");
        self.act_run.triggered.connect(self._run_sequence)
        self.act_stop.setShortcut("Shift+F5");
        self.act_stop.triggered.connect(self._stop_sequence)

        self.act_export.triggered.connect(self._export_report)
        self.act_about.triggered.connect(self._about)
        self.act_toggle_dev_mgr.triggered.connect(self.device_manager_dock.setVisible)
        self.act_toggle_cam_stat.triggered.connect(self.camera_status_dock.setVisible)
        self.act_mosaic_toggle.toggled.connect(self._on_mosaic_toggled)
        self.act_live_all.triggered.connect(self._on_live_all_cams)
        self.cmb_active_cam.currentTextChanged.connect(self._on_active_cam_changed)

    def _refresh_connections(self):
        """Discover / re-discover every camera and resynchronise the UI.

        1. `controller_pool.connect_all(replace_existing=True)`
           → 내부적으로 기존 풀을 flush 후 모든 카메라를 다시 검색·연결합니다.
        2. `post_init_setup()`
           → 새로 연결된 컨트롤러를 기준으로 UI·액션 상태를 재설정합니다.

        실패 시 예외를 로깅하고, 사용자에게 QMessageBox로 즉시 알립니다.
        """
        logger.info("Manual connection refresh requested by user…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # (Re)discover & connect every detectable camera
            controller_pool.connect_all(replace_existing=True)
        except Exception as exc:
            logger.exception("Camera reconnection failed: %s", exc)
            QMessageBox.critical(
                self,
                "Refresh Connections – Error",
                f"Failed to refresh camera connections:\n{exc}",
            )
        finally:
            # Re-bind active controller & update all action states
            self.post_init_setup()
            QApplication.restoreOverrideCursor()

    def _create_menus(self):
        mb = self.menuBar()
        m_file = mb.addMenu("&File")
        m_file.addActions([self.act_new, self.act_open, self.act_save, self.act_save_as])
        m_file.addSeparator()
        m_file.addAction(self.act_export)
        m_file.addSeparator()
        m_file.addAction(self.act_exit)

        m_cam = mb.addMenu("&Camera")
        m_cam.addActions([self.act_refresh, self.act_disconnect_all])
        m_cam.addSeparator()
        m_cam.addActions([self.act_settings, self.act_grab_start, self.act_grab_stop])

        m_view = mb.addMenu("&View")
        m_view.addActions([self.act_toggle_dev_mgr, self.act_toggle_cam_stat])

        m_run = mb.addMenu("&Run")
        m_run.addActions([self.act_run, self.act_stop])

        mb.addMenu("&Help").addAction(self.act_about)

    def _create_toolbar(self):
        tb = QToolBar("Main Toolbar", self)
        tb.setIconSize(QSize(24, 24))
        tb.setStyleSheet("""
            QToolBar {
                background: #455A64;
                border: none;
                spacing: 12px;
                padding: 12px;
                border-radius: 8px;
            }
            QToolButton {
                background: #607D8B;
                color: #F0F4F8;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
                border: 1px solid #90A4AE;
            }
            QToolButton:hover {
                background: #B0BEC5;
            }
            QToolButton:disabled {
                background: #90A4AE;
                color: #F0F4F8;
                opacity: 0.5;
                border: 1px solid #B0BEC5;
            }
            QComboBox {
                background: #455A64;
                color: #F0F4F8;
                border: 1px solid #90A4AE;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
                image: url(:/icons/camera.svg);
            }
            QComboBox QAbstractItemView {
                background: #1F2A30;
                color: #F0F4F8;
                selection-background-color: #607D8B;
            }
        """)
        self.addToolBar(Qt.TopToolBarArea, tb)

        actions_to_add = [
            self.act_new, self.act_open, self.act_save, None,
            self.act_refresh, self.act_disconnect_all, None,
            self.act_grab_start, self.act_grab_stop, None,
            self.act_run, self.act_stop, None,
            self.act_mosaic_toggle, self.act_live_all
        ]
        for a in actions_to_add:
            if a:
                tb.addAction(a)
            else:
                tb.addSeparator()

        tb.addSeparator()
        self.cmb_active_cam = QComboBox(minimumWidth=240)
        tb.addWidget(QLabel("Active Cam:", styleSheet="color: #B0BEC5; font-size: 13px; padding-right: 12px;"))
        tb.addWidget(self.cmb_active_cam)

    def _create_statusbar(self):
        sb = QStatusBar(self)
        sb.setStyleSheet("""
            QStatusBar {
                background-color: #424242;
                color: #B0B0B0;
                font-size: 11px;
                spacing: 10px;
            }
            QStatusBar::item { border: none; }
        """)
        self.setStatusBar(sb)

        # ── Camera 상태 라벨 ────────────────────────────────
        self.lbl_conn = QLabel(" Camera: Disconnected ")
        self.lbl_conn.setStyleSheet("""
            padding: 4px 10px;
            border: 1px solid #607D8B;
            background: #2D3A42;
            color: #B0BEC5;
            border-radius: 4px;
            min-width: 180px;
        """)
        sb.addPermanentWidget(self.lbl_conn)

        # ── Progress bar ────────────────────────────────────
        self.progress = QProgressBar(maximumWidth=200)
        self.progress.setStyleSheet("""
            QProgressBar {
                background-color: #455A64;
                border: 1px solid #607D8B;
                border-radius: 4px;
                text-align: center;
                color: #F0F4F8;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                  stop:0 #2A6FD4, stop:1 #40C4FF);
                border-radius: 2px;
            }
        """)
        sb.addPermanentWidget(self.progress)
        self.progress.hide()

        # ── Loop 정보 라벨 ───────────────────────────────────
        self.lbl_loop = QLabel(" Loop: - ")
        self.lbl_loop.setStyleSheet("""
            padding: 4px 10px;
            border: 1px solid #607D8B;
            background: #2D3A42;
            color: #E8ECEF;
            border-radius: 4px;
            min-width: 120px;
        """)
        sb.addPermanentWidget(self.lbl_loop)

    def _animate_button(self, button: QPushButton):
        from PyQt5.QtCore import QPropertyAnimation, QEasingCurve
        anim = QPropertyAnimation(button, b"geometry")
        anim.setDuration(100)
        original = button.geometry()
        anim.setStartValue(original)
        anim.setEndValue(original.adjusted(-2, -2, 2, 2))
        anim.setEasingCurve(QEasingCurve.InOutQuad)
        anim.start(QPropertyAnimation.DeleteWhenStopped)

    def _ctrl_connected(self) -> bool:
        return self.controller is not None and self.controller.is_connected()

    def _ctrl_grabbing(self) -> bool:
        return self.controller is not None and self.controller.is_grabbing()

    def _refresh_device_list(self):
        id_list = controller_pool.list_ids()
        active_cam_id = self.controller.cam_id if self.controller else None

        self.cmb_active_cam.blockSignals(True)
        self.cmb_active_cam.clear()
        if id_list:
            self.cmb_active_cam.addItems(id_list)
            if active_cam_id and active_cam_id in id_list:
                self.cmb_active_cam.setCurrentText(active_cam_id)
        self.cmb_active_cam.blockSignals(False)

    @pyqtSlot(str)
    def _on_active_cam_changed(self, cam_id: str):
        if not cam_id: return
        new_controller = controller_pool.get_controller(cam_id)
        if new_controller and new_controller is not self.controller:
            self.controller = new_controller
            self.sequence_editor.set_controller(new_controller)
            self.camera_status_dock.set_controller(new_controller)
            self._update_ui_states()

    def _connect_all_cameras(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        controller_pool.flush()
        try:
            new_ids = controller_pool.connect_all(enable_param_cache=True)
            if not new_ids:
                QMessageBox.information(self, "Connect All", "No cameras found or failed to connect.")
            else:
                self.controller = controller_pool.get_controller(new_ids[0])
                if self.controller:
                    self.sequence_editor.set_controller(self.controller)
                    self.camera_status_dock.set_controller(self.controller)
        except Exception as e:
            QMessageBox.critical(self, "Connection Error", f"Failed to connect all cameras: {e}")
        finally:
            QApplication.restoreOverrideCursor()
            self._refresh_device_list()
            self._update_ui_states()

    # ------------------------------------------------------------------
    # 6) _disconnect_all_cameras  ─ Live-View 정지 추가
    # ------------------------------------------------------------------
    def _disconnect_all_cameras(self):
        """모든 카메라 연결을 해제하고 풀을 비웁니다."""
        logger.info("Disconnecting all cameras from UI...")
        self._stop_live_view()  # 🆕 먼저 Live-View 종료
        controller_pool.flush(disconnect=True)

        self.controller = None
        self.sequence_editor.set_controller(None)
        self.camera_status_dock.set_controller(None)
        self._refresh_device_list()
        self._update_ui_states()
        logger.info("All cameras disconnected.")


    def _connect_camera(self):
        QMessageBox.information(self, "Info", "Single connect is not implemented. Please use 'Connect All'.")

    def _disconnect_camera(self):
        if not self._ctrl_connected(): return
        self._stop_live_view()
        cam_id = self.controller.cam_id

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.controller.disconnect_camera()
        controller_pool.unregister(cam_id)

        remaining_ids = controller_pool.list_ids()
        self.controller = controller_pool.get_controller(remaining_ids[0]) if remaining_ids else None
        self.sequence_editor.set_controller(self.controller)
        self.camera_status_dock.set_controller(self.controller)

        QApplication.restoreOverrideCursor()
        self._refresh_device_list()
        self._update_ui_states()

    def _start_grab(self):
        if not self._ctrl_connected() or self._ctrl_grabbing(): return
        try:
            self.controller.start_grab()
            self._start_live_view([self.controller])
        except Exception as e:
            QMessageBox.warning(self, "Grab Error", f"Failed to start grabbing: {e}")
        self._update_ui_states()

    def _stop_grab(self):
        if not self._ctrl_grabbing(): return
        self._stop_live_view()
        try:
            self.controller.stop_grab()
        except Exception as e:
            QMessageBox.warning(self, "Grab Error", f"Failed to stop grabbing: {e}")
        self._update_ui_states()

    # ------------------------------------------------------------------
    # 2) _start_live_view  ─ controller.frame_ready → LiveViewWidget
    # ------------------------------------------------------------------
    def _start_live_view(self, controllers_to_view: List[CameraController]):
        """주어진 컨트롤러 리스트의 스트림을 LiveViewWidget에 연결."""
        if not controllers_to_view:
            return

        # 이전 세션 정리
        self._stop_live_view()

        # Live-View 위젯 활성화
        self.live_stack.setCurrentIndex(1)

        # Mosaic 토글 상태에 따라 1-up ↔ N-up 결정
        if self._mosaic_enabled and len(controllers_to_view) > 1:
            self.live_widget.start(controllers_to_view, mosaic=True)
        else:
            self.live_widget.set_controller(controllers_to_view[0])

    # ------------------------------------------------------------------
    # 3) _stop_live_view  ─ 모든 스트림·UI 초기화
    # ------------------------------------------------------------------
    def _stop_live_view(self):
        """Live-View 중지 및 ‘Disconnected’ 플레이스홀더로 복귀."""
        self.live_widget.stop()
        self.live_stack.setCurrentIndex(0)

    @pyqtSlot(str, QImage)
    def _on_live_frame(self, cam_id: str, image: QImage):
        if self._mosaic_enabled:
            self.live_widget.update_frame(cam_id, image)
        elif self.controller and cam_id == self.controller.cam_id:
            if not image.isNull():
                pixmap = QPixmap.fromImage(image).scaled(self.live_view_single.size(), Qt.KeepAspectRatio,
                                                         Qt.SmoothTransformation)
                self.live_view_single.setPixmap(pixmap)

    # ------------------------------------------------------------------
    # 4) _on_mosaic_toggled  ─ 버튼 토글-로직 단순화
    # ------------------------------------------------------------------
    def _on_mosaic_toggled(self, checked: bool):
        self._mosaic_enabled = checked
        if checked:
            self._on_live_all_cams()
        elif self.controller and self.controller.is_connected() and self.controller.is_grabbing():
            self._start_live_view([self.controller])
        else:
            self._stop_live_view()

    # ------------------------------------------------------------------
    # 5) _on_live_all_cams  ─ 모든 활성 카메라로 라이브뷰
    # ------------------------------------------------------------------
    def _on_live_all_cams(self):
        cam_ids = controller_pool.list_ids()
        if not cam_ids:
            QMessageBox.information(self, "Live View", "No cameras are connected.")
            return
        ctrls = [controller_pool.get_controller(cid)
                 for cid in cam_ids if controller_pool.get_controller(cid)]
        self._start_live_view(ctrls)

    # in src/ui/main_window.py

    # src/ui/main_window.py 파일의 MainWindow 클래스 내부에 위치해야 합니다.

    # src/ui/main_window.py

    def _on_any_runner_finished(self, status: str, detail: str) -> None:
        """
        [안정화 버전]
        - 러너 스레드가 '실제 종료'했을 때만 cleanup/재동기화를 실행
        - 실패/에러 종료 시 카메라 스트림을 강제로 유휴화하여 버튼 잠김 방지
        - 최종적으로 모든 버튼 상태를 재계산
        """
        from PyQt5.QtCore import QTimer

        # 0) 로그
        runner = self.sequence_runner or self.runner_mgr
        cam_id = runner.context.get('cam_id', 'System') if runner and hasattr(runner, "context") else 'System'
        self._append_log("INFO", f"[{cam_id}] Sequence finished → Status: {status}, Detail: {detail}")

        # 1) 러너가 아직 살아있으면 조금 뒤에 다시 시도 (Stop/Fail 직후 흔함)
        if runner and hasattr(runner, "isRunning") and runner.isRunning():
            self._append_log("WARNING", "Runner thread is still finishing… deferring cleanup until it actually stops.")
            QTimer.singleShot(200, lambda: self._on_any_runner_finished(status, detail))
            return

        # 2) MultiRunnerManager가 있다면 모두 끝났는지 보장
        if hasattr(self, "runner_mgr") and self.runner_mgr:
            try:
                if getattr(self.runner_mgr, "is_running", lambda: False)():
                    QTimer.singleShot(200, lambda: self._on_any_runner_finished(status, detail))
                    return
            except Exception:
                pass

        # 3) 실패/에러 종료 시: 스트림을 반드시 유휴화(Stop) → 버튼 잠김(그랩 중) 방지
        try:
            if status in ("Failed", "Error"):
                # Live-View UI 먼저 끄고
                try:
                    self._stop_live_view()
                except Exception:
                    pass
                # 컨트롤러 스트림 강제 정지 (flush/revoke 포함)
                if self.controller and self.controller.is_connected() and self.controller.is_grabbing():
                    self._append_log("WARNING",
                                     "Sequence ended with failure/error — forcing camera to idle (stop_grab_safe).")
                    try:
                        self.controller.stop_grab_safe(flush=True, revoke=True, wait_ms=500)
                    except Exception as e:
                        self._append_log("WARNING", f"stop_grab_safe failed: {e}")
        except Exception as e:
            self._append_log("WARNING", f"Fail-safe idle step raised: {e}")

        # 4) 러너 객체 정리 (이 시점에서는 절대 isRunning()이면 안 됨)
        try:
            if hasattr(self, "sequence_runner") and self.sequence_runner:
                try:
                    self.sequence_runner.deleteLater()
                finally:
                    self.sequence_runner = None

            if hasattr(self, "runner_mgr") and self.runner_mgr:
                try:
                    self.runner_mgr.deleteLater()
                finally:
                    self.runner_mgr = None
        except Exception as e:
            self._append_log("WARNING", f"Runner object cleanup warning: {e}")

        # 5) UI 플래그/프로그레스 리셋
        self._seq_running_ui = False
        try:
            self.progress.hide()
            self.lbl_loop.setText("Loop: -")
        except Exception:
            pass

        # 6) 컨트롤러 재동기화 (좀비 참조 방지)
        from src.core import controller_pool
        self._append_log("INFO", "Resynchronizing main controller reference and UI.")
        try:
            self.controller = controller_pool.first_controller()
            if self.controller:
                self._append_log("INFO", f"Main controller is now set to '{self.controller.cam_id}'.")
                if hasattr(self, "sequence_editor") and hasattr(self.sequence_editor, "set_controller"):
                    self.sequence_editor.set_controller(self.controller)
                if hasattr(self, "camera_status_dock") and hasattr(self.camera_status_dock, "set_controller"):
                    self.camera_status_dock.set_controller(self.controller)
            else:
                self._append_log("WARNING", "No active controllers available after sequence finished.")
                if hasattr(self, "sequence_editor") and hasattr(self.sequence_editor, "set_controller"):
                    self.sequence_editor.set_controller(None)
                if hasattr(self, "camera_status_dock") and hasattr(self.camera_status_dock, "set_controller"):
                    self.camera_status_dock.set_controller(None)
        except Exception as e:
            self._append_log("WARNING", f"Controller resync warning: {e}")

        # 7) 장치 콤보/리스트 갱신
        try:
            self._refresh_device_list()
        except Exception:
            pass

        # 8) (선택) 실행 전 Live-View 복구
        #   - 실패/에러인 경우엔 복구하지 않고, 사용자가 명시적으로 켜게 두는 편이 안전합니다.
        try:
            if getattr(self, "_live_view_before_run", False) and status in ("Completed", "Stopped"):
                if self.controller and self.controller.is_connected():
                    self._append_log("INFO", "Restoring live view session...")
                    self._start_live_view([self.controller])
        except Exception:
            pass

        # 9) 버튼/메뉴 최종 재계산 (즉시 + 약간 뒤 1회 더)
        try:
            self._update_ui_states()  # 즉시
        except Exception:
            pass
        QTimer.singleShot(200, self._update_ui_states)

    # ------------------------------------------------------------------
    #  _run_sequence ― Run Sequence 버튼 핸들러 (최종 교체본)
    # ------------------------------------------------------------------
    def _run_sequence(self) -> None:
        """
        시퀀스를 실행하고 진행 상황을 UI에 연결한다.
        • 멀티 카메라는 MultiRunnerManager, 단일 카메라는 SequenceRunner 사용
        • 실행 직후 Stop 버튼을 켠 뒤 0.2 s 후 UI 상태를 재동기화
        • 단일 러너일 때 에러 이미지 시그널도 UI에 연결
        """
        # 이미 실행 중이면 무시
        if (self.runner_mgr and self.runner_mgr.is_running()) or \
                (self.sequence_runner and self.sequence_runner.is_running()):
            return

        # 1) 시퀀스/카메라 검증
        seq = self.sequence_editor.get_sequence()
        if not seq.steps:
            QMessageBox.warning(self, "Empty Sequence", "There are no steps to execute.")
            return

        ctrls = [controller_pool.get_controller(cid) for cid in controller_pool.list_ids()]
        ctrls = [c for c in ctrls if c and c.is_connected()]
        if not ctrls:
            QMessageBox.warning(self, "No Cameras", "Connect at least one camera.")
            return

        multi_cam = len(ctrls) > 1

        # 2) Live-View 일시 정지 + UI 초기화
        self._live_view_before_run = self.live_stack.currentIndex() == 1
        self._stop_live_view()
        self.log_view.clear()
        self.results_table.setRowCount(0)
        self.progress.setValue(0)
        self.progress.show()

        # 3) 러너 생성 · 시작
        if multi_cam:
            # ― MultiRunnerManager ― (기존 연결 유지)
            self.runner_mgr = MultiRunnerManager(self)
            mgr = self.runner_mgr

            mgr.log_message.connect(self._append_log)
            mgr.progress_update.connect(
                lambda cid, cur, tot: (
                    self.progress.setMaximum(tot),
                    self.progress.setValue(cur),
                    self.lbl_loop.setText(f"{cid}: {cur}/{tot}")
                )
            )
            mgr.sequence_finished.connect(self._on_any_runner_finished)
            mgr.step_started.connect(lambda _cid, idx, name: self._on_step_started(idx, name))
            mgr.step_result.connect(lambda _cid, idx, name, res: self._on_step_result(idx, name, res))
            mgr.preview_frame.connect(self._on_sequence_preview_frame, Qt.QueuedConnection)

            mgr.start(seq, ctrls)  # ▶ START
            self.live_stack.setCurrentIndex(1)

        else:
            # ― 단일 SequenceRunner ―
            from src.core.sequence_runner import SequenceRunner

            # 실행 파일 기준 memento 경로 계산
            try:
                exe_base = Path(sys.executable).parent if getattr(sys, "frozen", False) else \
                Path(__file__).resolve().parents[2]
            except Exception:
                exe_base = Path(__file__).resolve().parents[2]
            memento_dir = exe_base / "logs" / "memento"

            self.sequence_runner = SequenceRunner(
                seq,
                ctrls[0],
                parent=self,
                memento_dir=memento_dir,  # 실행파일 경로/logs/memento
            )
            sr = self.sequence_runner

            sr.log_message.connect(self._append_log)
            sr.progress_update.connect(
                lambda cur, tot: (
                    self.progress.setMaximum(tot),
                    self.progress.setValue(cur)
                )
            )
            sr.step_started.connect(self._on_step_started)
            sr.step_result.connect(self._on_step_result)
            sr.sequence_finished.connect(self._on_any_runner_finished)

            # Live preview 프레임
            sr.test_frame_grabbed.connect(self._on_sequence_preview_frame, Qt.QueuedConnection)

            # ★ 에러 이미지: 테이블/갤러리 갱신용 (UI 스레드 안전)
            sr.error_image_captured.connect(self._on_error_image_signal, Qt.QueuedConnection)

            sr.start()  # ▶ START
            self.live_stack.setCurrentIndex(1)

        # 4) UI 상태 갱신
        self._seq_running_ui = True
        self._update_ui_states()
        self.act_run.setEnabled(False)
        self.act_stop.setEnabled(True)
        QTimer.singleShot(200, self._update_ui_states)

    def _stop_sequence(self) -> None:
        if self.runner_mgr:
            self.runner_mgr.stop()
        elif self.sequence_runner:
            self.sequence_runner.stop()

        # Live-View는 러너가 완전히 끝나면 _on_any_runner_finished() 에서 다시 켠다
        QTimer.singleShot(0, self._update_ui_states)

    def _append_log(self, level: str, message: str):
        color = {"DEBUG": "gray", "INFO": "black", "WARNING": "orange", "ERROR": "red", "CRITICAL": "purple"}.get(level,
                                                                                                                  "black")
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.log_view.append(f'<font color="{color}">[{level}] {timestamp} - {message}</font>')

    def _on_step_started(self, index: int, name: str):
        if index >= self.results_table.rowCount():
            self.results_table.setRowCount(index + 1)
        self.results_table.setItem(index, 0, QTableWidgetItem(str(index + 1)))
        self.results_table.setItem(index, 1, QTableWidgetItem(name))

    @pyqtSlot(int, str, str, dict)
    def _on_step_result(self, index: int, name: str, action_id: str, result: dict):
        """
        [최종 안정화 버전] SequenceRunner로부터 step 결과를 받아 UI 테이블을 업데이트합니다.
        """
        status = result.get("status", "error").capitalize()
        message = result.get("message", "")
        exec_time = result.get("execution_time_ms", -1)

        status_item = QTableWidgetItem(status)
        if status == "Success":
            status_item.setForeground(QBrush(QColor("#4CAF50")))  # Green
        else:
            status_item.setForeground(QBrush(QColor("#F44336")))  # Red

        if index >= self.results_table.rowCount():
            self.results_table.setRowCount(index + 1)

        self.results_table.setItem(index, 0, QTableWidgetItem(str(index + 1)))
        self.results_table.setItem(index, 1, QTableWidgetItem(name))
        self.results_table.setItem(index, 2, QTableWidgetItem(action_id))
        self.results_table.setItem(index, 3, status_item)
        self.results_table.setItem(index, 4, QTableWidgetItem(f"{exec_time}" if exec_time >= 0 else "-"))
        self.results_table.setItem(index, 5, QTableWidgetItem(message))

    def _on_sequence_finished(self, final_status: str, message: str):
        self.progress.hide()
        self.lbl_loop.setText("Loop: -")
        if final_status == "Completed":
            QMessageBox.information(self, "Sequence Finished", message)
        elif final_status == "Stopped":
            QMessageBox.warning(self, "Sequence Stopped", message)
        else:  # Failed
            QMessageBox.critical(self, "Sequence Failed", message)

    def _export_report(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Report", f"Test_Report_{datetime.now():%Y%m%d_%H%M%S}.txt",
                                              "Text Files (*.txt)")
        if not path: return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(f"Test Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                f.write("Sequence Log:\n")
                f.write(self.log_view.toPlainText())
                f.write("\n\n" + "=" * 50 + "\n\n")
                f.write("Test Results:\n")
                for r in range(self.results_table.rowCount()):
                    row_data = [self.results_table.item(r, c).text() for c in range(self.results_table.columnCount())]
                    f.write("\t".join(row_data) + "\n")
            QMessageBox.information(self, "Export Successful", f"Report saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save report: {e}")

    def _show_camera_settings(self):
        if not self._ctrl_connected():
            QMessageBox.warning(self, "Not Connected", "A camera must be connected to view settings.")
            return
        dialog = CameraSettingsDialog(self.controller, self)
        dialog.exec_()

    def _about(self):
        QMessageBox.about(self, f"About {self.APP_NAME}",
                          f"<b>{self.APP_NAME}</b><br>© 2025 Vieworks Inc.<br>All rights reserved.<br><br>Python: {sys.version.split()[0]} / PyQt5")

    def _update_window_title(self):
        path = self.current_sequence_path
        name = Path(path).name if path else "Untitled"
        self.setWindowTitle(f"{self.APP_NAME} - {name}{'*' if self.is_sequence_modified else ''}")

    # ────────────────────────────────────────────────────────────────
    #  UI 상태 갱신 – (완전 교체 버전)
    # ────────────────────────────────────────────────────────────────
    def _update_ui_states(self) -> None:
        cam_connected = bool(self.controller and self.controller.is_connected())
        cam_grabbing = cam_connected and self.controller.is_grabbing()
        seq_running = self._is_sequence_running()
        multi_cam = len(controller_pool.list_ids()) > 1

        # ── Run / Stop / Save ────────────────────────────────
        self.act_run.setEnabled(cam_connected and not seq_running)
        self.act_stop.setEnabled(seq_running)
        self.act_save.setEnabled(self.is_sequence_modified)

        # ── Camera control ───────────────────────────────────
        self.act_settings.setEnabled(cam_connected and not seq_running)
        self.act_grab_start.setEnabled(cam_connected and not cam_grabbing and not seq_running)
        self.act_grab_stop.setEnabled(cam_grabbing and not seq_running)

        # ── Connection management ────────────────────────────
        self.act_refresh.setEnabled(not seq_running)
        self.act_disconnect_all.setEnabled(bool(controller_pool.list_ids()) and not seq_running)

        # ── Live-view / Mosaic ───────────────────────────────
        self.act_mosaic_toggle.setEnabled(multi_cam and not seq_running)
        self.act_live_all.setEnabled(multi_cam and not seq_running)

        # ── Status-bar appearance ────────────────────────────
        if cam_connected:
            self.lbl_conn.setText(f"<b>Active:</b> {self.controller.cam_id}")
            self.lbl_conn.setStyleSheet("""
                padding: 4px 10px;
                border: 1px solid #2E7D32;   /* 짙은 녹색 테두리 */
                background: #2E7D32;         /* 짙은 녹색 배경 */
                color: #F0F4F8;              /* 밝은 글자 */
                border-radius: 4px;
                font-weight: 600;
                min-width: 180px;
            """)
        else:
            self.lbl_conn.setText(" Camera: Disconnected ")
            self.lbl_conn.setStyleSheet("""
                padding: 4px 10px;
                border: 1px solid #607D8B;
                background: #2D3A42;
                color: #B0BEC5;
                border-radius: 4px;
                min-width: 180px;
            """)



    def _load_settings(self):
        settings = QSettings(self.ORG_NAME, self.APP_NAME)
        geom = settings.value("geometry")
        state = settings.value("windowState")
        if isinstance(geom, QByteArray): self.restoreGeometry(geom)
        if isinstance(state, QByteArray): self.restoreState(state)

    def closeEvent(self, event: QCloseEvent):
        if self.sequence_runner and self.sequence_runner.is_running():
            self.sequence_runner.stop()
            self.sequence_runner.wait(1000)

        if self.is_sequence_modified and not self._ask_to_save():
            event.ignore()
            return

        self._stop_live_view()
        self._disconnect_all_cameras()

        settings = QSettings(self.ORG_NAME, self.APP_NAME)
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        event.accept()

    def _ask_to_save(self) -> bool:
        reply = QMessageBox.question(self, "Save Changes?",
                                     "The current sequence has unsaved changes. Do you want to save them?",
                                     QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)

        if reply == QMessageBox.Save: return self._save_sequence()
        if reply == QMessageBox.Cancel: return False
        return True

    def _new_sequence(self):
        if not self._ask_to_save(): return
        self.current_sequence = Sequence(name="Untitled")
        self.current_sequence_path = None
        self.is_sequence_modified = False
        self.sequence_editor.load_sequence(self.current_sequence)
        self._update_window_title()

    def _open_sequence(self):
        if not self._ask_to_save(): return
        path, _ = QFileDialog.getOpenFileName(self, "Open Sequence", "", "JSON Files (*.json)")
        if not path: return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.current_sequence = Sequence.from_dict(data)
            self.current_sequence_path = path
            self.is_sequence_modified = False
            self.sequence_editor.load_sequence(self.current_sequence)
            self._update_window_title()
        except Exception as e:
            QMessageBox.critical(self, "Open Error", f"Failed to load sequence:\n{e}")

    def _save_sequence(self) -> bool:
        return self._save_sequence_as() if not self.current_sequence_path else self._write_sequence(
            self.current_sequence_path)

    def _save_sequence_as(self) -> bool:
        path, _ = QFileDialog.getSaveFileName(self, "Save Sequence As", "", "JSON Files (*.json)")
        if not path: return False
        if not path.lower().endswith(".json"): path += ".json"
        return self._write_sequence(path)

    def _write_sequence(self, path: str) -> bool:
        try:
            sequence_data = self.sequence_editor.get_sequence().to_dict()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(sequence_data, f, indent=2, ensure_ascii=False, cls=_DataclassJSONEncoder)
            self.current_sequence_path = path
            self.is_sequence_modified = False
            self._update_window_title()
            return True
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save sequence:\n{e}")
            return False