#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Camera parameter editor dialog (Qt / Euresys camera).

* 실시간 검색-필터
* 타입별 인라인 편집기(QSpinBox / QDoubleSpinBox / QComboBox / QLineEdit)
* 다중-변경 일괄 Apply
* 모든 예외 안전 처리 & 상세 로깅
"""

from __future__ import annotations
import logging
from typing import Optional, Dict, Any, Iterable, List

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox,
    QTreeView, QHBoxLayout, QHeaderView, QAbstractItemView, QStyledItemDelegate,
    QSpinBox, QDoubleSpinBox, QComboBox, QWidget
)
from PyQt5.QtCore import Qt, QTimer, QModelIndex, pyqtSignal, QEvent
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QBrush, QColor

logger = logging.getLogger(__name__)

# ────────── 컬럼/역할 상수 ─────────────────────────────────────────────
COLUMN_NAME, COLUMN_VALUE, COLUMN_ACCESS, COLUMN_UNIT, COLUMN_DESCRIPTION = range(5)

ROLE_FEATURE_NAME     = Qt.UserRole + 1
ROLE_FEATURE_TYPE     = Qt.UserRole + 2
ROLE_FEATURE_WRITABLE = Qt.UserRole + 3
ROLE_FEATURE_OLDVAL   = Qt.UserRole + 4    # str 형태로 저장

# ═══════════════════════════════════════════════════════════════════
#  ParameterDelegate
# ═══════════════════════════════════════════════════════════════════
class ParameterDelegate(QStyledItemDelegate):
    """Value 셀 더블-클릭 시 타입에 맞는 인라인 위젯을 생성한다."""

    def __init__(self, controller, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.controller = controller

    # ------------------------------------------------ createEditor
    def createEditor(self, parent: QWidget, option, index: QModelIndex):  # noqa: N802
        if index.column() != COLUMN_VALUE:
            return None
        try:
            fname = index.data(ROLE_FEATURE_NAME)
            ftype = index.data(ROLE_FEATURE_TYPE)
            writable = bool(index.data(ROLE_FEATURE_WRITABLE))
            if not (fname and writable):
                return None

            meta = self.controller.get_parameter_metadata(fname)
            cur = self.controller.get_param(fname)

            # ▸ Integer
            if ftype == "Integer":
                ed = QSpinBox(parent)
                ed.setRange(int(meta.get("min", -2**31)), int(meta.get("max", 2**31 - 1)))
                ed.setSingleStep(int(meta.get("inc") or 1))
                ed.setValue(int(cur))
                if unit := meta.get("unit"):
                    ed.setSuffix(f" {unit}")
                ed.setStyleSheet("""
                    QSpinBox {
                        background: #455A64;
                        color: #F0F4F8;
                        border: 1px solid #607D8B;
                        border-radius: 4px;
                        padding: 2px;
                    }
                    QSpinBox::up-button, QSpinBox::down-button {
                        background: #607D8B;
                        border: none;
                    }
                """)
                return ed

            # ▸ Float
            if ftype == "Float":
                ed = QDoubleSpinBox(parent)
                ed.setRange(float(meta.get("min", -1.0e18)), float(meta.get("max", 1.0e18)))
                ed.setDecimals(4)
                ed.setSingleStep(float(meta.get("inc") or 0.1))
                ed.setValue(float(cur))
                if unit := meta.get("unit"):
                    ed.setSuffix(f" {unit}")
                ed.setStyleSheet("""
                    QDoubleSpinBox {
                        background: #455A64;
                        color: #F0F4F8;
                        border: 1px solid #607D8B;
                        border-radius: 4px;
                        padding: 2px;
                    }
                    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                        background: #607D8B;
                        border: none;
                    }
                """)
                return ed

            # ▸ Boolean
            if ftype == "Boolean":
                ed = QComboBox(parent)
                ed.addItems(["False", "True"])
                ed.setCurrentIndex(1 if bool(cur) else 0)
                ed.setStyleSheet("""
                    QComboBox {
                        background: #455A64;
                        color: #F0F4F8;
                        border: 1px solid #607D8B;
                        border-radius: 4px;
                        padding: 2px;
                    }
                    QComboBox::drop-down {
                        border: none;
                        background: #607D8B;
                    }
                """)
                return ed

            # ▸ Enumeration
            if ftype == "Enumeration":
                ed = QComboBox(parent)
                enum_entries = meta.get("enum_entries", [])
                ed.addItems(enum_entries)
                if cur in enum_entries:
                    ed.setCurrentText(str(cur))
                ed.setStyleSheet("""
                    QComboBox {
                        background: #455A64;
                        color: #F0F4F8;
                        border: 1px solid #607D8B;
                        border-radius: 4px;
                        padding: 2px;
                    }
                    QComboBox::drop-down {
                        border: none;
                        background: #607D8B;
                    }
                """)
                return ed

            # ▸ Fallback (text)
            ed = QLineEdit(parent)
            ed.setText(str(cur))
            ed.setStyleSheet("""
                QLineEdit {
                    background: #455A64;
                    color: #F0F4F8;
                    border: 1px solid #607D8B;
                    border-radius: 4px;
                    padding: 2px;
                }
            """)
            return ed

        except Exception as exc:  # pragma: no cover
            logger.error("ParameterDelegate.createEditor error: %s", exc, exc_info=True)
            return None

    # ------------------------------------------------ setEditorData
    def setEditorData(self, editor: QWidget, index: QModelIndex) -> None:  # noqa: N802
        try:
            val = str(index.data(Qt.EditRole))
            if isinstance(editor, QSpinBox):
                editor.setValue(int(float(val)))
            elif isinstance(editor, QDoubleSpinBox):
                editor.setValue(float(val))
            elif isinstance(editor, QComboBox):
                i = editor.findText(val)
                if i >= 0:
                    editor.setCurrentIndex(i)
            elif isinstance(editor, QLineEdit):
                editor.setText(val)
        except Exception as exc:
            logger.warning("setEditorData failed: %s", exc, exc_info=True)

    # ------------------------------------------------ setModelData
    def setModelData(self, editor: QWidget, model, index: QModelIndex) -> None:  # noqa: N802
        try:
            if isinstance(editor, (QSpinBox, QDoubleSpinBox)):
                v = editor.value()
            elif isinstance(editor, QComboBox):
                v = editor.currentText()
            elif isinstance(editor, QLineEdit):
                v = editor.text()
            else:
                return
            model.setData(index, str(v), Qt.EditRole)
        except Exception as exc:
            logger.warning("setModelData failed: %s", exc, exc_info=True)

# ═══════════════════════════════════════════════════════════════════
#  CameraSettingsDialog
# ═══════════════════════════════════════════════════════════════════
class CameraSettingsDialog(QDialog):
    """GenICam Feature 트리를 편집하는 모델리스(dialog-window) 창."""

    settings_applied = pyqtSignal()

    # ------------------------------------------------ init
    def __init__(self, controller, parent: Optional[QWidget] = None):
        super().__init__(
            parent,
            flags=Qt.Window
            | Qt.WindowMinMaxButtonsHint
            | Qt.WindowCloseButtonHint,
        )

        # ── Modeless 설정 ────────────
        self.setWindowModality(Qt.NonModal)
        self.setAttribute(Qt.WA_DeleteOnClose, False)  # DeleteOnClose 제거, 수동 관리
        # ────────────────────────────

        self.controller = controller

        self.setWindowTitle("Camera Parameters")
        self.setMinimumSize(900, 650)

        self._model: QStandardItemModel | None = None
        self._param_tree: QTreeView | None = None
        self._feature_item_map: Dict[str, QStandardItem] = {}

        self._build_ui()

        # 연결 상태 검증
        if not self._validate_controller():
            self.reject()
            return

        # 비동기 트리 로드
        QTimer.singleShot(50, self._populate_and_delegate)

    # ------------------------------------------------ public API
    def set_controller(self, controller) -> None:
        """다이얼로그가 열린 상태에서 active-camera가 변경될 때 호출."""
        if controller is self.controller:
            return
        self.controller = controller
        if not self._validate_controller():
            return
        self._populate_tree()
        self._param_tree.setItemDelegateForColumn(
            COLUMN_VALUE, ParameterDelegate(self.controller, self)
        )
        dev_id = getattr(controller, "device_id", "")
        self.setWindowTitle(f"Camera Parameters — {dev_id}")
        logger.info("CameraSettingsDialog switched to %s", dev_id)

    # ──────────────────────────────────────────────────────────── UI
    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)

        # Search bar
        hl = QHBoxLayout()
        search_label = QLabel("Search:")
        search_label.setStyleSheet("""
            QLabel {
                background: #455A64;
                color: #F0F4F8;
                padding: 4px 8px;
                border-radius: 4px 0 0 4px;
            }
        """)
        hl.addWidget(search_label)
        self.ed_search = QLineEdit()
        self.ed_search.setPlaceholderText("Type feature name…")
        self.ed_search.textChanged.connect(self._filter_tree)
        self.ed_search.setStyleSheet("""
            QLineEdit {
                background: #455A64;
                color: #F0F4F8;
                border: 1px solid #607D8B;
                border-left: none;
                border-radius: 0 4px 4px 0;
                padding: 4px;
            }
            QLineEdit::placeholder {
                color: #B0BEC5;
            }
        """)
        hl.addWidget(self.ed_search)
        lay.addLayout(hl)

        # Tree + model
        self._model = QStandardItemModel(self)
        self._model.setHorizontalHeaderLabels(
            ["Name", "Value", "Access", "Unit", "Description"]
        )
        self._param_tree = QTreeView(self)
        self._param_tree.setModel(self._model)
        self._param_tree.setAlternatingRowColors(True)
        self._param_tree.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked
        )

        hdr: QHeaderView = self._param_tree.header()
        hdr.setSectionResizeMode(COLUMN_NAME, QHeaderView.Stretch)
        hdr.setSectionResizeMode(COLUMN_VALUE, QHeaderView.Interactive)
        hdr.setSectionResizeMode(COLUMN_ACCESS, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(COLUMN_UNIT, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(COLUMN_DESCRIPTION, QHeaderView.Stretch)

        self._param_tree.setStyleSheet("""
            QTreeView {
                background: #455A64;
                color: #F0F4F8;
                border: 1px solid #607D8B;
                border-radius: 4px;
            }
            QTreeView::item {
                padding: 6px;
                height: 28px;
                background: #455A64;
                color: #F0F4F8;
            }
            QTreeView::item:selected {
                background: #607D8B;
                color: #F0F4F8;
            }
            QTreeView::branch {
                background: #455A64;
            }
            QHeaderView::section {
                background: #455A64;
                color: #F0F4F8;
                padding: 6px;
                border: none;
                border-bottom: 1px solid #607D8B;
            }
        """)

        lay.addWidget(self._param_tree)

        # Button layout
        button_layout = QHBoxLayout()
        btn_apply = QPushButton("Apply")
        btn_apply.setStyleSheet("""
            QPushButton {
                background: #607D8B;
                color: #F0F4F8;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #B0BEC5;
            }
        """)
        btn_apply.clicked.connect(self._apply_changes)
        button_layout.addWidget(btn_apply)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet("""
            QPushButton {
                background: #607D8B;
                color: #F0F4F8;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #B0BEC5;
            }
        """)
        btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(btn_cancel)

        button_layout.addStretch()
        lay.addLayout(button_layout)

    # ─────────────────────────────────────────────────── controller check
    def _validate_controller(self) -> bool:
        ok = (
            self.controller
            and self.controller.is_connected()
            and getattr(self.controller, "params", None)
        )
        if not ok:
            QMessageBox.critical(
                self,
                "Camera Not Ready",
                "카메라가 연결되지 않았거나 파라미터를 가져올 수 없습니다.",
            )
        return bool(ok)

    # ─────────────────────────────────────────────────── populate tree
    def _populate_and_delegate(self) -> None:
        try:
            self._populate_tree()
            self._param_tree.setItemDelegateForColumn(
                COLUMN_VALUE, ParameterDelegate(self.controller, self)
            )
        except Exception as exc:
            logger.error("populate_and_delegate failed: %s", exc, exc_info=True)
            QMessageBox.critical(self, "Error", f"트리 생성 실패:\n{exc}")
            self.reject()

    def _populate_tree(self) -> None:
        mdl: QStandardItemModel = self._model  # type: ignore
        mdl.removeRows(0, mdl.rowCount())
        self._feature_item_map.clear()

        try:
            fnames: List[str] = self.controller.list_all_features(only_available=True)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"목록 조회 실패:\n{exc}")
            return

        root = mdl.invisibleRootItem()
        for fname in sorted(fnames, key=str.lower):
            try:
                meta = self.controller.get_parameter_metadata(fname)
                val = meta["value"]
                writable = bool(meta.get("is_writeable", False))
                unit = meta.get("unit", "")
                descr = meta.get("description", "")
                acc_str = "RW" if writable else "R"

                it_name = QStandardItem(fname)
                it_val = QStandardItem(str(val))
                it_acc = QStandardItem(acc_str)
                it_unit = QStandardItem(unit)
                it_desc = QStandardItem(descr)

                # role data
                it_val.setData(fname, ROLE_FEATURE_NAME)
                it_val.setData(meta.get("type", "String"), ROLE_FEATURE_TYPE)
                it_val.setData(writable, ROLE_FEATURE_WRITABLE)
                it_val.setData(str(val), ROLE_FEATURE_OLDVAL)
                if not writable:
                    it_val.setForeground(QBrush(QColor("gray")))

                root.appendRow([it_name, it_val, it_acc, it_unit, it_desc])
                self._feature_item_map[fname] = it_val
            except Exception:
                logger.debug("skip feature %s", fname, exc_info=True)

    # ─────────────────────────────────────────────────── filter
    def _filter_tree(self, txt: str) -> None:
        txt = txt.lower()
        mdl = self._model  # type: ignore
        for row in range(mdl.rowCount()):
            name_item = mdl.item(row, COLUMN_NAME)
            match = txt in name_item.text().lower() if name_item else False
            self._param_tree.setRowHidden(
                row, mdl.invisibleRootItem().index(), not match
            )

    # ─────────────────────────────────────────────────── apply
    def _iter_feature_items(self) -> Iterable[QStandardItem]:
        return self._feature_item_map.values()

    def _apply_changes(self) -> None:
        modified, errors = 0, []
        for it in self._iter_feature_items():
            fname = it.data(ROLE_FEATURE_NAME)
            ftype = it.data(ROLE_FEATURE_TYPE)
            writable = bool(it.data(ROLE_FEATURE_WRITABLE))
            old_val = it.data(ROLE_FEATURE_OLDVAL)
            new_val = it.text()

            if not writable or new_val == old_val:
                continue

            try:
                if ftype == "Integer":
                    casted: Any = int(float(new_val))
                elif ftype == "Float":
                    casted = float(new_val)
                elif ftype == "Boolean":
                    casted = new_val.strip().lower() in ("true", "1")
                else:
                    casted = new_val
                self.controller.set_param(fname, casted)
                it.setData(str(casted), ROLE_FEATURE_OLDVAL)
                modified += 1
                logger.info("Applied %s = %s", fname, casted)
            except Exception as exc:
                logger.error("apply %s failed: %s", fname, exc)
                errors.append(f"[{fname}] {exc}")

        # 결과 토스트
        if errors:
            QMessageBox.warning(
                self,
                "일부 실패",
                f"{modified} 개 성공, {len(errors)} 개 실패:\n" + "\n".join(errors[:10]),
            )
        else:
            QMessageBox.information(
                self,
                "완료",
                "모든 변경 사항이 적용되었습니다."
                if modified
                else "변경된 값이 없습니다.",
            )
        if modified:
            self.settings_applied.emit()

    # ─────────────────────────────────────────────────── overrides
    def closeEvent(self, event: QEvent):
        """창 닫기 이벤트 처리."""
        self.reject()
        event.accept()

    def exec_(self) -> int:  # pylint: disable=invalid-name
        """QDialog.exec_() 호출을 **비-모달**로 동작시키도록 오버라이드."""
        logger.debug("CameraSettingsDialog.exec_ called → forcing modeless show()")
        self.show()
        return 0  # Non-modal 동작

    def accept(self):  # noqa: D401
        self._apply_changes()
        self.close()

    def reject(self):  # noqa: D401
        self.close()