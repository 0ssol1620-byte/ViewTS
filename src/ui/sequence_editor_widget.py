#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/ui/sequence_editor_widget.py
──────────────────────────────────────────────────────────────────────────────
카메라 테스트 시퀀스를 위한 그래픽 편집기.

핵심:
- 팔레트(QTreeWidget) → 스텝(QListWidget) 드래그앤드랍 지원
- 스텝 내부 드래그 재정렬은 Qt InternalMove 사용
- 동일 위치로 드롭해도 사라지지 않도록 드롭 후 모델 동기화
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, List, Optional, cast, TYPE_CHECKING
from contextlib import suppress

# Qt
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QDropEvent
from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QListWidget,
    QListWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QAbstractItemView,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QComboBox,
    QPushButton,
    QLabel,
    QScrollArea,
    QGroupBox,
    QDialog,
    QDialogButtonBox,
    QMessageBox,
    QFileDialog,
    QSpacerItem,
    QSizePolicy,
)

logger = logging.getLogger(__name__)

# Core-module imports (fail-soft)
try:
    from src.core.sequence_types import (
        Sequence,
        SequenceStep,
        ValidationRule,
        ValidationSource,
        ValidationOperator,
    )
    from src.core.action_registry import list_available_actions, get_action_definition
    from src.core.actions_base import (
        ActionArgument,
        ActionDefinition,
        PARAM_TYPE_INT,
        PARAM_TYPE_FLOAT,
        PARAM_TYPE_STRING,
        PARAM_TYPE_BOOL,
        PARAM_TYPE_CAMERA_PARAM,
        PARAM_TYPE_ENUM,
        PARAM_TYPE_FILE_SAVE,
        PARAM_TYPE_FILE_LOAD,
        PARAM_TYPE_CONTEXT_KEY,
        PARAM_TYPE_CONTEXT_KEY_OUTPUT,
    )
    MODULE_IMPORT_OK = True
except ImportError as exc:  # pragma: no cover
    logging.basicConfig(level=logging.CRITICAL)
    logger.critical("SequenceEditorWidget – core import failed: %s", exc, exc_info=True)
    MODULE_IMPORT_OK = False

    # Minimal stubs to keep the widget instantiable
    Sequence = SequenceStep = ValidationRule = object
    ActionArgument = ActionDefinition = object
    ValidationSource = ValidationOperator = Any
    PARAM_TYPE_INT = "int"
    PARAM_TYPE_FLOAT = "float"
    PARAM_TYPE_STRING = "string"
    PARAM_TYPE_BOOL = "bool"
    PARAM_TYPE_CAMERA_PARAM = "camera_parameter"
    PARAM_TYPE_ENUM = "enum"
    PARAM_TYPE_FILE_SAVE = "file_save"
    PARAM_TYPE_FILE_LOAD = "file_load"
    PARAM_TYPE_CONTEXT_KEY = "context_key"
    PARAM_TYPE_CONTEXT_KEY_OUTPUT = "context_key_output"

if TYPE_CHECKING:
    from src.core.camera_controller import CameraController
else:
    CameraController = object  # Runtime placeholder

try:
    from src.utils.camera_parameters_model import CameraParametersModel
except ImportError:  # 테스트·Doc 빌드용 폴백
    class CameraParametersModel:
        def list_parameter_names(self): return []
        def list_values_for(self, *_): return []


# ─────────────────────────────────────────────────────────────
# StepsListWidget: 외부 드래그도 받아들이는 리스트
# ─────────────────────────────────────────────────────────────
class StepsListWidget(QListWidget):
    """
    QListWidget 확장:
    - dragEnter/dragMove에서 무조건 acceptProposedAction → 외부 위젯 드롭 허용
    - dropEvent는 에디터의 핸들러로 위임(내부 이동일 때는 부모 구현(super) 먼저 호출)
    """
    def __init__(self, editor: 'SequenceEditorWidget'):
        super().__init__(editor)
        self._editor = editor

    def dragEnterEvent(self, e):
        e.acceptProposedAction()

    def dragMoveEvent(self, e):
        e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent):
        self._editor._handle_drop(e)


# ╔═══════════════════════════════════════╗
#   Validation-rule editor dialog
# ╚═══════════════════════════════════════╝
class ValidationRuleDialog(QDialog):
    def __init__(self, rule: Optional[ValidationRule] = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Validation Rule")
        self.setMinimumWidth(420)

        self._valid_ops: List[ValidationOperator] = [
            '==', '!=', '>', '<', '>=', '<=', 'exists', 'not_exists', 'contains',
            'not_contains', 'is_true', 'is_false', 'is_none', 'is_not_none'
        ]
        self._valid_srcs: List[ValidationSource] = ["action_result", "context"]

        self.rule = rule if isinstance(rule, ValidationRule) else ValidationRule()

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        layout.addLayout(form_layout)

        self.ed_name = QLineEdit(self.rule.name or "")
        form_layout.addRow("Rule Name:", self.ed_name)

        self.cmb_src = QComboBox()
        self.cmb_src.addItems(self._valid_srcs)
        self.cmb_src.setCurrentText(self.rule.source)
        form_layout.addRow("Source:", self.cmb_src)

        self.ed_key = QLineEdit(self.rule.key)
        form_layout.addRow("Key:", self.ed_key)

        self.cmb_op = QComboBox()
        self.cmb_op.addItems(self._valid_ops)
        self.cmb_op.setCurrentText(self.rule.operator)
        form_layout.addRow("Operator:", self.cmb_op)

        self.ed_val = QLineEdit(str(self.rule.target_value))
        form_layout.addRow("Target Value:", self.ed_val)

        self.ed_msg = QLineEdit(self.rule.fail_message or "")
        form_layout.addRow("Fail Message:", self.ed_msg)

        self.chk_en = QCheckBox()
        self.chk_en.setChecked(self.rule.enabled)
        form_layout.addRow("Enabled:", self.chk_en)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self) -> None:
        if not self.ed_key.text().strip():
            QMessageBox.warning(self, "Input Error", "Key cannot be empty.")
            return
        self.rule.name = self.ed_name.text().strip() or ""
        self.rule.source = cast(ValidationSource, self.cmb_src.currentText())
        self.rule.key = self.ed_key.text().strip()
        self.rule.operator = cast(ValidationOperator, self.cmb_op.currentText())
        self.rule.target_value = self.ed_val.text()
        self.rule.fail_message = self.ed_msg.text().strip() or None
        self.rule.enabled = self.chk_en.isChecked()
        super().accept()

    def get_rule(self) -> ValidationRule:
        return self.rule


# ╔═══════════════════════════════════════╗
#   Main Editor Widget
# ╚═══════════════════════════════════════╝
class SequenceEditorWidget(QWidget):
    sequence_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._controller: Optional[CameraController] = None
        self._sequence: Sequence = Sequence(name="Untitled", steps=[])
        self._cur_step: Optional[SequenceStep] = None
        self._param_widgets: Dict[str, QWidget] = {}

        self._build_ui()
        self._populate_action_palette()

    # ─────────────────────────────────────────────────────────────
    # UI
    # ─────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root_layout = QHBoxLayout(self)

        # Available Actions
        actions_group = QGroupBox("Available Actions")
        root_layout.addWidget(actions_group, 2)
        actions_layout = QVBoxLayout(actions_group)

        self.tree_actions = QTreeWidget()
        self.tree_actions.setHeaderHidden(True)
        self.tree_actions.setDragEnabled(True)
        self.tree_actions.setDragDropMode(QAbstractItemView.DragOnly)  # ★ 드래그만
        actions_layout.addWidget(self.tree_actions)

        # Sequence Steps
        steps_group = QGroupBox("Sequence Steps")
        root_layout.addWidget(steps_group, 2)
        steps_layout = QVBoxLayout(steps_group)

        self.lst_steps = StepsListWidget(self)  # ★ 커스텀 리스트
        self.lst_steps.setAcceptDrops(True)
        self.lst_steps.setDragEnabled(True)
        self.lst_steps.setDragDropMode(QAbstractItemView.InternalMove)   # 내부 이동은 Qt가
        self.lst_steps.setDefaultDropAction(Qt.MoveAction)
        self.lst_steps.setDragDropOverwriteMode(False)
        self.lst_steps.setDropIndicatorShown(True)
        self.lst_steps.setSelectionMode(QAbstractItemView.SingleSelection)
        self.lst_steps.currentItemChanged.connect(self._on_step_selected)
        steps_layout.addWidget(self.lst_steps)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(QPushButton("➕ Add", clicked=self._add_default_step, toolTip="Append a NOP step"))
        btn_layout.addWidget(QPushButton("➖ Remove", clicked=self._remove_selected_step, toolTip="Delete current step"))
        btn_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        btn_layout.addWidget(QPushButton("⬆", clicked=lambda: self._move_step(-1), toolTip="Move up"))
        btn_layout.addWidget(QPushButton("⬇", clicked=lambda: self._move_step(1), toolTip="Move down"))
        steps_layout.addLayout(btn_layout)

        # Step Properties
        props_group = QGroupBox("Step Properties")
        root_layout.addWidget(props_group, 2)
        props_layout = QVBoxLayout(props_group)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        props_layout.addWidget(scroll_area)

        self._props_widget = QWidget()
        scroll_area.setWidget(self._props_widget)
        self.form_props = QFormLayout(self._props_widget)
        self._clear_props_panel(show_placeholder=True)

    # ─────────────────────────────────────────────────────────────
    # 액션 팔레트 로딩
    # ─────────────────────────────────────────────────────────────
    def _populate_action_palette(self) -> None:
        self.tree_actions.clear()
        categories: Dict[str, QTreeWidgetItem] = {}
        actions_list = list_available_actions()

        for action_id, display_name, category in actions_list:
            if category not in categories:
                cat_item = QTreeWidgetItem(self.tree_actions, [category])
                # 카테고리는 드래그 불가
                cat_item.setFlags(cat_item.flags() & ~Qt.ItemIsDragEnabled)
                categories[category] = cat_item

            parent_item = categories[category]
            action_def = get_action_definition(action_id)
            if not action_def:
                continue

            leaf = QTreeWidgetItem(parent_item, [display_name])
            leaf.setData(0, Qt.UserRole, action_id)
            leaf.setToolTip(0, action_def.description or "")
            # 리프는 드래그 가능
            leaf.setFlags(leaf.flags() | Qt.ItemIsDragEnabled | Qt.ItemIsEnabled | Qt.ItemIsSelectable)

        self.tree_actions.expandAll()
        self.tree_actions.sortByColumn(0, Qt.AscendingOrder)

    # ─────────────────────────────────────────────────────────────
    # 헬퍼
    # ─────────────────────────────────────────────────────────────
    def _item_text(self, step: SequenceStep, index: int) -> str:
        prefix = "▶" if step.enabled else "⏸"
        action_def = get_action_definition(step.action_id)
        display_name = step.name or (action_def.display_name if action_def else step.action_id)
        return f"{prefix} {index + 1}. {display_name}"

    def _refresh_list_items(self) -> None:
        for i in range(self.lst_steps.count()):
            item = self.lst_steps.item(i)
            step = item.data(Qt.UserRole)
            if isinstance(step, SequenceStep):
                item.setText(self._item_text(step, i))

    def _calc_insert_row(self, pos) -> int:
        """dropIndicatorPosition으로 안전하게 삽입 인덱스 계산."""
        item = self.lst_steps.itemAt(pos)
        dip = self.lst_steps.dropIndicatorPosition()
        if item is None or dip == QAbstractItemView.OnViewport:
            return self.lst_steps.count()
        row = self.lst_steps.row(item)
        if dip == QAbstractItemView.BelowItem:
            row += 1
        return row  # OnItem/AboveItem은 해당 인덱스에 삽입

    def _sync_sequence_with_list(self) -> None:
        """리스트 순서를 self._sequence.steps에 반영."""
        new_order: List[SequenceStep] = []
        for i in range(self.lst_steps.count()):
            it = self.lst_steps.item(i)
            st = it.data(Qt.UserRole)
            if isinstance(st, SequenceStep):
                new_order.append(st)
        self._sequence.steps = new_order
        self._refresh_list_items()
        self.sequence_changed.emit()

    # ─────────────────────────────────────────────────────────────
    # CRUD
    # ─────────────────────────────────────────────────────────────
    def _add_default_step(self) -> None:
        step_id = f"step_{int(time.time() * 1000)}_{random.randint(100, 999)}"
        new_step = SequenceStep(id=step_id, name="New No-Op Step", action_id="nop")
        self._insert_step(new_step, self.lst_steps.count())

    def _insert_step(self, step: SequenceStep, position: int) -> None:
        self._sequence.steps.insert(position, step)
        item = QListWidgetItem()
        item.setData(Qt.UserRole, step)
        item.setSizeHint(QSize(-1, 28))
        self.lst_steps.insertItem(position, item)
        self._refresh_list_items()
        self.lst_steps.setCurrentRow(position)
        self.sequence_changed.emit()

    def _remove_selected_step(self) -> None:
        row = self.lst_steps.currentRow()
        if row >= 0:
            self.lst_steps.takeItem(row)
            self._sequence.steps.pop(row)
            self._refresh_list_items()
            self.sequence_changed.emit()

    def _move_step(self, delta: int) -> None:
        row = self.lst_steps.currentRow()
        new_row = row + delta
        if 0 <= row < self.lst_steps.count() and 0 <= new_row < self.lst_steps.count():
            item = self.lst_steps.takeItem(row)
            self.lst_steps.insertItem(new_row, item)
            step = self._sequence.steps.pop(row)
            self._sequence.steps.insert(new_row, step)
            self._refresh_list_items()
            self.lst_steps.setCurrentRow(new_row)
            self.sequence_changed.emit()

    # ─────────────────────────────────────────────────────────────
    # Drag & Drop (핸들러 본체)
    # ─────────────────────────────────────────────────────────────
    def _handle_drop(self, event: QDropEvent) -> None:
        """
        안정형 Drag & Drop:
        - Tree → List : 우리가 새 스텝 삽입
        - List 내부   : 부모 구현(super)로 InternalMove 수행 후 모델 동기화
        """
        src = event.source()

        # ── 팔레트(Tree) → 리스트 : 새 스텝 삽입 ─────────────────
        if isinstance(src, QTreeWidget):
            item = src.currentItem()
            # 카테고리(루트)는 드래그 불가, 리프만 허용
            if not (item and item.parent()):
                event.ignore()
                return

            action_id = item.data(0, Qt.UserRole)
            insert_at = self._calc_insert_row(event.pos())

            step_id = f"step_{int(time.time() * 1000)}_{random.randint(100, 999)}"
            new_step = SequenceStep(id=step_id, name=item.text(0), action_id=action_id)

            # 모델 & 뷰에 삽입
            self._sequence.steps.insert(insert_at, new_step)
            new_item = QListWidgetItem()
            new_item.setData(Qt.UserRole, new_step)
            new_item.setSizeHint(QSize(-1, 28))
            self.lst_steps.insertItem(insert_at, new_item)

            self._refresh_list_items()
            self.lst_steps.setCurrentRow(insert_at)
            self.sequence_changed.emit()

            event.setDropAction(Qt.CopyAction)
            event.acceptProposedAction()
            return

        # ── 리스트 내부 이동 : 부모 구현(super)로 처리 후 동기화 ─────
        if src is self.lst_steps:
            # 부모 구현 호출로 InternalMove 수행
            super(StepsListWidget, self.lst_steps).dropEvent(event)  # super-call
            # 모델 동기화
            self._sync_sequence_with_list()

            event.setDropAction(Qt.MoveAction)
            event.acceptProposedAction()
            return

        event.ignore()

    # ─────────────────────────────────────────────────────────────
    # Properties panel
    # ─────────────────────────────────────────────────────────────
    def _clear_props_panel(self, show_placeholder: bool = True) -> None:
        while self.form_props.rowCount() > 0:
            self.form_props.removeRow(0)
        self._param_widgets.clear()
        if show_placeholder:
            placeholder = QLabel("Select a step to see its properties...", alignment=Qt.AlignCenter)
            placeholder.setStyleSheet("color: gray; font-style: italic;")
            self.form_props.addRow(placeholder)

    def _on_step_selected(self, current_item: Optional[QListWidgetItem], _: Optional[QListWidgetItem]) -> None:
        if not current_item:
            self._clear_props_panel(show_placeholder=True)
            self._cur_step = None
            return

        step = current_item.data(Qt.UserRole)
        if isinstance(step, SequenceStep):
            self._cur_step = step
            self._load_props_for_step(step)
        else:
            self._clear_props_panel(show_placeholder=True)
            self._cur_step = None

    def _load_props_for_step(self, step: SequenceStep) -> None:
        self._clear_props_panel(show_placeholder=False)
        action_def = get_action_definition(step.action_id)

        # 기본 속성
        self.form_props.addRow(QLabel("<b>Basic Properties</b>"))
        le_name = QLineEdit(step.name)
        le_name.editingFinished.connect(lambda: self._set_attr("name", le_name.text().strip()))
        self.form_props.addRow("Name:", le_name)

        chk_enabled = QCheckBox()
        chk_enabled.setChecked(step.enabled)
        chk_enabled.stateChanged.connect(lambda state: self._set_attr("enabled", state == Qt.Checked))
        chk_enabled.stateChanged.connect(self._refresh_list_items)
        self.form_props.addRow("Enabled:", chk_enabled)

        chk_continue = QCheckBox("Continue on Failure")
        chk_continue.setToolTip("If checked, the sequence will proceed even if this step fails.")
        chk_continue.setChecked(step.continue_on_fail)
        chk_continue.stateChanged.connect(lambda state: self._set_attr("continue_on_fail", state == Qt.Checked))
        self.form_props.addRow("Failure Policy:", chk_continue)

        # 파라미터
        self.form_props.addRow(QLabel("<b>Parameters</b>"))
        if not action_def:
            self.form_props.addRow(QLabel(f"<i style='color:red;'>Unknown Action ID: {step.action_id}</i>"))
        elif not action_def.arguments:
            self.form_props.addRow(QLabel("<i>(No parameters for this action)</i>"))
        else:
            for arg in action_def.arguments:
                self._add_param_editor(step, arg)

    def _set_attr(self, field: str, value: Any) -> None:
        if self._cur_step and getattr(self._cur_step, field, None) != value:
            setattr(self._cur_step, field, value)
            if field in ("name", "enabled"):
                self._refresh_list_items()
            self.sequence_changed.emit()

    def _update_param(self, name: str, value: Any) -> None:
        if self._cur_step:
            self._cur_step.parameters[name] = value
            self.sequence_changed.emit()

    def _add_param_editor(self, step: SequenceStep, arg: ActionArgument) -> None:
        model = CameraParametersModel()
        cur_val = step.parameters.get(arg.name, arg.default_value)

        label = QLabel(f"{arg.display_name}:")
        label.setToolTip(arg.description or "")

        # set_parameter.parameter_name
        if step.action_id == "set_parameter" and arg.name == "parameter_name":
            param_combo = QComboBox()
            param_combo.addItems(model.list_parameter_names())
            if cur_val and cur_val in model.list_parameter_names():
                param_combo.setCurrentText(cur_val)
            else:
                param_combo.setEditable(True)
                param_combo.setCurrentText(str(cur_val or ""))
            self._param_widgets[arg.name] = param_combo
            param_combo.currentTextChanged.connect(lambda text, n=arg.name: self._update_param(n, text))
            self.form_props.addRow(label, param_combo)
            return

        # set_parameter.value
        if step.action_id == "set_parameter" and arg.name == "value":
            param_combo: Optional[QComboBox] = self._param_widgets.get("parameter_name")  # type: ignore[assignment]
            value_combo = QComboBox()

            def _refresh_value_options(selected_param: str) -> None:
                options = model.list_values_for(selected_param)
                value_combo.blockSignals(True)
                value_combo.clear()
                if options:
                    value_combo.setEditable(False)
                    value_combo.addItems(options)
                else:
                    value_combo.setEditable(True)
                    value_combo.addItem("")
                saved = str(step.parameters.get("value", ""))
                if saved and (not options or saved in options):
                    value_combo.setCurrentText(saved)
                value_combo.blockSignals(False)

            _refresh_value_options(param_combo.currentText() if param_combo else "")
            if param_combo:
                param_combo.currentTextChanged.connect(_refresh_value_options)

            value_combo.currentTextChanged.connect(lambda text, n=arg.name: self._update_param(n, text))
            self.form_props.addRow(label, value_combo)
            self._param_widgets[arg.name] = value_combo
            return

        # generic editors
        widget_type = str(arg.type).lower()

        if widget_type == PARAM_TYPE_INT:
            widget = QSpinBox()
            widget.setRange(int(arg.min_value or -2 ** 31), int(arg.max_value or 2 ** 31 - 1))
            widget.setValue(int(cur_val or 0))
            widget.valueChanged.connect(lambda v, n=arg.name: self._update_param(n, v))

        elif widget_type == PARAM_TYPE_FLOAT:
            widget = QDoubleSpinBox()
            widget.setRange(float(arg.min_value or -1e12), float(arg.max_value or 1e12))
            widget.setDecimals(3)
            widget.setSingleStep(float(arg.step or 0.01))
            widget.setValue(float(cur_val or 0.0))
            widget.valueChanged.connect(lambda v, n=arg.name: self._update_param(n, v))

        elif widget_type == PARAM_TYPE_BOOL:
            widget = QCheckBox()
            widget.setChecked(bool(cur_val))
            widget.stateChanged.connect(lambda s, n=arg.name: self._update_param(n, s == Qt.Checked))

        elif widget_type == PARAM_TYPE_ENUM:
            widget = QComboBox()
            widget.addItems(arg.options or [])
            if cur_val in (arg.options or []):
                widget.setCurrentText(str(cur_val))
            widget.currentTextChanged.connect(lambda text, n=arg.name: self._update_param(n, text))

        elif widget_type in (PARAM_TYPE_FILE_SAVE, PARAM_TYPE_FILE_LOAD):
            line_edit = QLineEdit(str(cur_val or ""))
            browse_btn = QPushButton("…")
            browse_kind = widget_type

            browse_btn.clicked.connect(
                lambda _,
                       le=line_edit,
                       p_name=arg.name,
                       kind=browse_kind: self._browse_file(le, p_name, kind)
            )
            line_edit.editingFinished.connect(lambda le=line_edit, p_name=arg.name: self._update_param(p_name, le.text()))

            hbox = QHBoxLayout()
            hbox.setContentsMargins(0, 0, 0, 0)
            hbox.addWidget(line_edit)
            hbox.addWidget(browse_btn)

            container = QWidget()
            container.setLayout(hbox)
            widget = container

        else:  # string, context_key, camera_param(비-enum) …
            widget = QLineEdit(str(cur_val or ""))
            widget.editingFinished.connect(lambda le=widget, n=arg.name: self._update_param(n, le.text()))

        self.form_props.addRow(label, widget)
        self._param_widgets[arg.name] = widget

    def _browse_file(self, line_edit: QLineEdit, param_name: str, browse_type: str) -> None:
        if browse_type == PARAM_TYPE_FILE_SAVE:
            path, _ = QFileDialog.getSaveFileName(self, "Save File As", line_edit.text())
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Open File", line_edit.text())
        if path:
            line_edit.setText(path)
            self._update_param(param_name, path)

    # ─────────────────────────────────────────────────────────────
    # Controller / Sequence
    # ─────────────────────────────────────────────────────────────
    def set_controller(self, controller: Optional[CameraController]):
        self._controller = controller
        if controller:
            CameraParametersModel().ingest_from_controller(controller)

    def load_sequence(self, seq: Sequence) -> None:
        if not isinstance(seq, Sequence):
            logger.error("load_sequence expects a Sequence object, but got %s", type(seq))
            return
        self._sequence = seq
        self.lst_steps.clear()
        self._clear_props_panel()

        for step in self._sequence.steps:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, step)
            item.setSizeHint(QSize(-1, 28))
            self.lst_steps.addItem(item)

        self._refresh_list_items()
        if self.lst_steps.count() > 0:
            self.lst_steps.setCurrentRow(0)
        self.sequence_changed.emit()

    def get_sequence(self) -> Sequence:
        return self._sequence
