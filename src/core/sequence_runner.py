#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SequenceRunner – QThread-based sequence execution engine.

2025-06-24 refactor
+ numeric / bool literal auto-cast for parameters
+ robust loop-control handling  (loop_continue / loop_exit)
+ loop_progress_update signal emission
+ richer exception logging with traceback
+ grab-worker cleanup race-condition fix

2025-06-25 patch
+ safe early-stop handling (0xC0000409 crash fix)
"""

from __future__ import annotations

import importlib
import logging
import operator
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING, Literal, Union, Tuple
import src  # frozen-app safety (PyInstaller 등)

from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
from contextlib import suppress
# ────────────────────────────────────────────────────────────────
#  TYPE-CHECK 전용 CameraController & SDK lazy-loader
# ────────────────────────────────────────────────────────────────
if TYPE_CHECKING:  # IDE / mypy 용
    from src.core.camera_controller import CameraController
    from src.core.error_image_manager import ErrorImageManager
else:
    CameraController = Any  # 런타임에서는 구체 타입 불필요
    ErrorImageManager = Any
import src.core.actions_impl

def _cc_cls() -> "type[CameraController]":
    if not hasattr(_cc_cls, "_cached"):
        try:
            mod = importlib.import_module("src.core.camera_controller")
            _cc_cls._cached = mod.CameraController  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Camera SDK load failed while resolving CameraController: {exc}"
            ) from exc
    return _cc_cls._cached  # type: ignore[return-value]


# ────────────────────────────────────────────────────────────────
#  SDK 와 무관한 내부 모듈
# ────────────────────────────────────────────────────────────────
from src.core.action_registry import get_action_definition
from src.core.actions_base import ActionResult, ContextKey
from src.core.sequence_types import Sequence, ValidationRule
from src.utils.memento_recorder import MementoRecorder
from src.core.error_image_manager import ErrorImageManager
from src.core.events import ErrorImageCaptured, publish

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────
#  Helper – 문자열 치환 & 타입 캐스팅
# ────────────────────────────────────────────────────────────────
def _resolve_value(value: Any, ctx: Dict[ContextKey, Any]) -> Any:
    """'{ctx_key}' 패턴을 context 값으로 치환."""
    if isinstance(value, str):
        try:
            return value.format_map(ctx)
        except KeyError as exc:
            logger.warning("Context key %s not found while resolving %s", exc, value)
        except Exception as exc:
            logger.error("Error resolving %s: %s", value, exc)
    return value


def _auto_cast(val: Any) -> Any:
    """
    문자열인 경우 int / float / bool 리터럴을 자동 판별해
    원본 타입으로 캐스팅한다.
    """
    if not isinstance(val, str):
        return val

    txt = val.strip()
    if txt.lower() in {"true", "false"}:
        return txt.lower() == "true"
    if txt.isdigit():
        return int(txt)
    try:
        # float() 는 int 문자열도 변환하므로 isdigit 먼저 확인
        return float(txt)
    except ValueError:
        return val  # 그대로 유지


def _cast_nested(obj: Any) -> Any:
    """dict / list 컨테이너에서도 재귀적으로 _auto_cast 적용."""
    if isinstance(obj, dict):
        return {k: _cast_nested(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cast_nested(v) for v in obj]
    return _auto_cast(obj)

def _sr__coerce_number(x: Any) -> Any:
    """
    숫자 후보(문자열 포함)를 float/int로 변환. 실패 시 원본 반환.
    """
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, str):
        s = x.strip().replace("_", "")
        # int-like?
        try:
            if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                return int(s)
        except Exception:
            pass
        # float-like?
        try:
            return float(s)
        except Exception:
            return x
    return x

def _sr__safe_compare(a: Any, op: str, b: Any) -> bool:
    """
    • 두 피연산자를 가능한 숫자로 변환하여 (<, <=, >, >=) 비교 시 TypeError 방지
    • 변환 실패 또는 타입 충돌 시 문자열 비교로 폴백
    """
    ops = {
        "==": lambda p, q: p == q,
        "!=": lambda p, q: p != q,
        "<":  lambda p, q: p <  q,
        "<=": lambda p, q: p <= q,
        ">":  lambda p, q: p >  q,
        ">=": lambda p, q: p >= q,
        "exists":       lambda k, _v: bool(k is not None),
        "not_exists":   lambda k, _v: bool(k is None),
        "contains":     lambda p, q: (q in p) if hasattr(p, "__contains__") else False,
        "not_contains": lambda p, q: (q not in p) if hasattr(p, "__contains__") else True,
        "is_true":      lambda p, _q: bool(p) is True,
        "is_false":     lambda p, _q: bool(p) is False,
        "is_none":      lambda p, _q: p is None,
        "is_not_none":  lambda p, _q: p is not None,
    }
    fn = ops.get(op)
    if fn is None:
        raise ValueError(f"Unsupported operator '{op}'")

    # 존재성/불리언/포함계열은 그대로 처리
    if op in ("exists", "not_exists", "contains", "not_contains", "is_true", "is_false", "is_none", "is_not_none"):
        return fn(a, b)

    # 순서비교/동등비교는 숫자 우선 → 실패시 문자열
    a2 = _sr__coerce_number(a)
    b2 = _sr__coerce_number(b)
    try:
        return fn(a2, b2)
    except TypeError:
        return fn(str(a2), str(b2))

def _sr__evaluate_validation(rule, action_result: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, Any, Any, str]:
    """
    단일 ValidationRule 평가.
    반환: (ok, left_value, right_value, message)

    rule.source ∈ {'action_result', 'context'}
    rule.key    : 조회할 키
    rule.operator : '==','>','contains', ... (sequence_types.ValidationOperator)
    rule.target_value : 리터럴 또는 {ctx_key} 치환 문자열

    • 숫자/문자 혼합 비교에서도 안전하게 동작 (_sr__safe_compare 사용)
    • {xxx} 형태는 컨텍스트 치환
    """
    # 1) 좌항(left) 값 확보
    if rule.source == 'action_result':
        left = action_result.get(rule.key)
    elif rule.source == 'context':
        left = context.get(rule.key)
    else:
        return False, None, None, f"Unknown source '{rule.source}'"

    # 2) 우항(right) 값 준비 (+ {ctx_key} 치환)
    rv = rule.target_value
    if isinstance(rv, str) and rv.startswith("{") and rv.endswith("}"):
        ctxk = rv[1:-1]
        right = context.get(ctxk)
    else:
        right = rv

    # 3) 비교
    try:
        ok = _sr__safe_compare(left, rule.operator, right)
    except Exception as e:
        return False, left, right, f"Validation compare error: {e}"

    if not ok:
        # 실패 메시지
        msg = rule.fail_message or f"Validation '{rule.name}' failed: {left!r} {rule.operator} {right!r} (key={rule.key}, src={rule.source})"
        return False, left, right, msg

    return True, left, right, f"Validation '{rule.name}' passed."


class SequenceRunner(QThread):
    """
    백그라운드에서 Sequence 를 단계별로 실행하며
    다양한 Qt 시그널로 UI 에 진행 상황을 알린다.
    """

    # Qt signals ------------------------------------------------------------
    log_message = pyqtSignal(str, str)
    step_started = pyqtSignal(int, str)
    step_result = pyqtSignal(int, str, str, dict)
    """step_result(step_index, step_name, action_id, result_dict)"""
    validation_result = pyqtSignal(int, str, bool, str)
    progress_update = pyqtSignal(int, int)
    loop_progress_update = pyqtSignal(int, int)  # (current_loop, total_loop)
    sequence_finished = pyqtSignal(str, str)
    test_frame_grabbed = pyqtSignal(str, np.ndarray)
    error_image_captured = pyqtSignal(str, dict)

    # --------------------------------------------------------------------- #
    #                               __init__
    # --------------------------------------------------------------------- #
    def __init__(
            self,
            sequence: Sequence,
            controller: "CameraController | None",
            initial_context: Optional[Dict[str, Any]] = None,
            *,
            error_image_manager: "ErrorImageManager | None" = None,
            memento_dir: Optional[Path] = None,
            parent=None,
    ) -> None:
        super().__init__(parent)

        if controller is None:
            raise ValueError("SequenceRunner requires a valid CameraController instance, but got None.")

        self.sequence: Sequence = sequence
        self.controller: "CameraController" = controller
        self.initial_context: Dict[str, Any] = initial_context or {}
        self._finished_signalled = False
        self._error_mgr: "ErrorImageManager" = (
            error_image_manager if error_image_manager is not None else ErrorImageManager()
        )

        # ── 안전한 memento 출력 폴더 보정 ─────────────────────────
        def _safe_dir(p: Optional[Path]) -> Path:
            import os, sys, tempfile
            from pathlib import Path as _P

            def _writable(d: _P) -> bool:
                try:
                    d.mkdir(parents=True, exist_ok=True)
                    probe = d / ".probe_write"
                    with open(probe, "w", encoding="utf-8") as f:
                        f.write("ok")
                    probe.unlink(missing_ok=True)
                    return True
                except Exception:
                    return False

            # 0) 명시 인자 우선
            if p and _writable(p):
                return p

            # 1) 실행 파일 경로/logs/memento
            try:
                exe_base = _P(sys.executable).parent if getattr(sys, "frozen", False) else \
                _P(__file__).resolve().parents[2]
            except Exception:
                exe_base = _P(__file__).resolve().parents[2]
            cand = exe_base / "logs" / "memento"
            if _writable(cand):
                return cand

            # 2) 사용자 홈 ~/memento
            home = _P(os.getenv("USERPROFILE", "")) if os.name == "nt" else _P.home()
            fb = (home / "memento").expanduser()
            if _writable(fb):
                return fb

            # 3) 시스템 temp/memento
            fb = _P(tempfile.gettempdir()) / "memento"
            fb.mkdir(parents=True, exist_ok=True)
            return fb

        self._memento_dir = _safe_dir(Path(memento_dir) if memento_dir else None)

        # 런타임 상태
        self.context: Dict[ContextKey, Any] = {}
        self._stop_requested: bool = False
        self._is_running: bool = False
        self.current_step_index: int = -1

        # 루프 시작점 정보
        self._loop_start_map: Dict[str, int] = {}

    # ---------------------------------------------------------------- log --
    def _emit_log(self, level: str, msg: str, *, step: int | None = None) -> None:
        """Python logger + Qt 시그널 동시 발행 (thread-safe)."""
        tag = f"[cam:{self.context.get('cam_id', '-')}]"
        if step is not None:
            tag += f"[step:{step + 1}]"
        text = f"{tag} {msg}"

        getattr(logger, level.lower(), logger.info)(text)
        try:
            self.log_message.emit(level.upper(), text)
        except RuntimeError:
            # 수신 측이 이미 파괴된 경우
            pass

    # --------------------------------------------------------------- param --
    @staticmethod
    def _resolve_parameters(raw: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        """문자열 치환 → 리터럴 캐스팅."""
        resolved = {k: _resolve_value(v, ctx) for k, v in raw.items()}
        return _cast_nested(resolved)

    # src/core/sequence_runner.py 파일의 SequenceRunner 클래스 내에 위치
    def _sleep_with_abort(self, ms: int) -> bool:
        """
        ms 만큼 대기하되, 중간에 stop 요청이 들어오면 즉시 False 를 반환한다.
        QThread.msleep() 를 짧게 반복해 이벤트 루프가 굳지 않도록 한다.
        """
        if ms <= 0:
            return True
        deadline = time.monotonic() + (ms / 1000.0)
        while time.monotonic() < deadline:
            if self._stop_requested:
                return False
            self.msleep(10)  # 10ms 간격
        return True

    def _capture_error_image(
            self,
            exc: Exception,
            *,
            step_index: int,
            step_name: str,
    ) -> None:
        if not self.controller:
            logger.warning("Cannot capture error image: controller is not available.")
            return

        frame_to_save: Optional[np.ndarray] = None
        source_of_frame = "N/A"
        saved_path_str: Optional[str] = None

        # 1) 가장 최근 np 프레임 우선
        try:
            if hasattr(self.controller, 'get_last_np_frame'):
                frame_to_save = self.controller.get_last_np_frame()
                if frame_to_save is not None:
                    source_of_frame = "Last successfully cached np.ndarray"
        except Exception as np_grab_exc:
            logger.debug("Failed to get last cached numpy frame during error handling: %s", np_grab_exc)
            frame_to_save = None

        # 2) 저장
        if frame_to_save is not None:
            try:
                path_obj = self._error_mgr.save_error_image_from_numpy(
                    frame_to_save, self.controller.cam_id
                )
                if path_obj:
                    # ★ 절대경로 보장
                    saved_path_str = str(Path(path_obj).resolve())
                    logger.info("Error image persisted → %s (Source: %s)", saved_path_str, source_of_frame)
            except AttributeError:
                logger.error("ErrorImageManager missing 'save_error_image_from_numpy'.")
            except Exception as save_exc:
                logger.warning("Failed to save error image from numpy array: %s", save_exc, exc_info=True)
        else:
            try:
                if hasattr(self.controller, 'get_last_buffer') and self.controller.get_last_buffer() is not None:
                    source_of_frame = "Raw egrabber.Buffer was available but could not be converted."
            except Exception:
                pass
            logger.warning("Could not capture error image: no valid numpy frame was available. (Source: %s)",
                           source_of_frame)

        # 3) 메타데이터
        meta: Dict[str, Any] = {
            "camera_id": self.controller.cam_id if self.controller else "N/A",
            "step_index": step_index,
            "step_name": step_name,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[:8192],
            "image_path": saved_path_str or "",  # ★ 절대경로 문자열
            "image_source": source_of_frame,
        }

        # 4) Qt 시그널 + 이벤트버스 발행
        try:
            self.error_image_captured.emit(saved_path_str or "", meta)
        except RuntimeError:
            pass

        try:
            path_for_event = Path(saved_path_str).resolve() if saved_path_str else None
            publish(ErrorImageCaptured(path=path_for_event, meta=meta))
        except Exception as bus_exc:
            logger.debug("Event bus publish failed: %s", bus_exc)

    # ------------------------------------------------------- validation -----
    def _run_validation_rule(
            self,
            rule: ValidationRule,
            action_result: ActionResult,
            step_index: int,
    ) -> bool:
        """
        안전 비교 버전:
        - {ctx_key} 치환 지원
        - 숫자/문자 혼합 비교 안전 (_sr__safe_compare)
        - exists / contains / is_true 등 특수 연산자 지원
        """
        rule_name = rule.name or f"{rule.source}['{rule.key}']"

        # 좌항 값 (source 선택)
        if rule.source == "action_result":
            actual = action_result.get(rule.key)
        elif rule.source == "context":
            actual = self.context.get(rule.key)
        else:
            msg = f"Unsupported source '{rule.source}'"
            self._emit_log("ERROR", msg, step=step_index)
            self.validation_result.emit(step_index, rule_name, False, msg)
            return False

        # 우항 값: {ctx_key} → 치환
        tgt_raw = rule.target_value
        if isinstance(tgt_raw, str) and tgt_raw.startswith("{") and tgt_raw.endswith("}"):
            tgt = self.context.get(tgt_raw[1:-1])
        else:
            tgt = tgt_raw

        # 비교 수행
        try:
            passed = _sr__safe_compare(actual, rule.operator, tgt)
        except Exception as exc:
            detail = f"Validation compare error: {exc}"
            self._emit_log("ERROR", detail, step=step_index)
            self.validation_result.emit(step_index, rule_name, False, detail)
            return False

        detail = f"{actual!r} {rule.operator} {tgt!r}"
        self._emit_log("INFO" if passed else "WARNING", f"{rule_name}: {detail}", step=step_index)
        self.validation_result.emit(step_index, rule_name, passed, detail)
        return passed

    # -------------------------------------------------------- loop anchors --
    def _resolve_loop_anchors(self) -> None:
        """
        • 각 loop 스텝의 `loop_start_label` / `loop_end_label` 을
          실행 인덱스(`loop_start_index`) 로 치환한다.
        • parameters 로만 지정된 label 도 자동 반영.
        • 중복·누락 라벨이 있으면 ValueError.
        """
        name_to_idx: dict[str, int] = {}

        for idx, st in enumerate(self.sequence.steps):
            if st.name:
                if st.name in name_to_idx:
                    raise ValueError(f"Duplicate step name “{st.name}”")
                name_to_idx[st.name] = idx

        for st in self.sequence.steps:
            if not st.is_loop:
                continue

            if not getattr(st, "loop_start_label", None):
                lbl = (
                    st.parameters.get("loop_start_label")  # type: ignore[arg-type]
                    if isinstance(getattr(st, "parameters", None), dict)
                    else None
                )
                if lbl:
                    st.loop_start_label = lbl

            if st.loop_start_index is None:
                if not st.loop_start_label:
                    raise ValueError(f"Loop step “{st.name}” lacks loop_start_label")
                if st.loop_start_label not in name_to_idx:
                    raise ValueError(f"Loop label “{st.loop_start_label}” not found")
                st.loop_start_index = name_to_idx[st.loop_start_label]

            if st.loop_end_label and st.loop_end_label not in name_to_idx:
                raise ValueError(f"Loop end label “{st.loop_end_label}” not found")

    def _abort_if_requested(self, step_index: int | None = None) -> bool:
        """
        UI ‘Stop’ 버튼을 누르면 self._stop_requested 가 True 로 세팅된다.
        • True 를 리턴하면 호출 측에서 즉시 루프를 빠져나가야 한다.
        """
        if self._stop_requested:
            try:
                self._emit_log("WARNING", "▶ Abort requested – exiting loop.", step=step_index)
            except RuntimeError:  # 수신자(QT 오브젝트)가 이미 파괴된 경우
                pass
            return True
        return False

    def run(self) -> None:
        """
        Stable SequenceRunner.run (ASCII-only)

        - Cooperative stop checks at step boundaries and delays.
        - Split-sleep for delays so Stop is responsive.
        - Loop control via action result: status in {"loop_continue","loop_exit"}.
        - On failure/error: capture error image (best-effort) and force camera idle.
        - Emit sequence_finished exactly once using _finished_signalled.
        """
        import time
        import traceback
        from contextlib import suppress

        # ---------------- Helpers ----------------
        def _sleep_with_abort(ms: int) -> bool:
            """Use self._sleep_with_abort if present, otherwise 10ms-slice sleep."""
            if hasattr(self, "_sleep_with_abort"):
                try:
                    return bool(self._sleep_with_abort(int(ms)))
                except Exception:
                    pass
            if ms <= 0:
                return True
            deadline = time.monotonic() + (ms / 1000.0)
            while time.monotonic() < deadline:
                if getattr(self, "_stop_requested", False):
                    return False
                # Prefer QThread.msleep if available
                try:
                    self.msleep(10)  # type: ignore[attr-defined]
                    continue
                except Exception:
                    time.sleep(0.01)
            return True

        def _active_controller():
            """Return the active controller for this run."""
            # Prefer controller_pool if available
            try:
                from src.core import controller_pool  # adjust path if needed
                if hasattr(controller_pool, "first_controller"):
                    return controller_pool.first_controller()
            except Exception:
                pass
            return getattr(self, "controller", None)

        def _get_action_def(action_id: str):
            """Fetch action definition from registry (path-flexible)."""
            try:
                from src.core.action_registry import get_action_definition
                return get_action_definition(action_id)
            except Exception:
                try:
                    from action_registry import get_action_definition  # type: ignore
                    return get_action_definition(action_id)
                except Exception:
                    if "get_action_definition" in globals():
                        return globals()["get_action_definition"](action_id)
                    raise

        def _resolve_params(params: dict, ctx: dict) -> dict:
            if hasattr(self, "_resolve_parameters"):
                try:
                    return self._resolve_parameters(params or {}, ctx)
                except Exception:
                    pass
            return params or {}

        def _emit_safe(sig, *args):
            with suppress(Exception):
                sig.emit(*args)

        # ---------------- Init state ----------------
        final_status = "Error"
        final_message = "An unexpected error occurred."

        if not hasattr(self, "_finished_signalled"):
            self._finished_signalled = False
        if not hasattr(self, "_is_running"):
            self._is_running = False
        self._stop_requested = False

        try:
            # Validate sequence
            if not getattr(self, "sequence", None) or not getattr(self.sequence, "steps", None):
                raise RuntimeError("No sequence loaded or sequence is empty.")

            steps = self.sequence.steps
            total_steps = len(steps)

            # Build loop anchors
            self._loop_start_map = {}
            for i, st in enumerate(steps):
                if getattr(st, "action_id", None) == "repeat_block_start":
                    block_id = None
                    try:
                        block_id = st.parameters.get("block_id")
                    except Exception:
                        block_id = None
                    if not block_id:
                        block_id = getattr(st, "id", None) or ("_blk_%d" % i)
                    if i + 1 < total_steps:
                        self._loop_start_map[str(block_id)] = i + 1

            # Optional: label-based anchors
            with suppress(Exception):
                if hasattr(self, "_resolve_loop_anchors"):
                    self._resolve_loop_anchors()

            # Context
            cam = _active_controller()
            cam_id = getattr(cam, "cam_id", None) or (
                        getattr(self, "controller", None) and self.controller.cam_id) or "N/A"
            base_ctx = getattr(self, "initial_context", {}) or {}
            self.context = {
                **base_ctx,
                "sequence_name": getattr(self.sequence, "name", "Unnamed"),
                "cam_id": cam_id,
                "sequence_start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Flags and index
            self._is_running = True
            self.current_step_index = 0

            # Start log/progress
            with suppress(Exception):
                self.progress_update.emit(0, total_steps)
            self._emit_log("INFO", ">> Running sequence '%s' (%d steps)" % (self.sequence.name, total_steps))

            # Optional memento recorder
            _memento_cm = suppress(Exception)
            try:
                from src.utils.memento_recorder import MementoRecorder  # adjust path
                from datetime import datetime
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                _memento_cm = MementoRecorder(getattr(self, "_memento_dir", None),
                                              basename="%s_%s" % (self.sequence.name, ts),
                                              follow=True)  # type: ignore
            except Exception:
                pass

            with _memento_cm:
                # --------------- Main loop ---------------
                while self.current_step_index < total_steps:
                    # Stop check
                    if self._stop_requested:
                        final_status, final_message = "Stopped", "Sequence stopped by user."
                        break

                    idx = self.current_step_index
                    step = steps[idx]
                    step_name = getattr(step, "name", None) or ("Step %d" % (idx + 1))
                    action_id = getattr(step, "action_id", None) or ""

                    # Skip disabled
                    if not getattr(step, "enabled", True):
                        self._emit_log("INFO", "Skipping disabled step %s" % step_name, step=idx)
                        self.current_step_index += 1
                        with suppress(Exception):
                            self.progress_update.emit(self.current_step_index, total_steps)
                        continue

                    # Controller check (allow some actions without camera)
                    active = _active_controller()
                    allowed_nc = {"connect_camera", "reset_all_counters", "wait", "repeat_block_start",
                                  "repeat_block_end"}
                    if not active and action_id not in allowed_nc:
                        final_status = "Failed"
                        final_message = "No active camera controller for step '%s'." % step_name
                        self._emit_log("ERROR", final_message, step=idx)
                        break

                    # Step start
                    with suppress(Exception):
                        self.step_started.emit(idx, step_name)
                        self.progress_update.emit(idx + 1, total_steps)
                    self._emit_log("INFO", "-- %s (Action: %s)" % (step_name, action_id), step=idx)

                    # Pre-delay
                    pre_delay = int(getattr(step, "delay_before_ms", 0) or 0)
                    if pre_delay > 0:
                        if not _sleep_with_abort(pre_delay):
                            final_status, final_message = "Stopped", "Sequence stopped by user during pre-delay."
                            break

                    # Execute action
                    action_result = {"status": "error", "message": "Not executed"}
                    try:
                        action_def = _get_action_def(action_id)
                        if not action_def:
                            raise ValueError("Action '%s' is not registered." % action_id)

                        params = {}
                        # defaults from signature (if provided)
                        with suppress(Exception):
                            for a in getattr(action_def, "arguments", []) or []:
                                if getattr(a, "default_value", None) is not None:
                                    params[a.name] = a.default_value
                        if getattr(step, "parameters", None):
                            params.update(step.parameters)

                        kwargs = _resolve_params(params, self.context)

                        t0 = time.monotonic()
                        action_result = action_def.execute_func(active, self.context, runner=self,
                                                                **kwargs)  # type: ignore[attr-defined]
                        if "status" not in action_result:
                            action_result["status"] = "success"
                        action_result["execution_time_ms"] = int(round((time.monotonic() - t0) * 1000))

                    except Exception as exc:
                        # On exception: capture error image and build rich result
                        with suppress(Exception):
                            self._capture_error_image(exc, step_index=idx, step_name=step_name)
                        tb_full = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                        action_result = {
                            "status": "error",
                            "message": "%s: %s" % (type(exc).__name__, exc),
                            "exception_type": type(exc).__name__,
                            "exception_repr": repr(exc),
                            "traceback": tb_full[-8192:],
                        }

                    # Emit result
                    with suppress(Exception):
                        self.step_result.emit(idx, step_name, action_id, dict(action_result))

                    # Early stop/user abort
                    if self._stop_requested or action_result.get("user_aborted"):
                        final_status, final_message = "Stopped", "Sequence stopped by user."
                        break

                    status_lower = str(action_result.get("status", "error")).lower()

                    # Loop control
                    if status_lower == "loop_continue":
                        jump_index = None
                        blk = action_result.get("block_id")
                        if blk is not None:
                            jump_index = self._loop_start_map.get(str(blk))
                        if jump_index is None and hasattr(step, "loop_start_index"):
                            jump_index = getattr(step, "loop_start_index")
                        if jump_index is None:
                            final_status = "Failed"
                            final_message = "Loop continue requested but loop anchor not found (step %d)." % (idx + 1)
                            self._emit_log("ERROR", final_message, step=idx)
                            break
                        self._emit_log("INFO", "Loop Continue -> jumping to step %d" % jump_index, step=idx)
                        self.current_step_index = int(jump_index)
                        with suppress(Exception):
                            self.progress_update.emit(self.current_step_index, total_steps)
                        continue

                    if status_lower == "loop_exit":
                        self._emit_log("INFO", "Loop Exit requested by action.", step=idx)
                        # fall-through to validations and post-delay

                    # Validations
                    step_passed = status_lower in {"success", "loop_exit", "loop_continue"}
                    validations = getattr(step, "validations", []) or []
                    if step_passed and validations:
                        for rule in validations:
                            try:
                                enabled = getattr(rule, "enabled", True)
                            except Exception:
                                enabled = True
                            if not enabled:
                                continue
                            ok = False
                            try:
                                ok = self._run_validation_rule(rule, action_result, idx)
                            except Exception as ve:
                                self._emit_log("ERROR", "Validation raised error: %r" % ve, step=idx)
                            if not ok:
                                step_passed = False
                                break

                    if not step_passed:
                        with suppress(Exception):
                            self._capture_error_image(
                                RuntimeError(action_result.get("message", "Validation failed")),
                                step_index=idx,
                                step_name=step_name,
                            )
                        final_status = "Failed"
                        final_message = "Failed at step %d ('%s'): %s" % (
                            idx + 1, step_name, action_result.get("message", "Validation failed"))
                        self._emit_log("ERROR", "Step FAILED -> aborting sequence. Reason: %s" % final_message,
                                       step=idx)
                        break

                    # Post-delay
                    post_delay = int(getattr(step, "delay_after_ms", 0) or 0)
                    if post_delay > 0:
                        if not _sleep_with_abort(post_delay):
                            final_status, final_message = "Stopped", "Sequence stopped by user during post-delay."
                            break

                    # Next step
                    self.current_step_index += 1

                else:
                    # while completed naturally
                    if final_status != "Stopped":
                        final_status, final_message = "Completed", "Sequence finished successfully."

        except Exception as e:
            # Critical error
            self._emit_log("ERROR", "Runner crashed: %r" % e)
            final_status, final_message = "Error", "Runtime error: %s" % e
            with suppress(Exception):
                name_for_img = "Runner Main"
                try:
                    if 0 <= getattr(self, "current_step_index", -1) < len(steps):
                        name_for_img = steps[self.current_step_index].name or name_for_img
                except Exception:
                    pass
                self._capture_error_image(e, step_index=getattr(self, "current_step_index", -1), step_name=name_for_img)

        finally:
            # On Failed/Error: ensure camera is idle (idempotent)
            with suppress(Exception):
                if final_status in ("Failed", "Error"):
                    if getattr(self, "controller", None):
                        with suppress(Exception):
                            self.controller.stop_live_view()
                        with suppress(Exception):
                            self.controller.stop_grab_safe(flush=True, revoke=True, wait_ms=500)

            # End flags
            self._is_running = False

            # Emit finish exactly once
            if not getattr(self, "_finished_signalled", False):
                self._finished_signalled = True
                _emit_safe(self.sequence_finished, final_status, final_message)
                self._emit_log("INFO", ">> Sequence %s: %s" % (final_status, final_message))
            else:
                self._emit_log("INFO", ">> Sequence %s: %s (already signalled)" % (final_status, final_message))

    # ------------------------------------------------------------ public --
    # src/core/sequence_runner.py

    def stop(self) -> None:
        """UI Stop: 하드웨어 먼저 유휴화 → 충분히 대기 → 최후에 terminate() + 신호 보장."""
        if not self._is_running:
            return

        # 0) 로그 + stop 플래그
        try:
            self._emit_log("WARNING", "Stop requested by UI. Attempting graceful shutdown...")
        finally:
            self._stop_requested = True

        # 1) 하드웨어를 먼저 유휴화하여(프레임 대기/루프를 풀어줌) 협조적 종료 유도
        try:
            if self.controller:
                from contextlib import suppress
                with suppress(Exception):
                    self.controller.stop_live_view()  # 내부적으로 grab worker 분리
                with suppress(Exception):
                    self.controller.stop_grab_safe(flush=True, revoke=True, wait_ms=500)
        except Exception as e:
            self._emit_log("WARNING", f"Graceful stop helper failed: {e}")

        # 2) 자연 종료 대기 (최대 5초)
        if self.wait(5000):
            return  # run()의 finally가 실행되며 sequence_finished가 정상 발행됨

        # 3) 아직 살아있다면 강제 종료
        self._emit_log("ERROR", "Runner did not stop within 5 seconds. Forcing termination.")
        try:
            self.terminate()
        except Exception as e:
            self._emit_log("ERROR", f"terminate() raised: {e}")

        # 4) OS/드라이버 정리 시간 추가 대기 (최대 2초)
        self.wait(2000)

        # 5) 여전히 run()의 finally가 실행되지 않았다면 → UI 해제를 보장
        if not getattr(self, "_finished_signalled", False):
            # finally가 안 돌았으므로 직접 내려준다
            self._is_running = False
            self._finished_signalled = True
            try:
                self.sequence_finished.emit("Stopped", "Force-stopped by UI (terminate)")
            except RuntimeError:
                pass

    def is_running(self) -> bool:
        return self._is_running

