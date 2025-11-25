#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/core/actions_impl.py

모든 execute_* 함수 구현 및 FrameLossMonitor 정의.
actions_base.py 의 타입, 유틸리티, 그리고 register_action, ActionArgument 을 사용합니다.
"""
import math
from egrabber import query
import logging
import time
import operator
import copy
from typing import Any, Dict, Optional, Iterable, List, Union, Protocol, Sequence, Tuple, Callable
from types import SimpleNamespace
from datetime import datetime
from pathlib import Path
from contextlib import suppress
import inspect
# actions_base 에 정의된 공용 타입/유틸 및 데코레이터
from src.core.actions_base import (
    ContextKey,
    ActionResult,
    register_action,
    ActionArgument,
    PARAM_TYPE_INT,
    PARAM_TYPE_FLOAT,
    PARAM_TYPE_STRING,
    PARAM_TYPE_BOOL,
    PARAM_TYPE_CAMERA_PARAM,
    PARAM_TYPE_ENUM,
    PARAM_TYPE_FILE_SAVE,
    PARAM_TYPE_CONTEXT_KEY,
    PARAM_TYPE_CONTEXT_KEY_OUTPUT,
    _resolve_context_vars,
    _convert_value,
    _ar_success,
    _ar_fail,
    NUMPY_AVAILABLE,
    StepActionResult,
)
from egrabber.generated.errors import GenTLException
from src.core.controller_pool import get_controller
from src.core import controller_pool
from uuid import uuid4
from src.utils.sandbox import safe_eval
from src.core.error_image_manager import ErrorImageManager
# NumPy
try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

# SSIM
try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    ssim = None  # type: ignore
    SKIMAGE_AVAILABLE = False

logger = logging.getLogger(__name__)
if logger.level > logging.DEBUG:
    logger.setLevel(logging.DEBUG)

_root = logging.getLogger()               # 루트 로거
if _root.level > logging.DEBUG:
    _root.setLevel(logging.DEBUG)

for _h in _root.handlers:                 # 이미 붙어있는 File/Stream 핸들러까지
    if _h.level > logging.DEBUG:          # (보통 INFO)
        _h.setLevel(logging.DEBUG)
try:
    from src.core.camera_controller import CameraController
except ImportError:
    logger.warning("CameraController not found; a comprehensive dummy class will be used.")


    # --- Dummy EGrabber-related classes for static analysis ---
    class _DummyBuffer:
        def get_numpy_array(self) -> np.ndarray:
            if np:
                return np.zeros((10, 10), dtype=np.uint8)
            raise ImportError("NumPy not available for dummy buffer.")

        def queue(self): pass


    class _DummyEGrabber:
        def get_buffer(self, timeout: int = 1000) -> _DummyBuffer:
            return _DummyBuffer()

        def realloc_buffers(self, *args, **kwargs): pass

        def flush_buffers(self): pass

        def stop(self): pass

        def start(self): pass

        def is_grabbing(self) -> bool: return False


    class CameraController:
        """
        Dummy CameraController for static analysis. Its interface mirrors the real one.
        """

        def __init__(self, *args, **kwargs):
            self.cam_id: str = "dummy_cam_id"
            self.logger = logging.getLogger("dummy_controller")
            self._live_view: bool = False
            self.grabber: Optional[_DummyEGrabber] = _DummyEGrabber()
            self.params: Optional[Dict] = {}  # A simple dict to mock nodemaps

        def is_connected(self) -> bool: return False

        def is_grabbing(self) -> bool: return getattr(self.grabber, 'is_grabbing', lambda: False)()

        def connect_camera(self) -> None: raise ImportError("Dummy method")

        def disconnect_camera(self) -> None: pass

        def start_grab(self, *args, **kwargs) -> None: raise ImportError("Dummy method")

        def stop_grab(self, *args, **kwargs) -> None: pass

        def get_next_frame(self, *args, **kwargs) -> Optional[np.ndarray]: raise ImportError("Dummy method")

        def set_param(self, name: str, value: Any, *args, **kwargs) -> None: raise ImportError("Dummy method")

        def get_param(self, name: str, *args, **kwargs) -> Any: raise ImportError("Dummy method")

        def execute_command(self, name: str, *args, **kwargs) -> None: raise ImportError("Dummy method")

        def get_parameter_metadata(self, name: str) -> Dict[str, Any]: raise ImportError("Dummy method")

        def get_grabber(self) -> Optional[_DummyEGrabber]: return self.grabber

        # --- [수정] 모든 누락된 메서드 및 속성 추가 ---
        def execute_software_trigger(self, *args, **kwargs) -> None: raise ImportError("Dummy method")

        def get_last_buffer(self) -> Any: return None

        def set_last_buffer(self, buf: Any) -> None: pass

        def get_last_np_frame(self) -> Optional[np.ndarray]: return None

        def enable_internal_grabber_trigger(self, *args, **kwargs) -> bool: return False

        def setup_for_hardware_trigger(self, *args, **kwargs) -> None: pass

        def run_synchronized_trigger_test(self, *args, **kwargs) -> None: pass

        def trigger_software_safe(self, *args, **kwargs) -> None: pass

        def execute_device_command(self, *args, **kwargs) -> None: pass

        def set_device_param(self, *args, **kwargs) -> None: pass

        def get_device_param(self, *args, **kwargs) -> Any: pass

        def flush_buffers(self, *args, **kwargs) -> None: pass

        def wait_until_idle(self, *args, **kwargs) -> None: pass

        # [핵심 수정] configure_trigger, stop_grab_safe 추가
        def configure_trigger(self, *args, **kwargs) -> None: pass

        def stop_grab_safe(self, *args, **kwargs) -> None: self.stop_grab()

try:
    from src.core.camera_exceptions import (
        CameraError,
        ParameterError, ParameterSetError, ParameterGetError,
        CommandExecutionError,
        GrabberError, GrabberStartError, GrabberNotActiveError,
        FrameTimeoutError, FrameAcquisitionError,
        CameraConnectionError, CameraNotConnectedError,
        GrabberStopError
    )
except ImportError:
    class CameraError(Exception): pass
    class ParameterError(CameraError): pass
    class ParameterSetError(ParameterError): pass
    class ParameterGetError(ParameterError): pass
    class CommandExecutionError(CameraError): pass
    class GrabberError(CameraError): pass
    class GrabberStartError(GrabberError): pass
    class GrabberNotActiveError(GrabberError): pass
    class FrameTimeoutError(GrabberError): pass
    class FrameAcquisitionError(GrabberError): pass
    class CameraConnectionError(CameraError): pass
    class CameraNotConnectedError(CameraError): pass
    class GrabberStopError(GrabberError): pass

# ── NEW: GenICam-safe setter ────────────────────────────────
try:
    import GenApi                       # Euresys GenApi (present in eGrabber)
except ImportError:
    GenApi = None                       # 타입 체크만에 사용

class NodeManager:
    """
    Wraps a *RemoteModule* → NodeMap → Node so you can call
        NodeManager(remote).set_value("Gain", 2.0)
    without worrying about SDK 버전(getRemoteNodeMap / get_node_map).
    """
    def __init__(self, remote):
        nm_fn = (getattr(remote, "getRemoteNodeMap", None) or
                 getattr(remote, "get_node_map", None))
        if nm_fn is None:
            raise AttributeError("RemoteModule lacks NodeMap accessor")
        self.nodemap = nm_fn()          # INodeMap

    def _node(self, name):
        node = self.nodemap.getNode(name)
        if node is None:
            raise AttributeError(f"Node '{name}' not found")
        if not node.isWritable():
            raise AttributeError(f"Node '{name}' not writable")
        return node

    def set_value(self, name, value):
        node = self._node(name)
        # Enum ↔ fromString / 기타 ↔ setValue
        if GenApi and node.getPrincipalInterfaceType() == GenApi.intfIEnumeration:
            node.fromString(str(value))
        else:
            node.setValue(value)


class _FeatureProxy:                   # <— 추가
    """features()/set()/execute() 만 있는 'flat' 노드맵을
    GenApi Node 처럼 다룰 수 있게 하는 최소 래퍼."""
    def __init__(self, nm, name):
        self._nm, self._name = nm, name
    def isValid(self):     return True
    def isWritable(self):  return self._name in self._nm.features()
    def setValue(self, v): self._nm.set(self._name, v)
    def getInterfaceType(self): return "IEnumeration"
    def getEnumEntries(self):    # 온전한 Enum API 는 없으므로 best-effort
        return []

from src.utils.image_utils import compare_frames_advanced
from src.utils.image_utils import save_frame, calculate_stats
from src.utils.image_utils import IMAGE_SAVE_SUPPORTED as IMAGE_UTILS_AVAILABLE

def _live_view_active(ctrl: Optional[CameraController]) -> bool:
    """
    Checks if the controller is in a persistent grab state (live view).
    """
    if not ctrl:
        return False
    # The `_live_view` flag in CameraController is the primary indicator.
    return getattr(ctrl, "_live_view", False)
# --- END OF PATCH ------------------------------------------------------------



def _get_active_controller(
    default_controller: "CameraController",
    context: Dict[ContextKey, Any]
) -> Tuple[Optional["CameraController"], Optional[ActionResult]]:
    """
    컨텍스트에 'current_cam' 키가 있는지 확인하여 액티브 컨트롤러를 결정합니다.
    (foreach_controller 루프 내에서 올바른 컨트롤러를 선택하기 위함)

    - 'current_cam' 키가 있으면: 해당 ID의 컨트롤러를 풀에서 조회하여 반환합니다.
    - 'current_cam' 키가 없으면: 인자로 받은 default_controller를 반환합니다.
    - 컨트롤러 조회 실패 시: 에러 ActionResult를 함께 반환합니다.

    Returns:
        Tuple[Optional[CameraController], Optional[ActionResult]]:
        (선택된 컨트롤러, 에러 발생 시 결과 메시지)
    """
    target_cam_id = context.get("current_cam")
    if target_cam_id:
        active_controller = get_controller(str(target_cam_id))
        if not active_controller:
            err_msg = f"Foreach target '{target_cam_id}' not found in controller_pool."
            return None, _ar_fail(err_msg)
        return active_controller, None
    return default_controller, None

# ------------- 내부 헬퍼: 컨트롤러 리스트 정규화 ------------------------------
def _as_controller_list(c: Any) -> List["CameraController"]:
    """
    controller 인자로 들어온 *아무것*을 → [CameraController,…] 로 변환.

    • 단일 CameraController  → [it]
    • iterable               → list(element)  (값·value 만 사용)
    • dict                   → list(dict.values())
    • None / 불지원 타입     → []
    """
    from src.core.camera_controller import CameraController   # local import
    if isinstance(c, CameraController):
        return [c]
    if isinstance(c, dict):
        c = c.values()
    if isinstance(c, (list, tuple, set)):
        return [x for x in c if isinstance(x, CameraController)]
    return []

# ─── Node-map 최소 인터페이스 ─────────────────────────────────────────
class _NodeMap(Protocol):
    def features(self) -> Iterable[str]: ...
    def set(self, name: str, value: Any) -> None: ...
    def execute(self, name: str) -> None: ...
    def wait_until_done(self, timeout: float = ...) -> None: ...
# ==================================
# --- Concrete Action Functions ---
# ==================================

# ────────────────────── GenICam node helper ──────────────────────
def _find_feature_node(controller: "CameraController", feature: str):
    """
    camera-remote → grabber-remote 순으로 feature(노드)를 찾아서 반환.
    어떤 노드맵에서도 못 찾으면 None.
    """
    # ① 카메라-remote (표준 GenICam)
    cam_nm = getattr(controller, "params", None)
    if cam_nm and feature in cam_nm.features():
        return cam_nm

    # ② grabber-remote (Euresys RemoteModule)
    grab = getattr(controller, "grabber", None)
    g_remote = getattr(grab, "remote", None)
    if g_remote and feature in g_remote.features():
        return g_remote

    return None

def _iter_nodemaps(ctx):
    """Yield available nodemaps in preferred-search order."""
    # ① camera 노드맵
    cam = getattr(ctx, "camera", None)
    if cam is not None:
        yield getattr(cam, "nodemap", None)

    # ② controller.remote(Device) / grabber.remote 등
    ctrl = getattr(ctx, "controller", None)
    if ctrl is not None:
        yield getattr(ctrl, "remote_nodemap", None)
        yield getattr(ctrl, "nodemap", None)

    # ③ 잡히지 않은 기타 nodemap
    yield getattr(ctx, "nodemap", None)

def _get_feature_node(ctx, name: str, write: bool = True):
    """
    Returns first node that exists **and** is available
    (RW for write=True, RO for write=False). None ↔ not found.
    """
    for nm in _iter_nodemaps(ctx):
        if nm is None:
            continue
        node = nm.get_node(name)
        if node is None or not node.is_available():
            continue
        if write and not node.is_writable():
            continue
        if not write and not node.is_readable():
            continue
        return node
    return None

def wait_until_stream_idle(controller, timeout: float = 0.1) -> None:
    """
    Stop 직후 feature write race를 막기 위해
    DataStream 큐가 비거나 timeout 까지 대기 (grabber.stream 사용)
    """
    s = getattr(controller, "grabber", None)
    s = getattr(s, "stream", None)
    if not s or not hasattr(s, "get_info"):
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if int(s.get_info("QueuedBuf")) == 0:
                return
        except Exception:
            return
        time.sleep(0.005)



def _save_error_image_helper(controller: CameraController, frame_to_save: np.ndarray, reason: str,
                             path_template: str) -> str:
    """오류 이미지를 저장하고, 저장 결과 메시지를 반환하는 공통 함수."""
    if not isinstance(frame_to_save, np.ndarray):
        return " | Invalid frame provided for saving."
    try:
        cam_id = getattr(controller, 'cam_id', 'unknown_cam')
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        final_path_str = path_template.replace("{{timestamp}}", ts_str).replace("{timestamp}", ts_str)
        final_path_str = final_path_str.replace("{{error_type}}", reason).replace("{error_type}", reason)
        final_path_str = final_path_str.replace("{cam_id}", cam_id)

        err_img_mgr = ErrorImageManager()
        saved_path = err_img_mgr.save_error_image_from_numpy(frame_to_save, cam_id, custom_path=final_path_str)

        if saved_path:
            return f" | Error image saved to: {saved_path.name}"
        else:
            return " | Image save was skipped by manager policy."
    except Exception as e_save:
        logger.error(f"EXCEPTION during error image save: {e_save}", exc_info=True)
        return f" | Image save EXCEPTION: {e_save}"

@register_action(
    id="nop",
    display_name="No Operation",
    category="Control",
    description="Does nothing. Useful as a placeholder.",
)
def execute_nop(controller: CameraController, context: Dict[ContextKey, Any], **kwargs) -> ActionResult:
    logger.info("No operation")
    return _ar_success("No operation performed.")

@register_action(
    id="wait",
    display_name="Wait",
    category="Control",
    description="Pauses execution for a duration (ms).",
    arguments=[
        ActionArgument("duration_ms", "Duration (ms)", PARAM_TYPE_INT,
                       "Time to wait in milliseconds.",
                       default_value=1000, min_value=0,
                       max_value=60000, step=100)
    ]
)
def execute_wait(controller: CameraController, context: Dict[ContextKey, Any], **kwargs) -> ActionResult:
    ms = _resolve_context_vars(kwargs.get('duration_ms', 1000), context)
    runner = kwargs.get("runner")
    try:
        ms = int(ms)
        if ms < 0:
            raise ValueError("Negative duration")

        waited = 0
        tick = 50  # ms
        while waited < ms:
            if runner and getattr(runner, "_stop_requested", False):
                return _ar_fail("Stopped by user.", {"user_aborted": True, "waited_ms": waited})
            chunk = min(tick, ms - waited)
            time.sleep(chunk / 1000.0)
            waited += chunk

        return _ar_success(f"Waited {ms} ms.")
    except Exception as e:
        return _ar_fail(f"Wait error: {e}")


@register_action(
    id="log_message",
    display_name="Log Message",
    category="Logging",
    description="Logs a custom message. Supports context vars.",
    arguments=[
        ActionArgument("message", "Message", PARAM_TYPE_STRING,
                       "Message template to log.", default_value="", required=True),
        ActionArgument("level", "Log Level", PARAM_TYPE_ENUM,
                       "Log severity level.", default_value="INFO",
                       options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], required=True)
    ]
)
def execute_log_message(controller: CameraController, context: Dict[ContextKey, Any], **kwargs) -> ActionResult:
    msg = _resolve_context_vars(kwargs.get('message', ''), context)
    level = kwargs.get('level', 'INFO').upper()
    lvl = getattr(logging, level, logging.INFO)
    logger.log(lvl, msg)
    return _ar_success(f"Logged at {level}: {msg}")


# in src/core/actions_impl.py

@register_action(
    id="connect_camera",
    display_name="Connect Camera",
    category="Camera Control",
    description="Connects to the specified camera. If already connected, it ensures a fresh, clean connection.",
)
def execute_connect_camera(controller: CameraController, context: Dict[ContextKey, Any], **kwargs) -> ActionResult:
    """
    [안정화 버전] 지정된 카메라를 (재)연결합니다. reboot 후에 호출될 경우,
    새로운 물리적 장치를 찾아 연결하고 컨트롤러 풀을 업데이트합니다.
    """
    # 이 액션은 특정 카메라 ID를 대상으로 하는 것이 더 명확할 수 있지만,
    # 현재 구조에서는 첫 번째 사용 가능한 카메라를 연결하는 것으로 가정합니다.

    # 더 나은 설계: 'cam_id'를 파라미터로 받도록 수정
    # cam_id_to_connect = kwargs.get('cam_id') or controller_pool.discover_one()

    # 현재 설계 유지: 첫 번째 카메라 연결
    try:
        # controller_pool의 connect_all이 내부적으로 발견/연결/등록을 처리하도록 위임
        # replace_existing=True가 핵심: 기존 연결을 정리하고 새로 만듦
        connected_ids = controller_pool.connect_all(replace_existing=True)

        if not connected_ids:
            return _ar_fail("Connection failed: No cameras found or could not connect.")

        # 첫 번째 연결된 카메라를 활성 컨트롤러로 간주
        new_controller = controller_pool.get_controller(connected_ids[0])
        if new_controller:
            # MainWindow가 참조를 업데이트할 수 있도록 컨텍스트에 정보를 남길 수 있음
            context['active_cam_id'] = new_controller.cam_id
            return _ar_success(f"Successfully connected/reconnected. Active camera is now '{new_controller.cam_id}'.")
        else:
            return _ar_fail("Connected to camera, but failed to retrieve it from the pool.")

    except Exception as e:
        return _ar_fail(f"Failed to connect to camera: {e}", exc_info=True)
@register_action(
    id="disconnect_camera",
    display_name="Disconnect Camera",
    category="Camera Control",
    description="Disconnects from the camera."
)
def execute_disconnect_camera(controller: CameraController, context: Dict[ContextKey, Any], **kwargs) -> ActionResult:
    logger.info("Disconnecting camera...")
    if not controller.is_connected():
        return _ar_success("Already disconnected.")
    try:
        controller.disconnect_camera()
        return _ar_success("Camera disconnected.")
    except Exception as e:
        return _ar_fail(f"Disconnect failed: {e}")

# --- ① generic feature setter ---
def _generic_set_feature(node_map, feat_name: str, tgt_value):
    # 존재/쓰기 가능 1차 확인
    if hasattr(node_map, "features") and callable(node_map.features):
        if feat_name not in node_map.features():
            raise KeyError(f"{feat_name} not found in nodemap")

    node = None
    if hasattr(node_map, "getNode"):
        with suppress(Exception):
            node = node_map.getNode(feat_name)

    # ① 숫자 범위 클램프는 '정수/실수' 노드에서만
    if node:
        try:
            import GenApi
            pit = node.getPrincipalInterfaceType()
            is_numeric = pit in (GenApi.intfIInteger, GenApi.intfIFloat)
        except Exception:
            is_numeric = False
    else:
        is_numeric = False

    if is_numeric and isinstance(tgt_value, (int, float)) and query:
        try:
            # GenApi info 핸들로 안전 조회
            qmin = query.info(feat_name, "Min"); qmax = query.info(feat_name, "Max")
            vmin = node_map.get(qmin, float) if qmin else None
            vmax = node_map.get(qmax, float) if qmax else None
            tv = float(tgt_value)
            if vmin is not None and tv < float(vmin): tv = float(vmin)
            if vmax is not None and tv > float(vmax): tv = float(vmax)
            tgt_value = tv
        except Exception:
            pass  # 조회 실패 시 그냥 set 시도

    # ② 최종 쓰기
    if node and hasattr(node, "isWritable") and node.isWritable():
        # Enum이면 문자열 심볼, 나머지는 setValue가 처리
        try:
            node.setValue(tgt_value)  # GenApi가 타입 맞춰줌
            return
        except Exception as e:
            raise RuntimeError(f"Failed to set {feat_name}: {e}") from e

    # Node API가 없거나 실패 → 평면 set() (마지막 수단)
    if hasattr(node_map, "set"):
        node_map.set(feat_name, tgt_value)
        return

    raise AttributeError("Nodemap has neither 'getNode' nor 'set'")



# ---------------------------------------------------------------------------
#  list_camera_features  – robust, new-signature
# ---------------------------------------------------------------------------
@register_action(
    id="list_camera_features",
    display_name="List Camera Features",
    category="Diagnostics",
    description=("Lists all GenICam features in the camera's Remote Device "
                 "Module.  If the module is missing it returns SUCCESS with "
                 "an empty list so that sequences never abort."),
    arguments=[
        ActionArgument("output_key", "Output Key", PARAM_TYPE_STRING,
                       default_value="camera_features",
                       description="Context key to store feature list.")
    ]
)
def execute_list_camera_features(controller: CameraController,
                                 context: Dict[ContextKey, Any],
                                 *, output_key: str = "camera_features",
                                 **__) -> ActionResult:
    """
    ① remote(=grabber.device) 가 있으면 features() 를 저장
    ② remote 가 없거나 예외 → 빈 리스트를 저장하고 **성공** 으로 리턴
    """
    try:
        remote = getattr(controller.grabber, "device", None)
        feats: list[str] = list(remote.features()) if remote else []
        context[output_key] = feats
        n = len(feats)
        controller.logger.info("[%s] Stored %d feature names into context[%s].",
                               controller.cam_id, n, output_key)
        return _ar_success(f"{n} features captured", {output_key: feats})
    except Exception as exc:
        # 절대 실패로 돌려보내지 않는다
        context[output_key] = []
        controller.logger.warning("[%s] list_camera_features: %s - fallback to empty list.",
                                  controller.cam_id, exc)
        return _ar_success("Feature list unavailable; stored empty list.",
                           {output_key: []})
@register_action(

    id="setup_grabber_cic_for_host_trigger",
    display_name="Setup Grabber CIC for Host Trigger",
    category="Grabber",
    description="[Leader-Only] Configures the grabber so the host can emit LinkTrigger pulses.",
    arguments=[
        ActionArgument("link_trigger_target", "Link Trigger",
                       "Target LinkTrigger to output.",
                       PARAM_TYPE_ENUM,
                       options=["LinkTrigger0", "LinkTrigger1"],
                       default_value="LinkTrigger0"),
        ActionArgument("cycle_trigger_source", "Cycle Trigger Source",
                       "Source node for the cycle trigger.",
                       PARAM_TYPE_STRING,
                       default_value="StartCycle"),
        ActionArgument("cycle_period_us", "Cycle Period (µs)",
                       "Cycle generator period in µ-seconds.",
                       PARAM_TYPE_FLOAT,
                       default_value=3360.0),
    ],
)
def execute_setup_grabber_cic_for_host_trigger(controller: CameraController, context: Dict[ContextKey, Any],
                                               **kwargs) -> ActionResult:
    """
    Configures the grabber's CIC to send a trigger pulse via LinkTrigger on host command.
    Includes diagnostics for firmware and fallback for unsupported devices.
    """
    leader_controller = controller_pool.first_controller()
    if not leader_controller:
        return _ar_fail("No leader controller found in the pool.")

    try:
        target = str(_resolve_context_vars(kwargs.get('link_trigger_target', 'LinkTrigger0'), context))
        cycle_source = str(_resolve_context_vars(kwargs.get('cycle_trigger_source', 'StartCycle'), context))
        period_us = float(_resolve_context_vars(kwargs.get('cycle_period_us', 3360.0), context))

        leader_controller.logger.info(f"[{leader_controller.cam_id}] Configuring Grabber CIC via Device Module...")

        dev = getattr(leader_controller.grabber, "device", None)
        if dev is None:
            return _ar_fail(f"[{leader_controller.cam_id}] Device module unavailable.")

        # Log firmware version for diagnostics
        firmware_version = "Unknown"
        if "DeviceFirmwareVersion" in dev.features():
            try:
                firmware_version = leader_controller.get_device_param("DeviceFirmwareVersion")
                leader_controller.logger.info(f"[{leader_controller.cam_id}] Grabber firmware version: {firmware_version}")
            except Exception as e:
                leader_controller.logger.warning(f"[{leader_controller.cam_id}] Failed to read firmware version: {e}")

        # Available features
        feats = dev.features()
        leader_controller.logger.debug(f"[{leader_controller.cam_id}] Device features: {feats}")

        # Required nodes for modern firmware
        required_nodes = ["DeviceLinkTriggerToolSelector", "DeviceLinkTriggerToolSource", "CycleTriggerSource"]
        missing_nodes = [n for n in required_nodes if n not in feats]

        if missing_nodes:
            leader_controller.logger.warning(f"[{leader_controller.cam_id}] Missing nodes: {missing_nodes}")
            # Check for legacy DLT nodes
            dlt_nodes = {"DeviceLinkTriggerToolSelector", "DeviceLinkTriggerToolSource", "DeviceLinkTriggerToolActivation"}
            if not dlt_nodes.issubset(feats):
                leader_controller.logger.error(
                    f"[{leader_controller.cam_id}] CIC and DLT nodes unavailable. Firmware: {firmware_version}. "
                    f"Consider software triggering or external trigger."
                )
                return _ar_fail(
                    f"[{leader_controller.cam_id}] Grabber does not support CIC or DLT triggering. "
                    f"Missing nodes: {missing_nodes}. Firmware: {firmware_version}."
                )

            # Legacy configuration
            try:
                leader_controller.set_device_param("DeviceLinkTriggerToolSelector", target)
                leader_controller.set_device_param("DeviceLinkTriggerToolSource", "LIN1")
                leader_controller.set_device_param("DeviceLinkTriggerToolActivation", "RisingEdge")
                leader_controller.logger.info(f"[{leader_controller.cam_id}] Configured legacy DLT: {target} -> LIN1")
                return _ar_success(f"[{leader_controller.cam_id}] Grabber configured with legacy DLT for {target}.")
            except Exception as e:
                return _ar_fail(f"[{leader_controller.cam_id}] Failed to configure legacy DLT: {e}", exc_info=True)

        # Modern firmware configuration
        try:
            # Set DeviceLinkTriggerTool
            leader_controller.set_device_param("DeviceLinkTriggerToolSelector", target)
            leader_controller.set_device_param("DeviceLinkTriggerToolSource", "CycleGenerator0")
            leader_controller.set_device_param("DeviceLinkTriggerToolActivation", "RisingEdge")

            # Set CycleTrigger
            leader_controller.set_device_param("CycleTriggerSource", cycle_source)
            if cycle_source.lower() in ("immediate", "startcycle", "cyclegenerator0"):
                if "CycleTriggerPeriod" in feats:
                    leader_controller.set_device_param("CycleTriggerPeriod", period_us)
                else:
                    leader_controller.logger.warning(f"[{leader_controller.cam_id}] CycleTriggerPeriod not available, skipping period setting.")

            # Optional: Set CameraControlMethod
            if "CameraControlMethod" in feats:
                leader_controller.set_device_param("CameraControlMethod", "RC")

            # Optional: Set CxpTriggerMessage
            if "CxpTriggerMessageSelector" in feats:
                leader_controller.set_device_param("CxpTriggerMessageSelector", target)
                leader_controller.set_device_param("CxpTriggerMessageSource", "CycleTrigger")

            # Verify CycleLostTriggerCount
            if "CycleLostTriggerCount" in feats:
                lost_count = leader_controller.get_device_param("CycleLostTriggerCount")
                if int(lost_count) > 0:
                    raise GrabberError(f"[{leader_controller.cam_id}] Non-zero CycleLostTriggerCount: {lost_count}")

            leader_controller.logger.info(f"[{leader_controller.cam_id}] Grabber CIC configured: CycleGenerator0 -> {target} on '{cycle_source}' (Period={period_us} µs).")
            return _ar_success(f"[{leader_controller.cam_id}] Grabber CIC configured for Host Trigger to {target}.")

        except Exception as e:
            return _ar_fail(f"[{leader_controller.cam_id}] Failed to configure CIC: {e}", exc_info=True)

    except Exception as e:
        return _ar_fail(f"[{leader_controller.cam_id}] Unexpected error in CIC setup: {e}", exc_info=True)
# src/core/actions_impl.py
@register_action(
    id="probe_trigger_features",
    display_name="Probe Trigger Features",
    category="Diagnostics",
    description="Lists all GenICam features in the Device Module and highlights potential trigger-related nodes.",
    arguments=[]
)
def execute_probe_trigger_features(controller: CameraController, context: Dict[ContextKey, Any],
                                  **kwargs) -> ActionResult:
    try:
        leader_controller = controller_pool.first_controller()
        if not leader_controller:
            return _ar_fail("No leader controller found in the pool.", exc_info=True)

        dev = getattr(leader_controller.grabber, "device", None)
        if dev is None:
            leader_controller.logger.error(f"[{leader_controller.cam_id}] Device module unavailable. Checking grabber initialization...")
            grabber = leader_controller.get_grabber()
            if grabber is None:
                leader_controller.logger.error(f"[{leader_controller.cam_id}] Grabber instance is None. Available attributes: {dir(leader_controller)}")
                return _ar_fail(f"[{leader_controller.cam_id}] Grabber instance is None.", exc_info=True)
            try:
                leader_controller.logger.info(f"[{leader_controller.cam_id}] Attempting to reinitialize Device module...")
                dev = getattr(grabber, "device", None)
                if dev is None:
                    leader_controller.logger.error(f"[{leader_controller.cam_id}] Failed to initialize Device module after retry. Grabber attributes: {dir(grabber)}")
                    return _ar_fail(f"[{leader_controller.cam_id}] Failed to initialize Device module after retry.", exc_info=True)
            except GenTLException as e:
                leader_controller.logger.error(f"[{leader_controller.cam_id}] GenTLException during Device module reinitialization: {str(e)} (Error Code: {getattr(e, 'gc_err', -1)})", exc_info=True)
                return _ar_fail(f"[{leader_controller.cam_id}] Device module reinitialization error: {e}", exc_info=True)
            except Exception as e:
                leader_controller.logger.error(f"[{leader_controller.cam_id}] Unexpected error during Device module reinitialization: {str(e)}", exc_info=True)
                return _ar_fail(f"[{leader_controller.cam_id}] Device module reinitialization error: {e}", exc_info=True)

        # Log firmware version
        firmware_version = "Unknown"
        if "DeviceFirmwareVersion" in dev.features():
            try:
                firmware_version = leader_controller.get_device_param("DeviceFirmwareVersion")
                leader_controller.logger.info(f"[{leader_controller.cam_id}] Grabber firmware version: {firmware_version}")
            except Exception as e:
                leader_controller.logger.warning(f"[{leader_controller.cam_id}] Failed to read firmware version: {e}", exc_info=True)

        # List all features with error handling
        try:
            features = dev.features()
            leader_controller.logger.debug(f"[{leader_controller.cam_id}] All Device features: {features}")
        except GenTLException as e:
            leader_controller.logger.error(f"[{leader_controller.cam_id}] GenTLException accessing Device features: {str(e)} (Error Code: {getattr(e, 'gc_err', -1)})", exc_info=True)
            return _ar_fail(f"[{leader_controller.cam_id}] Failed to access Device features: {e}", exc_info=True)
        except Exception as e:
            leader_controller.logger.error(f"[{leader_controller.cam_id}] Unexpected error accessing Device features: {str(e)}", exc_info=True)
            return _ar_fail(f"[{leader_controller.cam_id}] Failed to access Device features: {e}", exc_info=True)

        # Highlight trigger-related features
        trigger_keywords = ["Trigger", "Cycle", "Pulse", "Event", "Line", "CxpTrigger", "LinkTrigger"]
        trigger_features = [f for f in features if any(kw.lower() in f.lower() for kw in trigger_keywords)]
        if trigger_features:
            leader_controller.logger.info(f"[{leader_controller.cam_id}] Potential trigger-related features: {trigger_features}")
            for feat in trigger_features:
                try:
                    value = leader_controller.get_device_param(feat)
                    leader_controller.logger.info(f"[{leader_controller.cam_id}] {feat}: {value}")
                except GenTLException as e:
                    leader_controller.logger.warning(f"[{leader_controller.cam_id}] GenTLException reading {feat}: {str(e)} (Error Code: {getattr(e, 'gc_err', -1)})", exc_info=True)
                except Exception as e:
                    leader_controller.logger.warning(f"[{leader_controller.cam_id}] Failed to read {feat}: {str(e)}", exc_info=True)
        else:
            leader_controller.logger.warning(f"[{leader_controller.cam_id}] No trigger-related features found.")

        return _ar_success(f"[{leader_controller.cam_id}] Trigger feature probe completed. Check logs for details.")

    except Exception as e:
        leader_controller.logger.error(f"[{leader_controller.cam_id}] Unexpected error probing trigger features: {str(e)}", exc_info=True)
        return _ar_fail(f"[{leader_controller.cam_id}] Unexpected error probing trigger features: {e}", exc_info=True)

@register_action(
    id="setup_grabber_internal_trigger",
    display_name="Setup Grabber Internal Trigger",
    category="Grabber",
    description="[Leader-Only] Enables the grabber’s own trigger generator.",
    arguments=[ ... ]
)
def execute_setup_grabber_internal_trigger(controller: CameraController,
                                           context: Dict[ContextKey, Any],
                                           **kwargs) -> ActionResult:
    leader = controller_pool.first_controller()
    if not leader:
        return _ar_fail("No leader controller in pool.")

    try:
        link  = str(_resolve_context_vars(kwargs.get("link_trigger_target", "LinkTrigger0"), context))
        per   = float(_resolve_context_vars(kwargs.get("cycle_period_us", 3.36), context))
        act   = str(_resolve_context_vars(kwargs.get("trigger_format", "RisingEdge"), context))

        ok = leader.enable_internal_grabber_trigger(
            link_trigger        = link,
            min_period_us       = per,
            trigger_activation  = act,            # ← 인자명 수정
        )
        if not ok:
            raise RuntimeError("CIC / DLT nodes missing on this firmware.")

        return _ar_success(f"[{leader.cam_id}] Internal trigger enabled → {link} "
                           f"(period {per} µs, {act}).")
    except Exception as e:
        return _ar_fail(f"[{leader.cam_id}] Internal trigger setup failed: {e}", exc_info=True)

# ---------------------------------------------------------------------------
#  NEW  ―  Get-next-frame  (live-safe, stop/start 안 함)
# ---------------------------------------------------------------------------
@register_action(
    id="get_next_frame",
    display_name="Get Next Frame",
    category="Image",
    description="Dequeues a single frame from the active controller without "
                "stopping or restarting acquisition.",
    arguments=[
        ActionArgument("output_context_key", "Output Key",
                       PARAM_TYPE_CONTEXT_KEY_OUTPUT,
                       "Context key for the frame.",
                       default_value="frame", required=True),
        ActionArgument("timeout_ms", "Timeout (ms)", PARAM_TYPE_INT,
                       "Per-frame timeout.",
                       default_value=1500, min_value=100, max_value=10000, step=100)
    ]
)
def execute_get_next_frame(
        controller: CameraController,
        context: Dict[ContextKey, Any],
        *,
        runner: Optional[Any] = None,  # SequenceRunner 인스턴스를 받기 위한 특수 인자
        **kwargs
) -> ActionResult:
    """[최종 수정] 프레임을 획득하여 context에 저장하고, UI에도 표시합니다."""
    active_ctrl, err = _get_active_controller(controller, context)
    if err: return err
    if not active_ctrl: return _ar_fail("No active controller resolved.")

    key = str(_resolve_context_vars(kwargs.get("output_context_key", "frame"), context))
    timeout = int(_resolve_context_vars(kwargs.get("timeout_ms", 1500), context))

    try:
        frame = active_ctrl.get_next_frame(timeout_ms=timeout)
        if not isinstance(frame, np.ndarray):
            return _ar_fail(f"Failed to grab frame (timeout: {timeout}ms).")

        context[key] = frame.copy()

        # ★★★ [수정] execute_grab_frames와 동일한 UI 업데이트 로직 추가 ★★★
        if runner and hasattr(runner, 'test_frame_grabbed'):
            runner.test_frame_grabbed.emit(active_ctrl.cam_id, frame)

        return _ar_success(f"Frame grabbed into ctx['{key}'] and displayed.", {"shape": frame.shape})

    except FrameTimeoutError as e:
        return _ar_fail(f"[{active_ctrl.cam_id}] get_next_frame timeout ({timeout} ms): {e}")
    except Exception as e:
        return _ar_fail(f"[{active_ctrl.cam_id}] get_next_frame error: {e}", exc_info=True)


@register_action(
    id="execute_grabber_cycle_start",
    display_name="Execute Grabber CycleStart",
    category="Grabber",
    description="[Leader-Only] Executes the 'StartCycle' command on the grabber to generate a single trigger pulse.",
)
def execute_execute_grabber_cycle_start(controller: CameraController, context: Dict[ContextKey, Any], **kwargs) -> ActionResult:
    """
    그래버의 'StartCycle' 커맨드를 실행하여 설정된 CIC를 통해 한 번의 트리거 펄스를 발생시킵니다.
    """
    leader_controller = controller_pool.first_controller()
    if not leader_controller:
        return _ar_fail("No leader controller to send trigger.")

    try:
        # CameraController의 그래버 직접 제어 메서드 사용
        leader_controller.execute_device_command("StartCycle", timeout=0.5)
        return _ar_success(f"Grabber 'StartCycle' command executed by '{leader_controller.cam_id}'.")
    except Exception as e:
        return _ar_fail(f"Failed to execute 'StartCycle': {e}", exc_info=True)


def _resolve_active_controller(
    controller: Any,
    context: Dict[ContextKey, Any],
) -> Optional["CameraController"]:
    """
    foreach-loop 지원용 ‘액티브’ CameraController 결정 헬퍼.

    1) controller 인자가 CameraController            → 그 값을 기본값으로 사용
       • list/tuple/set/dict                          → 첫 번째 CameraController 추출
       • None                                        → 기본값 없음
    2) context["current_cam"] 가 지정돼 있으면       → 해당 ID를 풀(pool)에서 우선 조회
    3) 기본값이 결정되면 _get_active_controller() 로 최종 확정
    4) 실패하면 None 반환 (호출부에서 _ar_fail 처리)

    Returns:
        CameraController | None
    """
    # 0. controller 인자를 CameraController 리스트로 정규화
    ctrls = _as_controller_list(controller)
    default_ctrl: Optional["CameraController"] = ctrls[0] if ctrls else None

    # 1. foreach 루프 등에서 넘겨준 current_cam 우선 적용
    if "current_cam" in context:
        from src.core.controller_pool import get_controller
        target = get_controller(str(context["current_cam"]))
        if target is not None:
            default_ctrl = target

    # 2. 기본 컨트롤러조차 없으면 실패
    if default_ctrl is None:
        return None

    # 3. _get_active_controller 로 ActionResult 래퍼 제거
    active, err = _get_active_controller(default_ctrl, context)
    if err is not None:
        return None
    return active


@register_action(
    id="set_parameter",
    display_name="Set Parameter",
    category="Camera Control",
    description="Generic GenICam setter (camera-remote → grabber-remote → grabber-device). "
                "Enumeration pre-check + optional verify.",
    arguments=[
        ActionArgument("parameter_name", "Parameter", PARAM_TYPE_CAMERA_PARAM,
                       "GenICam feature name.", required=True),
        ActionArgument("value", "Value", PARAM_TYPE_STRING,
                       "Value to set (supports context vars).", required=True),
        ActionArgument("verify", "Read-back verify?", PARAM_TYPE_BOOL,
                       "Read back and compare?", default_value=True),
    ],
)
def execute_set_parameter(
        controller,
        context,
        *,
        parameter_name: Optional[str] = None,
        value: Any = None,
        verify: bool = True,
        **__,
):
    from src.utils.camera_parameters_model import CameraParametersModel
    model = CameraParametersModel()

    if not parameter_name:
        return _ar_fail("parameter_name is required.")

    resolved_value = _resolve_context_vars(value, context)
    allowed_vals = model.list_values_for(parameter_name)
    if allowed_vals and str(resolved_value) not in map(str, allowed_vals):
        return _ar_fail(f"Invalid value '{resolved_value}' for '{parameter_name}'. "
                        f"Allowed: {allowed_vals}")

    active_controller = _resolve_active_controller(controller, context)
    if not active_controller:
        return _ar_fail("No active controller resolved for set_parameter.")

    feat_name = str(parameter_name)
    tgt_value = _convert_value(resolved_value)

    # ★ camera → grabber-remote → grabber-DEVICE 순서로 탐색 (Device 추가)
    search_chain = [
        (getattr(active_controller, "params", None), "camera_params"),
        (getattr(active_controller.grabber, "remote", None), "grabber_remote"),
        (getattr(active_controller.grabber, "device", None), "grabber_device"),
    ]

    errors: list[str] = []
    for nm, label in search_chain:
        if nm is None:
            continue
        if hasattr(nm, "features") and callable(nm.features) and feat_name not in nm.features():
            continue

        try:
            _generic_set_feature(nm, feat_name, tgt_value)

            if verify and hasattr(nm, "get"):
                try:
                    read_back = nm.get(feat_name)
                    if isinstance(tgt_value, float):
                        if not math.isclose(float(read_back), float(tgt_value), rel_tol=1e-5):
                            raise RuntimeError(f"verify failed (read {read_back})")
                    elif str(read_back) != str(tgt_value):
                        raise RuntimeError(f"verify failed (read {read_back})")
                except Exception as verify_exc:
                    active_controller.logger.warning("[%s] Verify for %s failed: %s",
                                                     active_controller.cam_id, feat_name, verify_exc)

            active_controller.logger.info("[%s] %s ← %s  via %s",
                                          active_controller.cam_id, feat_name, tgt_value, label)
            return _ar_success(f"Set {feat_name} to {tgt_value} via {label}",
                               {"value_set": tgt_value, "target_nodemap": label})

        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}('{exc}')")

    return _ar_fail(" / ".join(errors) or f"Failed to set parameter '{feat_name}'. Not found or not writable.")


@register_action(
    id="set_trigger_mode",
    display_name="Set TriggerMode",
    category="Camera Control",
    description="Turns TriggerMode On/Off directly.",
    arguments=[ActionArgument("mode", "Mode", PARAM_TYPE_ENUM,
               "On / Off", options=["On", "Off"], default_value="Off")]
)
def execute_set_trigger_mode(controller: CameraController, context: Dict[ContextKey, Any], *, mode="Off", **_):
    """
    Sets the 'TriggerMode' for the currently selected 'TriggerSelector'.
    It finds the TriggerMode node on camera-remote or grabber-remote.
    """
    # 액티브 컨트롤러 가져오기
    active_controller, err = _get_active_controller(controller, context)
    if err:
        return err

    # 파라미터 값 resolve
    target_mode = str(_resolve_context_vars(mode, context))
    if target_mode not in ["On", "Off"]:
        return _ar_fail(f"Invalid mode '{target_mode}' for set_trigger_mode. Must be 'On' or 'Off'.")

    # 카메라/그래버 노드맵에서 TriggerMode 노드 찾기
    node_maps = [
        getattr(active_controller, "params", None),
        getattr(active_controller.grabber, "remote", None)
    ]

    for nm in node_maps:
        if nm and "TriggerMode" in nm.features():
            try:
                # set_param을 사용하여 안전하게 설정
                active_controller.set_param("TriggerMode", target_mode)
                active_controller.logger.info(
                    "[%s] TriggerMode set to '%s'", active_controller.cam_id, target_mode
                )
                return _ar_success(f"[{active_controller.cam_id}] TriggerMode = {target_mode}")
            except Exception as e:
                msg = f"[{active_controller.cam_id}] Failed to set TriggerMode to '{target_mode}': {e}"
                logger.error(msg, exc_info=True)
                return _ar_fail(msg)

    # TriggerMode를 찾지 못한 경우
    msg = f"[{active_controller.cam_id}] TriggerMode feature not found in any nodemap."
    logger.warning(msg)
    # 실패가 아닌 경고로 처리하여, TriggerMode가 없는 카메라에서도 시퀀스가 진행되도록 할 수 있음
    # 하지만 이 시나리오에서는 필수적이므로 실패로 처리하는 것이 더 안전함
    return _ar_fail(msg)
# --- END OF FILE ------------------------------------------------------------
# in src/core/actions_impl.py

@register_action(
    id="camera_reboot",
    display_name="Camera Reboot",
    category="Camera Control",
    description="Executes a device reset and safely removes the controller from the pool.",
)
def execute_camera_reboot(controller: CameraController, context: Dict[ContextKey, Any], **kwargs) -> ActionResult:
    """
    [안정화 버전] 카메라에 재부팅 명령을 보내고, 즉시 컨트롤러 풀에서 해당 객체를 제거하여
    '좀비' 객체가 남는 것을 방지합니다.
    """
    active_controller, err = _get_active_controller(controller, context)
    if err:
        return err
    if not active_controller:
        return _ar_fail("No active controller to reboot.")

    cam_id_to_reboot = active_controller.cam_id
    reboot_commands = ["DeviceReset", "Reboot"]
    reboot_command_sent = False

    for command in reboot_commands:
        try:
            active_controller.execute_command(command, timeout=0.1)  # 짧은 타임아웃
            reboot_command_sent = True
            logger.info(f"[{cam_id_to_reboot}] Reboot command '{command}' sent.")
            break
        except CommandExecutionError:
            continue
        except Exception as e:
            logger.warning(f"[{cam_id_to_reboot}] Error sending reboot command '{command}': {e}")

    if not reboot_command_sent:
        return _ar_fail(
            f"Reboot failed: None of the candidate commands {reboot_commands} were found or writable on '{cam_id_to_reboot}'.")

    # ★★★ [핵심] 명령을 보낸 직후, 컨트롤러 풀에서 즉시 제거합니다. ★★★
    try:
        # disconnect=True 플래그를 사용하여 내부 리소스도 정리하도록 합니다.
        controller_pool.unregister(cam_id_to_reboot, disconnect=True)
        logger.info(f"[{cam_id_to_reboot}] Controller unregistered from pool following reboot.")

        # MainWindow가 참조하는 컨트롤러가 제거된 경우를 대비하여 None으로 설정하도록 유도
        # (이 부분은 MainWindow의 로직과 연계되어야 더 견고해집니다)
        if 'main_controller_ref' in context and context['main_controller_ref'] == active_controller:
            context['main_controller_ref'] = None

    except Exception as e:
        # 이미 풀에 없거나 하는 경우도 있으므로 경고만 기록
        logger.warning(f"[{cam_id_to_reboot}] Could not unregister controller from pool after reboot: {e}")

    return _ar_success(f"Reboot command sent to '{cam_id_to_reboot}' and controller was removed from the pool.")

# --- END REPLACEMENT --------------------------------------------------------

@register_action(
    id="set_context_variable",
    display_name="Set Context Variable",
    category="Context",
    description=(
        "Stores/updates a context variable.\n"
        "• {context.key} replacement\n"
        "• 'eval:' expressions evaluated in a SAFE environment"
    ),
    arguments=[
        ActionArgument("key", "Context Key", PARAM_TYPE_CONTEXT_KEY_OUTPUT, "Name of the variable.", required=True),
        ActionArgument("value", "Value", PARAM_TYPE_STRING, "Literal value or 'eval:' expression.", required=True),
        ActionArgument(
            "value_type",
            "Value Type",
            PARAM_TYPE_STRING,
            "Optional cast target (int / float / bool / str).",
            required=False,
            default_value=None,
        ),
    ],
)
def execute_set_context_variable(
    controller: "CameraController",
    context: Dict[ContextKey, Any],
    **kwargs,
) -> ActionResult:
    # ── 1) 필수 인자 확인 ────────────────────────────────────────────
    if "key" not in kwargs or "value" not in kwargs:
        return _ar_fail("'key' and 'value' are required.")

    try:
        key = str(_resolve_context_vars(kwargs["key"], context))
    except Exception as exc:
        return _ar_fail(f"Key resolution error: {exc}")

    raw_val = _resolve_context_vars(kwargs["value"], context)

    # ── 2) 'eval:' 처리  (utils.sandbox.safe_eval 사용) ──────────────
    if isinstance(raw_val, str) and raw_val.strip().lower().startswith("eval:"):
        expr = raw_val.strip()[5:]
        try:
            extra_env = {
                **context,                       # 기존 컨텍스트 변수 사용
                "ctx": SimpleNamespace(**context),
                "math": math,
            }
            if NUMPY_AVAILABLE and np is not None:
                extra_env["np"] = np
            evaluated_value = safe_eval(expr, extra_env)
        except Exception as exc:
            logger.error("[eval] error in '%s': %s", expr, exc, exc_info=True)
            return _ar_fail(f"Eval error: {exc} in expression '{expr}'")
    else:
        evaluated_value = _convert_value(raw_val)

    # ── 3) 선택적 타입 캐스팅 ─────────────────────────────────────────
    final_value = evaluated_value
    value_type: Optional[str] = kwargs.get("value_type")

    if value_type and evaluated_value is not None:
        try:
            if value_type == "int":
                final_value = int(evaluated_value)
            elif value_type == "float":
                final_value = float(evaluated_value)
            elif value_type == "bool":
                if isinstance(evaluated_value, str):
                    final_value = evaluated_value.strip().lower() == "true"
                else:
                    final_value = bool(evaluated_value)
            elif value_type == "str":
                final_value = str(evaluated_value)
        except (ValueError, TypeError):
            logger.warning(
                "Type-cast to %s failed – keeping original.", value_type,
                exc_info=True,
            )

    # ── 4) 저장 & 완료 ───────────────────────────────────────────────
    context[key] = final_value
    return _ar_success(
        f"Set context['{key}'] = {final_value!r}",
        {"context_key": key, "context_value": final_value},
    )

@register_action(
    id="endloop_control",
    display_name="End Loop",
    category="Flow",
    description="Marks the end of a loop block and jumps back to the start label if the condition is met.",
    arguments=[
        ActionArgument("loop_start_label", "Loop Start Label", PARAM_TYPE_STRING,
                       "The 'name' or 'id' of the step that starts the loop (e.g., a 'loop_control' step).",
                       required=True),
        ActionArgument("loop_counter_key", "Counter Key", PARAM_TYPE_CONTEXT_KEY,
                       "The context key for the current loop iteration count.",
                       required=True),
        ActionArgument("loop_total_key", "Total Iterations Key", PARAM_TYPE_CONTEXT_KEY,
                       "The context key for the total number of iterations.",
                       required=True),
    ]
)
def execute_endloop_control(controller: CameraController, context: Dict[ContextKey, Any], **kwargs) -> StepActionResult:
    log = logging.getLogger(__name__)

    # 파라미터 유효성 검사
    start_label = kwargs.get("loop_start_label")
    counter_key = kwargs.get("loop_counter_key")
    total_key = kwargs.get("loop_total_key")

    if not all([start_label, counter_key, total_key]):
        err_msg = "endloop_control FAILED: 'loop_start_label', 'loop_counter_key', and 'loop_total_key' are all required parameters."
        log.error(err_msg)
        return StepActionResult(status="error", message=err_msg)

    # 컨텍스트에서 카운터 및 전체 값 가져오기
    try:
        # 값을 가져오고 정수로 변환 시도
        counter = int(context.get(counter_key, 0))
        total = int(context.get(total_key))
    except (TypeError, ValueError):
        err_msg = f"endloop_control FAILED: Context values for '{counter_key}' or '{total_key}' are missing or not integers."
        log.error(err_msg)
        return StepActionResult(status="error", message=err_msg)

    # 루프 조건 확인: 현재 카운터가 전체 반복 횟수보다 작은지 확인
    # 루프는 0부터 total-1까지 실행됩니다. IncCnt 액션이 이미 카운터를 1 증가시켰다고 가정합니다.
    if counter < total:
        log.info(f"Loop continue triggered. Jumping to '{start_label}'. (Counter: {counter}/{total})")
        return StepActionResult(status="loop_continue", next_step=start_label)
    else:
        log.info(f"Loop completed. (Counter: {counter}/{total})")
        return StepActionResult(status="success")  # 루프 정상 종료

@register_action(
    id="set_exposure_auto",
    display_name="Set Exposure Auto Mode",
    category="Camera Control",
    description="Switches camera exposure mode between Auto and Manual.",
    arguments=[
        ActionArgument("enable", "Exposure Auto", PARAM_TYPE_BOOL,
                       "Set True for Auto exposure, False for Manual.", default_value=True, required=True)
    ]
)
def execute_set_exposure_auto(controller: CameraController, context: Dict[ContextKey, Any], **kwargs) -> ActionResult:
    en = bool(_convert_value(_resolve_context_vars(kwargs.get('enable'), context)))
    try:
        controller.set_param("ExposureAuto", en)
        mode = "Auto" if en else "Manual"
        return _ar_success(f"ExposureAuto = {mode}")
    except Exception as e:
        return _ar_fail(f"Set exposure error: {e}")

@register_action(
    id="set_width_with_increment",
    display_name="Set Width with Increment",
    category="Camera Control",
    description="Increases or decreases Width by step*inc. Stores updated Width in context.",
    arguments=[
        ActionArgument("direction", "Direction", PARAM_TYPE_ENUM,
                       "Change direction: increase or decrease", default_value="decrease",
                       options=["increase", "decrease"], required=True),
        ActionArgument("step_multiplier", "Step Multiplier", PARAM_TYPE_INT,
                       "Multiplier for increment step size", default_value=1,
                       min_value=1, max_value=100, step=1, required=False),
        ActionArgument("context_width_key", "Context Width Key", PARAM_TYPE_CONTEXT_KEY,
                       "Context key holding current Width value", default_value="Width", required=False)
    ]
)
def execute_set_width_with_increment(controller: CameraController, context: Dict[ContextKey, Any], **kwargs) -> ActionResult:
    direction = _resolve_context_vars(kwargs.get('direction'), context)
    step_mul = int(_resolve_context_vars(kwargs.get('step_multiplier', 1), context))
    key = str(_resolve_context_vars(kwargs.get('context_width_key', 'Width'), context))
    try:
        curr = int(context.get(key, controller.get_param("Width")))
        inc = controller.get_param("WidthInc")
        if direction == "decrease":
            new = max(controller.get_param("WidthMin"), curr - inc * step_mul)
        else:
            new = min(controller.get_param("WidthMax"), curr + inc * step_mul)
        controller.set_param("Width", new)
        context[key] = new
        return _ar_success(f"Width = {new}", {"new_width": new})
    except Exception as e:
        return _ar_fail(f"Width increment error: {e}")


# src/core/actions_impl.py (이 함수를 찾아서 아래 내용으로 교체)

@register_action(
    id="set_multi_roi_region",
    display_name="Set Multi ROI Region",
    category="Camera Control",
    description="Sets one Multi ROI region: Selector→Off→Offsets→Size→On",
    arguments=[
        ActionArgument("roi_index", "ROI Index", PARAM_TYPE_INT, "ROI selector (0-31)", required=True),
        ActionArgument("offset_x", "Offset X", PARAM_TYPE_INT, "Horizontal offset", required=True),
        ActionArgument("offset_y", "Offset Y", PARAM_TYPE_INT, "Vertical offset", required=True),
        ActionArgument("width", "Width", PARAM_TYPE_INT, "ROI width", required=True),  # width 인자 확인
        ActionArgument("height", "Height", PARAM_TYPE_INT, "ROI height", required=True)
    ]
)
def execute_set_multi_roi_region(controller: CameraController, context: Dict[ContextKey, Any],
                                 **kwargs) -> ActionResult:
    try:
        i = int(_resolve_context_vars(kwargs['roi_index'], context))
        x = int(_resolve_context_vars(kwargs['offset_x'], context))
        y = int(_resolve_context_vars(kwargs['offset_y'], context))
        w = int(_resolve_context_vars(kwargs['width'], context))  # width 처리 로직 확인
        h = int(_resolve_context_vars(kwargs['height'], context))

        active_controller, err = _get_active_controller(controller, context)
        if err:
            return err

        active_controller.set_param("MultiRoiSelector", i)
        active_controller.set_param("MultiRoiMode", "Off")
        active_controller.set_param("MultiRoiOffsetX", x)
        active_controller.set_param("MultiRoiOffsetY", y)
        active_controller.set_param("MultiRoiWidth", w)  # width 설정 로직 확인
        active_controller.set_param("MultiRoiHeight", h)
        active_controller.set_param("MultiRoiMode", "On")

        # MultiRoiValid는 RO(읽기전용)이므로 get_param 사용
        if hasattr(active_controller, 'get_param') and active_controller.get_param("MultiRoiValid") != 1:
            return _ar_fail(f"ROI{i} configuration reported as invalid by the camera.")

        return _ar_success(f"ROI{i} set X{x},Y{y},W{w},H{h}")
    except Exception as e:
        return _ar_fail(f"ROI error: {e}")


# src/core/actions_impl.py 파일에 아래 두 함수를 추가하세요.

@register_action(
    id="user_set_save",
    display_name="Save User Set",
    category="Camera Control",
    description="Saves the camera's current settings to a non-volatile memory slot (User Set).",
    arguments=[
        ActionArgument("set_id", "User Set ID", PARAM_TYPE_ENUM,
                       "The memory slot to save the settings to.",
                       default_value="UserSet1",
                       options=["Default", "UserSet1", "UserSet2", "UserSet3"])
    ]
)
def execute_user_set_save(controller: CameraController, context: Dict[ContextKey, Any], **kwargs) -> ActionResult:
    """
    지정된 User Set에 현재 카메라 설정을 저장합니다.
    1. UserSetSelector로 저장할 위치를 선택합니다.
    2. UserSetSave 명령을 실행합니다.
    """
    active_ctrl, err = _get_active_controller(controller, context)
    if err:
        return err

    set_id = str(_resolve_context_vars(kwargs.get("set_id", "UserSet1"), context))

    try:
        # 1. 저장할 슬롯 선택
        active_ctrl.set_param("UserSetSelector", set_id)

        # 2. 저장 명령 실행
        active_ctrl.execute_command("UserSetSave")

        logger.info(f"[{active_ctrl.cam_id}] Successfully saved current settings to '{set_id}'.")
        return _ar_success(f"Camera settings saved to '{set_id}'.")
    except Exception as e:
        msg = f"Failed to save User Set '{set_id}': {e}"
        logger.error(f"[{active_ctrl.cam_id}] {msg}", exc_info=True)
        return _ar_fail(msg)


@register_action(
    id="user_set_load",
    display_name="Load User Set",
    category="Camera Control",
    description="Loads settings from a non-volatile memory slot (User Set) into the camera.",
    arguments=[
        ActionArgument("set_id", "User Set ID", PARAM_TYPE_ENUM,
                       "The memory slot to load settings from.",
                       default_value="UserSet1",
                       options=["Default", "UserSet1", "UserSet2", "UserSet3"])
    ]
)
def execute_user_set_load(controller: CameraController, context: Dict[ContextKey, Any], **kwargs) -> ActionResult:
    """
    지정된 User Set에서 설정을 불러와 카메라에 적용합니다.
    1. UserSetSelector로 불러올 위치를 선택합니다.
    2. UserSetLoad 명령을 실행합니다.
    """
    active_ctrl, err = _get_active_controller(controller, context)
    if err:
        return err

    set_id = str(_resolve_context_vars(kwargs.get("set_id", "UserSet1"), context))

    try:
        # 1. 불러올 슬롯 선택
        active_ctrl.set_param("UserSetSelector", set_id)

        # 2. 로드 명령 실행
        active_ctrl.execute_command("UserSetLoad")

        # 로드 후에는 파라미터가 변경되므로 잠시 대기하는 것이 안정적일 수 있습니다.
        time.sleep(0.1)

        logger.info(f"[{active_ctrl.cam_id}] Successfully loaded settings from '{set_id}'.")
        return _ar_success(f"Camera settings loaded from '{set_id}'.")
    except Exception as e:
        msg = f"Failed to load User Set '{set_id}': {e}"
        logger.error(f"[{active_ctrl.cam_id}] {msg}", exc_info=True)
        return _ar_fail(msg)

@register_action(
    id="set_binning",
    display_name="Set Binning",
    category="Camera Control",
    description="Robust binning setter that respects BinningLocked and uses proper enum symbols.",
    arguments=[
        ActionArgument("selector", "Selector", PARAM_TYPE_ENUM,
                       "Which binning engine to control.", default_value="Logic",
                       options=["", "Logic", "Sensor"], required=False),
        ActionArgument("horizontal_mode", "Horizontal Mode", PARAM_TYPE_ENUM,
                       "Sum/Average; leave empty to keep.", default_value="",
                       options=["", "Sum", "Average"], required=False),
        ActionArgument("vertical_mode", "Vertical Mode", PARAM_TYPE_ENUM,
                       "Sum/Average; leave empty to keep.", default_value="",
                       options=["", "Sum", "Average"], required=False),
        ActionArgument("horizontal_factor", "Horizontal Factor", PARAM_TYPE_INT,
                       "1 disables binning.", default_value=1, min_value=1),
        ActionArgument("vertical_factor", "Vertical Factor", PARAM_TYPE_INT,
                       "1 disables binning.", default_value=1, min_value=1),
    ]
)
def execute_set_binning(controller: CameraController, context: Dict[ContextKey, Any], **kwargs) -> ActionResult:
    ctrl, err = _get_active_controller(controller, context)
    if err: return err

    sel   = str(_resolve_context_vars(kwargs.get("selector", "Logic"), context)).strip()
    hmode = str(_resolve_context_vars(kwargs.get("horizontal_mode", ""), context)).strip()
    vmode = str(_resolve_context_vars(kwargs.get("vertical_mode", ""), context)).strip()
    hf    = int(_resolve_context_vars(kwargs.get("horizontal_factor", 1), context))
    vf    = int(_resolve_context_vars(kwargs.get("vertical_factor", 1), context))

    def _nm(name): return ctrl._find_nodemap(name)
    def _entries(name):
        try: return ctrl.get_enumeration_entries(name) or []
        except: return []
    def _set_enum(name, val):
        nm = _nm(name);
        if not nm or name not in nm.features(): return False
        try:
            node = nm.getNode(name)
            if node and node.isWritable():
                node.setValue(val)  # ← 문자열 심볼
                return True
            if val in _entries(name):
                nm.set(name, val)
                return True
        except: pass
        return False
    def _set_int(name, val):
        nm = _nm(name)
        if not nm or name not in nm.features(): return False
        try: nm.set(name, int(val)); return True
        except: return False

    # 0) 그래빙 중이면 정지(잠금 해제용)
    with suppress(Exception):
        if ctrl.is_grabbing(): ctrl.stop_grab_safe(flush=True, revoke=True, wait_ms=500)

    # 1) Selector = Logic (기본) — Sensor는 AreaScan/TDI 조건에서 잠김
    if sel:
        _set_enum("BinningSelector", sel)  # values: Sensor(0)/Logic(1)  :contentReference[oaicite:8]{index=8}

    # 진단용: BinningLocked 확인
    locked_val = None
    try:
        nmH = _nm("BinningHorizontal")
        if nmH and "BinningLocked" in nmH.features():
            locked_val = int(nmH.get("BinningLocked"))
    except: pass

    # 2) 모드(있으면)
    if hmode:
        _set_enum("BinningHorizontalMode", hmode)  # Sum/Average  :contentReference[oaicite:9]{index=9}
    if vmode:
        _set_enum("BinningVerticalMode", vmode)    # Sum/Average  :contentReference[oaicite:10]{index=10}

    # 3) 팩터 → 'Xn' 문자열 매핑
    def _apply_factor(base_name, factor):
        nm = _nm(base_name)
        if not nm: return None
        entries = _entries(base_name)              # X1..X8  :contentReference[oaicite:11]{index=11}
        sym = f"X{factor}"
        if factor == 1:
            # X1 우선, 없으면 모드 Off 류 시도, 마지막으로 정수 1
            if sym in entries and _set_enum(base_name, sym): return f"{base_name}=X1"
            for off_node in ("BinningMode","BinningEnable","BinningSwitch"):
                if _set_enum(off_node, "Off"): return f"{base_name}=OFF(mode)"
            if _set_int(base_name, 1): return f"{base_name}=1"
            return None
        else:
            if sym in entries and _set_enum(base_name, sym): return f"{base_name}={sym}"
            # 일부 모델에서 정수 쓰기 허용 시
            if _set_int(base_name, factor): return f"{base_name}={factor}(int)"
            return None

    h_log = _apply_factor("BinningHorizontal", hf)
    v_log = _apply_factor("BinningVertical",   vf)

    # 실패 시 잠금/엔트리/셀렉터 진단 메시지 보강
    if not h_log or not v_log:
        diag = []
        try:
            sel_now = ctrl.get_param("BinningSelector")
            diag.append(f"Selector={sel_now}")
        except: pass
        try:
            diag.append(f"Locked={locked_val}")
        except: pass
        diag.append(f"H_entries={_entries('BinningHorizontal')}")
        diag.append(f"V_entries={_entries('BinningVertical')}")
        msg = " / ".join(diag)
        if not h_log and not v_log:
            return _ar_fail(f"Failed to set both H/V binning. {msg}")
        if not h_log:
            return _ar_fail(f"Failed to set BinningHorizontal. {msg}")
        if not v_log:
            return _ar_fail(f"Failed to set BinningVertical. {msg}")

    return _ar_success(f"Binning configured: {h_log}, {v_log}")


# --- REPLACE WHOLE FUNCTION --------------------------------------------------
@register_action(
    id="start_grab",
    display_name="Start Grabbing",
    category="Camera Control",
    description="Starts image acquisition. In live-view mode, this just sends AcquisitionStart.",
    arguments=[ActionArgument("buffer_count", "Buffer Count",
                              PARAM_TYPE_INT, "Number of buffers (ignored in live-view).",
                              default_value=64, min_value=4, max_value=128, step=1)]
)
def execute_start_grab(controller: CameraController, context, **kw):
    active_ctrl, err = _get_active_controller(controller, context)
    if err: return err

    buf_cnt = int(_resolve_context_vars(kw.get("buffer_count", 8), context))

    try:
        if _live_view_active(active_ctrl):
            # In live view, grabber is running. Just "arm" the camera.
            with suppress(Exception):
                active_ctrl.execute_command("AcquisitionStart", timeout=0.5)
            return _ar_success(f"[{active_ctrl.cam_id}] Acquisition armed (live-view active).")
        else:
            # Traditional mode: stop, then start.
            if active_ctrl.is_grabbing():
                active_ctrl.stop_grab(flush=True)
                time.sleep(0.05)
            active_ctrl.start_grab(buffer_count=buf_cnt)
            return _ar_success(f"[{active_ctrl.cam_id}] Grab started (buffers={buf_cnt}).")
    except Exception as exc:
        return _ar_fail(f"[{active_ctrl.cam_id}] start_grab failed: {exc}", exc_info=True)

# ---------------------------------------------------------------------------
@register_action(
    id="execute_feature_command_flexible",
    display_name="Execute Feature Command (Flexible)",
    category="Camera Control",
    description="Executes the first available command from a list of candidates.",
    arguments=[
        ActionArgument("command_candidates", "Command Candidates", PARAM_TYPE_STRING,
                       "Comma-separated list of command names to try in order (e.g., LUTGenerate,GenerateLUT).",
                       required=True),
    ]
)
def execute_feature_command_flexible(controller: CameraController, context: Dict[ContextKey, Any],
                                     **kwargs) -> ActionResult:
    """
    Tries to execute a command from a list of possible names.
    This is useful for handling features with slightly different names across camera models.
    """
    active_controller, err = _get_active_controller(controller, context)
    if err:
        return err

    candidates_str = str(_resolve_context_vars(kwargs.get('command_candidates', ''), context))
    if not candidates_str:
        return _ar_fail("No command candidates provided.")

    candidates = [cmd.strip() for cmd in candidates_str.split(',') if cmd.strip()]

    for cmd in candidates:
        try:
            # Try to execute the command. The controller's method will search all nodemaps.
            active_controller.execute_command(cmd, timeout=1.0)
            return _ar_success(f"[{active_controller.cam_id}] Successfully executed command '{cmd}'.")
        except CommandExecutionError:
            # This is expected if the command is not found, so we just log it and continue.
            logger.debug(f"[{active_controller.cam_id}] Candidate command '{cmd}' not found, trying next.")
            continue
        except Exception as e:
            # For other unexpected errors, it's better to fail.
            msg = f"[{active_controller.cam_id}] An unexpected error occurred while trying to execute '{cmd}': {e}"
            logger.error(msg, exc_info=True)
            return _ar_fail(msg)

    # If the loop completes without finding any valid command
    return _ar_fail(f"[{active_controller.cam_id}] None of the candidate commands were found or writable: {candidates}")

@register_action(
    id="stop_grab",
    display_name="Stop Grabbing",
    category="Camera Control",
    description="Stops image acquisition. In live-view mode, this just sends AcquisitionStop."
)
def execute_stop_grab(controller: CameraController, context, **__):
    active_ctrl, err = _get_active_controller(controller, context)
    if err: return err

    try:
        if _live_view_active(active_ctrl):
            # In live view, only stop the camera. The runner handles final cleanup.
            with suppress(Exception):
                 active_ctrl.execute_command("AcquisitionStop", timeout=0.5)
            return _ar_success(f"[{active_ctrl.cam_id}] Acquisition disarmed (live-view active).")
        else:
            # Traditional mode: stop the whole grabber.
            if not active_ctrl.is_grabbing():
                return _ar_success(f"[{active_ctrl.cam_id}] Already idle.")
            active_ctrl.stop_grab(flush=True)
            return _ar_success(f"[{active_ctrl.cam_id}] Grab stopped.")
    except Exception as exc:
        return _ar_fail(f"[{active_ctrl.cam_id}] stop_grab failed: {exc}", exc_info=True)


@register_action(
    id="execute_command",
    display_name="Execute Camera Command",
    category="Camera Control",
    description="Executes a specific camera command node. Special handling for 'AcquisitionStop'.",
    arguments=[
        ActionArgument("command_name", "Command Name", PARAM_TYPE_CAMERA_PARAM,
                       "The specific command node name to execute.", required=True)
    ]
)
def execute_execute_command(controller: CameraController, context: Dict[ContextKey, Any], **kwargs) -> ActionResult:
    """
    Executes a single, specific camera command.
    'AcquisitionStop' failures are ignored to improve sequence stability, as the camera might already be stopped.
    """
    active_controller, err = _get_active_controller(controller, context)
    if err:
        return err

    cmd = str(_resolve_context_vars(kwargs.get('command_name'), context))
    if not cmd:
        return _ar_fail("Command name is required but was not provided or resolved to an empty string.")

    try:
        active_controller.execute_command(cmd)
        return _ar_success(f"[{active_controller.cam_id}] Executed command '{cmd}'.")
    except Exception as e:
        # For 'AcquisitionStop', a failure is often not a critical error.
        # It usually means the camera was already stopped. We can safely ignore it.
        if 'acquisitionstop' in cmd.lower():
            msg = f"[{active_controller.cam_id}] Command '{cmd}' failed (likely already stopped), proceeding. Original error: {e}"
            logger.warning(msg)
            return _ar_success(msg)

        # For all other commands, a failure is a real issue.
        msg = f"[{active_controller.cam_id}] Command '{cmd}' failed: {e}"
        logger.error(msg, exc_info=True)
        return _ar_fail(msg)

@register_action(
    id="broadcast_camera_command",
    display_name="Broadcast Camera Command",
    category="Camera Control",
    description="Executes a GenICam *Command* node on a camera-list.",
    arguments=[
        ActionArgument("command_name", "Command Name", PARAM_TYPE_CAMERA_PARAM,
                       "GenICam command node.", required=True),
        ActionArgument("list_key", "List Key", PARAM_TYPE_CONTEXT_KEY,
                       "Context key holding cam-id list (optional).",
                       default_value="", required=False),
        ActionArgument("timeout", "Timeout (s)", PARAM_TYPE_FLOAT,
                       "wait_until_done timeout", default_value=0.5,
                       min_value=0.05, max_value=10.0, step=0.05),
    ],
)
def execute_broadcast_camera_command(controller, context, **kwargs):
    from src.core import controller_pool

    cmd      = str(_resolve_context_vars(kwargs["command_name"], context))
    list_key = kwargs.get("list_key") or ""
    timeout  = float(_resolve_context_vars(kwargs.get("timeout", 0.5), context))

    cam_ids = context.get(list_key) if list_key else list(controller_pool.controllers)
    if not cam_ids:
        return _ar_fail("No camera IDs available for broadcast")

    results: Dict[str, Union[str, Exception]] = {}
    for cid in cam_ids:
        ctrl = controller_pool.get(cid)
        if ctrl is None:
            results[cid] = "not registered"
            continue
        try:
            # 통합 execute_command() 사용 — DeviceModule 포함!
            ctrl.execute_command(cmd, timeout)
            results[cid] = "OK"
        except Exception as exc:                            # noqa: BLE001
            results[cid] = exc

    failed = {cid: str(v) for cid, v in results.items() if v != "OK"}
    if failed:
        return _ar_fail(f"Broadcast '{cmd}' NG on {len(failed)}/{len(cam_ids)} cam(s)",
                        {"details": failed})
    return _ar_success(f"Broadcast '{cmd}' OK on {len(cam_ids)} cam(s).")

@register_action(
    id="save_image",
    display_name="Save Image to File",
    category="Image",
    description="Saves image from context to file. Supports context vars in path. "
                "If no path is given, uses the current working directory.",
    arguments=[
        ActionArgument(
            "frame_context_key", "Frame Key", PARAM_TYPE_CONTEXT_KEY,
            "Context key of image (numpy array).",
            default_value="frame", required=True
        ),
        ActionArgument(
            "filepath", "File Path", PARAM_TYPE_FILE_SAVE,
            "Save path (supports context vars).  "
            "Leave empty → ./<cam_id>_<timestamp>.tiff",
            default_value="",            # ← 기본값 추가
            required=False               # ← 필수 ➔ 선택
        ),
    ]
)
def execute_save_image(controller: CameraController,
                       context: Dict[ContextKey, Any],
                       **kwargs) -> ActionResult:

    # 1) 프레임 가져오기 ---------------------------------------------------
    key = str(_resolve_context_vars(kwargs.get('frame_context_key'), context))
    frame = context.get(key)
    if frame is None or not isinstance(frame, np.ndarray):
        return _ar_fail(f"Invalid frame '{key}'")

    if not (NUMPY_AVAILABLE and IMAGE_UTILS_AVAILABLE):
        return _ar_fail("Save-image dependencies missing (NumPy / image_utils)")

    # 2) 경로 결정 (빈 값 → 실행 경로) --------------------------------------
    raw_path = _resolve_context_vars(kwargs.get('filepath', ''), context) or ""
    if raw_path.strip():
        save_path = Path(str(raw_path))
    else:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        cid = getattr(controller, "cam_id", "cam")
        save_path = Path.cwd() / f"{cid}_{ts}.tiff"

    # 3) 디렉터리 준비 & 저장 ---------------------------------------------
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ok = save_frame(frame, str(save_path))
        if ok:
            return _ar_success(f"Saved image → {save_path}")
        return _ar_fail(f"cv2.imwrite() returned False for {save_path}")
    except Exception as e:
        return _ar_fail(f"Save-image error: {e}")

@register_action(
    id="compare_brightness",
    display_name="Compare Brightness",
    category="Image",
    description="Compares the mean brightness of two images and fails if the difference exceeds the threshold.",
    arguments=[
        ActionArgument(
            "current_brightness_key", "Current Brightness Key", PARAM_TYPE_CONTEXT_KEY,
            "Context key for the current brightness value.",
            default_value="brightness_sd_mean", required=True
        ),
        ActionArgument(
            "previous_brightness_key", "Previous Brightness Key", PARAM_TYPE_CONTEXT_KEY,
            "Context key for the previous brightness value.",
            default_value="prev_brightness_sd", required=False
        ),
        # ⬇⬇⬇ FLOAT → STRING (min/max/step 제거)
        ActionArgument(
            "threshold", "Threshold", PARAM_TYPE_STRING,
            "Numeric literal or {ctx_key}. Ex) 10.0 or {brightness_threshold}",
            default_value="10.0", required=True
        ),
    ]
)
def execute_compare_brightness(controller: CameraController, context: Dict[ContextKey, Any], **kwargs) -> ActionResult:
    """
    current/previous 값 차이가 threshold(문자열·컨텍스트 치환 허용)를 초과하면 실패.
    prev 값이 없으면 cur 값을 baseline으로 저장하고 통과.
    """

    def _resolve_ctx(var):
        # "{key}" 형태면 context 치환, 아니면 원본 반환
        if isinstance(var, str) and var.startswith("{") and var.endswith("}"):
            return context.get(var[1:-1])
        return var

    def _coerce_float(x):
        # 숫자 문자열/숫자 → float, 실패 시 예외
        if x is None:
            raise ValueError("value is None")
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            s = x.strip().replace("_", "")
            return float(s)
        return float(x)

    # 키 이름도 컨텍스트 치환 허용(필요 시)
    curk = str(_resolve_ctx(kwargs.get("current_brightness_key", "brightness_sd_mean")))
    prevk = str(_resolve_ctx(kwargs.get("previous_brightness_key", "prev_brightness_sd")))

    # threshold: 문자열/치환 모두 허용 → 여기서 숫자로 변환
    thr_raw = _resolve_ctx(kwargs.get("threshold", "10.0"))
    try:
        thr = _coerce_float(thr_raw)
    except Exception:
        return _ar_fail(f"Threshold must be numeric, got {thr_raw!r}")

    cur = context.get(curk)
    prev = context.get(prevk)

    # 현재 값이 없으면 비교 자체가 불가
    if cur is None:
        return _ar_fail(f"Current brightness '{curk}' missing in context.")

    # prev 없으면 baseline 저장 후 성공(차이=0)
    if prev is None:
        context[prevk] = cur
        return _ar_success("Initial brightness stored", {"difference": 0.0})

    # 숫자 변환
    try:
        cur_f = _coerce_float(cur)
        prev_f = _coerce_float(prev)
    except Exception:
        return _ar_fail(f"Brightness values must be numeric; cur={cur!r}, prev={prev!r}")

    diff = abs(cur_f - prev_f)

    # 다음 비교를 위해 이번 값을 prev로 저장
    context[prevk] = cur

    if diff > thr:
        return _ar_fail(f"Brightness diff {diff}>{thr}", {"difference": diff})
    return _ar_success(f"Brightness diff {diff}<={thr}", {"difference": diff})

@register_action(
    id="compare_images_advanced",
    display_name="Compare Images (Advanced)",
    category="Image",
    description="Compares two images using an advanced metric.",
    arguments=[
        ActionArgument("frame_context_key_a", "Frame A Key", PARAM_TYPE_CONTEXT_KEY,
                       "Key of first image.", default_value="ref_frame", required=True),
        ActionArgument("frame_context_key_b", "Frame B Key", PARAM_TYPE_CONTEXT_KEY,
                       "Key of second image.", default_value="frame", required=True),
        ActionArgument("threshold", "Threshold", PARAM_TYPE_FLOAT,
                       "Threshold for mean brightness difference.", default_value=10.0, required=False),
        ActionArgument("fail_on_mismatch", "Fail on Mismatch?", PARAM_TYPE_BOOL,
                       "If True, action fails on mismatch.", default_value=True, required=False)
    ]
)
def execute_compare_images_advanced(controller: CameraController, context: Dict[ContextKey, Any],
                                    **kwargs) -> ActionResult:
    """[수정] _ar_success/_ar_fail만 사용 (ActionResult 직접 생성 제거)"""
    # 1) 파라미터
    a_key = str(_resolve_context_vars(kwargs.get('frame_context_key_a'), context))
    b_key = str(_resolve_context_vars(kwargs.get('frame_context_key_b'), context))
    metric = str(_resolve_context_vars(kwargs.get('metric', 'AbsDiffMean'), context))
    threshold = float(_resolve_context_vars(kwargs.get('threshold', 10.0), context))
    fail_on = bool(_convert_value(_resolve_context_vars(kwargs.get('fail_on_mismatch', True), context)))
    save_err = bool(_convert_value(_resolve_context_vars(kwargs.get('save_on_error', True), context)))
    path_tpl = kwargs.get("error_image_path", "") or ""

    # 2) 프레임 검증
    if not NUMPY_AVAILABLE:
        return _ar_fail("NumPy missing")
    frame_a = context.get(a_key)
    frame_b = context.get(b_key)
    if not isinstance(frame_a, np.ndarray) or not isinstance(frame_b, np.ndarray):
        return _ar_fail(f"Invalid frames in context keys '{a_key}' or '{b_key}'")

    if frame_a.shape != frame_b.shape:
        err_msg = f"Shape mismatch: {frame_a.shape} vs {frame_b.shape}"
        save_msg = _save_error_image_helper(controller, frame_b, "shape_mismatch", path_tpl) if save_err else ""
        return _ar_fail(err_msg + save_msg)

    # 3) 비교
    result = compare_frames_advanced(frame_a, frame_b, metric=metric)
    if result["status"] == "error":
        return _ar_fail(f"Image comparison failed: {result['message']}")

    value = float(result["value"])

    # 4) 임계 판단 (부등호 방향 메트릭별 상이)
    if metric.lower() in ["absdiffmean", "mse"]:
        is_mismatch = value > threshold
        op_char = ">"
    else:  # psnr, ssim
        is_mismatch = value < threshold
        op_char = "<"

    details = {"metric": metric, "value": round(value, 4), "threshold": threshold, "match": not is_mismatch}

    if is_mismatch:
        err_msg = f"Image mismatch: {metric} value {value:.4f} {op_char} threshold {threshold:.4f}."
        save_msg = _save_error_image_helper(controller, frame_b, f"{metric}_mismatch", path_tpl) if save_err else ""
        full_msg = err_msg + save_msg
        return _ar_fail(full_msg, details) if fail_on else _ar_success(full_msg, details)

    return _ar_success(f"Images match: {metric} value {value:.4f} is within threshold.", details)


@register_action(
    id="calculate_image_stats",
    display_name="Calculate Image Stats",
    category="Image",
    description="Calculates image statistics and stores them in context.",
    arguments=[
        ActionArgument("frame_context_key", "Frame Key", PARAM_TYPE_CONTEXT_KEY,
                       "Key of image.", default_value="frame", required=True),
        ActionArgument("output_context_key_prefix", "Output Prefix", PARAM_TYPE_STRING,
                       "Prefix for stored stats.", default_value="{context.frame_context_key}_stats_", required=False),
        ActionArgument("calculate_mean", "Mean?", PARAM_TYPE_BOOL,
                       "Calculate mean.", default_value=True, required=False),
        ActionArgument("calculate_stddev", "Std Dev?", PARAM_TYPE_BOOL,
                       "Calculate stddev.", default_value=True, required=False),
        ActionArgument("calculate_minmax", "Min/Max?", PARAM_TYPE_BOOL,
                       "Calculate min/max.", default_value=True, required=False)
    ]
)
def execute_calculate_image_stats(controller: CameraController, context: Dict[ContextKey, Any],
                                  **kwargs) -> ActionResult:
    key = str(_resolve_context_vars(kwargs.get('frame_context_key'), context))
    prefix = str(_resolve_context_vars(kwargs.get('output_context_key_prefix'), context))
    frame = context.get(key)
    if not isinstance(frame, np.ndarray):
        return _ar_fail(f"Invalid frame in context key '{key}'")

    res = calculate_stats(frame)
    if res['status'] == 'error':
        return _ar_fail(res.get('message', 'Stats calculation error'))

    vals = res.get('values', {})
    for k, v in vals.items():
        context[f"{prefix}{k}"] = v
    return _ar_success("Stats calculated", vals)

@register_action(
    id="wait_until",
    display_name="Wait Until",
    category="Flow",
    description="Polls a context key until the condition is satisfied or times out.",
    arguments=[
        ActionArgument(
            "condition_context_key", "Condition Context Key", PARAM_TYPE_CONTEXT_KEY,
            "Context key to read the left value from.", required=True
        ),
        ActionArgument(
            "operator", "Operator", PARAM_TYPE_STRING,
            "One of: >, >=, <, <=, ==, !=", required=True
        ),
        ActionArgument(
            "target_value", "Target Value", PARAM_TYPE_STRING,
            "Numeric literal or {ctx_key}.", required=True
        ),
        ActionArgument(
            "poll_interval_ms", "Poll Interval (ms)", PARAM_TYPE_INT,
            "Interval (in milliseconds) between checks.", default_value=50, required=False
        ),
        ActionArgument(
            "timeout_ms", "Timeout (ms)", PARAM_TYPE_INT,
            "Maximum time to wait (in milliseconds) before failing.", default_value=3000, required=False
        ),
    ]
)
def execute_wait_until(controller, context, **kwargs) -> ActionResult:
    import time
    op = str(kwargs.get("operator", ">")).strip()
    left_key = str(kwargs.get("condition_context_key"))
    target_raw = kwargs.get("target_value")
    poll_ms = int(kwargs.get("poll_interval_ms", 50)) or 50  # 0 방지
    timeout_ms = int(kwargs.get("timeout_ms", 3000))
    runner = kwargs.get("runner")

    def _resolve_ctx(var):
        if isinstance(var, str) and var.startswith("{") and var.endswith("}"):
            return context.get(var[1:-1])
        return var

    def _to_float(x):
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            s = x.strip().replace("_", "")
            try:
                return float(s)
            except ValueError:
                return None
        try:
            return float(x)
        except Exception:
            return None

    # 미리 우변(target) 치환 시도 (치환 결과가 아직 None이면 루프에서 재시도)
    start = time.monotonic()
    attempts = 0

    def _cmp(a, b):
        if op == ">":  return a > b
        if op == ">=": return a >= b
        if op == "<":  return a < b
        if op == "<=": return a <= b
        if op == "==": return a == b
        if op == "!=": return a != b
        raise ValueError(f"Unsupported operator: {op}")

    while True:
        attempts += 1

        # ✨ Stop 체크
        if runner and getattr(runner, "_stop_requested", False):
            return _ar_fail("Stopped by user.", {"user_aborted": True, "attempts": attempts})

        left_val_raw = context.get(left_key)
        right_val_raw = _resolve_ctx(target_raw)

        left = _to_float(left_val_raw)
        right = _to_float(right_val_raw)

        # 둘 중 하나라도 숫자가 아니면 비교하지 않고 대기(타임아웃 시 명확한 에러)
        if left is not None and right is not None:
            try:
                ok = _cmp(left, right)
            except TypeError as te:
                return _ar_fail(f"Type error during comparison: {te}", {
                    "left_raw": left_val_raw, "right_raw": right_val_raw,
                    "left": left, "right": right, "operator": op
                })
            if ok:
                return _ar_success(f"Condition satisfied: {left} {op} {right}", {
                    "left": left, "right": right, "attempts": attempts
                })

        elapsed_ms = (time.monotonic() - start) * 1000
        if elapsed_ms >= timeout_ms:
            # 왜 실패했는지 진단 정보 제공
            reason = []
            if left is None:
                reason.append(f"left '{left_key}' is None or non-numeric (raw={left_val_raw!r})")
            if right is None:
                reason.append(f"right target is None or non-numeric (raw={right_val_raw!r})")
            diag = "; ".join(reason) or "comparison returned False"
            return _ar_fail(f"Timeout waiting for {left_key} {op} target. Reason: {diag}", {
                "left_raw": left_val_raw, "right_raw": right_val_raw,
                "left": left, "right": right, "operator": op,
                "elapsed_ms": int(elapsed_ms), "attempts": attempts
            })

        time.sleep(poll_ms / 1000.0)


@register_action(
    id="get_parameter_metadata",
    display_name="Get Parameter Metadata",
    category="Camera Control",
    description="Safely fetches min/max/inc/access for a feature and stores them in context.  AccessMode 가 RW/RO/WO 가 아니면 실패 처리.",
    arguments=[
        ActionArgument("parameter_name", "Parameter Name", PARAM_TYPE_CAMERA_PARAM,
                       "Target parameter.", required=True),
        ActionArgument("output_context_prefix", "Output Prefix", PARAM_TYPE_CONTEXT_KEY_OUTPUT,
                       "Prefix for context keys.", default_value="", required=False)
    ]
)
def execute_get_parameter_metadata(controller: CameraController,
                                   context: Dict[ContextKey, Any], **kwargs) -> ActionResult:
    name   = _resolve_context_vars(kwargs.get("parameter_name"), context)
    prefix = str(_resolve_context_vars(kwargs.get("output_context_prefix", ""), context))

    try:
        meta_raw = controller.get_parameter_metadata(name) or {}
    except Exception as e:
        return _ar_fail(f"카메라에서 메타데이터 조회 실패: {e}")

    # ── 필드 보강 & AccessMode 검증 ──────────────────────────────────────────
    meta = {
        "value":        meta_raw.get("value"),
        "is_writeable": meta_raw.get("is_writeable", True),
        "access":       meta_raw.get("access", "").upper() or "NA",
        "type":         meta_raw.get("type", "Unknown"),
        "min":          meta_raw.get("min", 0),
        "max":          meta_raw.get("max", 2**31 - 1),
        "inc":          meta_raw.get("inc", 1),
        "unit":         meta_raw.get("unit", ""),
        "description":  meta_raw.get("description", ""),
    }

    if meta["access"] not in {"RW", "RO", "WO"}:
        return _ar_fail(f"{name} AccessMode 불명확: {meta['access']}")

    # ── 컨텍스트에 저장 (long form + short form) ───────────────────────────────
    for k, v in meta.items():
        # long form: e.g. "MW_MultiRoiWidth_min"
        context[f"{prefix}{name}_{k}"] = v
        # short form: e.g. "MW_min"
        context[f"{prefix}{k}"] = v

    return _ar_success(
        f"Metadata for {name}",
        {f"{prefix}{name}_{k}": v for k, v in meta.items()}
    )
@register_action(
    id="flush_grabber",
    display_name="Flush Grabber",
    category="Camera Control",
    description="Flushes DMA FIFO / buffer queue of the grabber."
)
def execute_flush_grabber(controller, context, **_):
    # [수정] 액티브 컨트롤러 가져오기
    active_controller, err = _get_active_controller(controller, context)
    if err:
        return err

    # [수정] _as_controller_list에 active_controller 전달
    ctrls = _as_controller_list(active_controller)
    if not ctrls:
        return _ar_success("No valid camera controller – nothing to flush.")
    results = {}
    for c in ctrls:
        cid = getattr(c, "cam_id", "?")
        try:
            g = c.get_grabber()
            if g is None:
                results[cid] = "skipped (grabber is None)"
            elif hasattr(g, "flush"):
                g.flush();        results[cid] = "flushed"
            elif hasattr(g, "flush_buffers"):
                g.flush_buffers(); results[cid] = "flushed"
            else:
                results[cid] = "skipped (no flush method)"
        except Exception as exc:
            logger.warning("[%s] flush_grabber warning: %s", cid, exc)
            results[cid] = f"skipped ({exc})"
    return _ar_success("Flush-grabber completed.", {"details": results})



@register_action(
    id="copy_context_value",
    display_name="Copy Context Value",
    category="Context",
    description="Copies a value from one context key to another. Deep copies numpy arrays or Python objects.",
    arguments=[
        ActionArgument("source_key", "Source Key", PARAM_TYPE_CONTEXT_KEY,
                       "Context key to copy from.", required=True),
        ActionArgument("target_key", "Target Key", PARAM_TYPE_CONTEXT_KEY_OUTPUT,
                       "Context key to store copied value.", required=True)
    ]
)
def execute_copy_context_value(controller: CameraController, context: Dict[ContextKey, Any], **kwargs) -> ActionResult:
    raw_src = kwargs.get("source_key")
    raw_tgt = kwargs.get("target_key")
    if raw_src is None or raw_tgt is None:
        return _ar_fail("Missing 'source_key' or 'target_key'.")
    src_key = str(_resolve_context_vars(raw_src, context))
    tgt_key = str(_resolve_context_vars(raw_tgt, context))
    if src_key not in context:
        return _ar_fail(f"Source key '{src_key}' not found in context.")
    val = context[src_key]
    try:
        new_val = val.copy() if isinstance(val, np.ndarray) else copy.deepcopy(val)
    except Exception:
        logger.warning("Deepcopy failed, using reference.")
        new_val = val
    context[tgt_key] = new_val
    return _ar_success(f"Copied context['{src_key}'] to context['{tgt_key}'].", {"copied_type": str(type(new_val))})



# ────────────────────────────── 내부 유틸 ───────────────────────────────────
def _to_num(s: str) -> Any:
    try:
        return float(s) if "." in s else int(s)
    except ValueError:
        return s


def _resolve(raw: str, ctx: Dict[str, Any]) -> Any:
    raw = raw.strip()
    if raw.startswith("{") and raw.endswith("}"):
        return ctx.get(raw[1:-1])
    return _to_num(raw)

def _cmp(a: Any, op: str, b: Any) -> bool:
    """
    안전(compare) 헬퍼 – execute_loop_control()에서 사용.
    • 숫자 형태의 문자열은 자동으로 int/float 로 변환
    • 비교연산 실패 시 마지막으로 문자열 비교로 폴백
    """
    def _coerce(x):
        # int/float 로 변환 시도 → 실패하면 그대로
        if isinstance(x, (int, float)):
            return x
        if isinstance(x, str):
            try:
                return int(x) if x.strip().isdigit() else float(x)
            except ValueError:
                pass
        return x                          # 원본 유지

    a_n, b_n = _coerce(a), _coerce(b)

    ops: dict[str, Callable[[Any, Any], bool]] = {
        "==": lambda p, q: p == q,
        "!=": lambda p, q: p != q,
        "<":  lambda p, q: p <  q,
        "<=": lambda p, q: p <= q,
        ">":  lambda p, q: p >  q,
        ">=": lambda p, q: p >= q,
    }
    if op not in ops:
        raise ValueError(f"Unsupported operator '{op}'")

    try:
        return ops[op](a_n, b_n)
    except TypeError:
        # 타입이 달라서 비교 실패 → 문자열 비교
        return ops[op](str(a_n), str(b_n))
@register_action(
    id="loop_control",
    display_name="Loop Control",
    category="Flow",
    description=(
        "Controls loop execution.\n"
        "① counter 방식 : loop_counter_key / loop_total_key\n"
        "② legacy 비교식: loop_condition_key / loop_operator / loop_target_value\n"
        "공통으로 loop_start_label 필수."
    ),
    arguments=[
        # legacy
        ActionArgument("loop_condition_key", "Condition Key",
                       PARAM_TYPE_CONTEXT_KEY, "Context key for current value."),
        ActionArgument("loop_operator", "Operator", PARAM_TYPE_STRING,
                       "Comparison operator (<, <=, ==, !=, >=, >)."),
        ActionArgument("loop_target_value", "Target Value", PARAM_TYPE_STRING,
                       "Literal or {ctx_var} to compare against."),
        # counter
        ActionArgument("loop_counter_key", "Counter Key",
                       PARAM_TYPE_CONTEXT_KEY, "Current loop counter."),
        ActionArgument("loop_total_key", "Total Key",
                       PARAM_TYPE_CONTEXT_KEY, "Total iterations."),
        # common
        ActionArgument("loop_start_label", "Loop Start Label",
                       PARAM_TYPE_STRING, "Step label to jump back to.", required=True),
    ],
)
def execute_loop_control(controller, context, **kwargs) -> StepActionResult:
    """
    Loop *entry*.
    • counter-모드 : loop_counter_key / loop_total_key
    • legacy-모드  : loop_condition_key / loop_operator / loop_target_value
    """
    log = logging.getLogger(__name__)

    # ── 1) counter-기반 모드 ──────────────────────────────────
    if "loop_counter_key" in kwargs or "loop_total_key" in kwargs:
        c_key = kwargs.get("loop_counter_key", "loop_counter")
        t_key = kwargs.get("loop_total_key", "loop_total")
        total = context.get(t_key)
        if total is None:
            return StepActionResult(status="error",
                                     message=f"loop_control: context['{t_key}'] missing")
        counter = context.get(c_key, 0)
        if counter >= total:
            log.info("Loop finished (%s/%s)", counter, total)
            return StepActionResult(status="loop_exit")
        log.debug("Loop continue (%s/%s)", counter, total)
        return StepActionResult(status="success")

    # ── 2) legacy 비교식 모드 ────────────────────────────────
    try:
        key     = kwargs["loop_condition_key"]
        op      = kwargs.get("loop_operator", "<")
        target  = kwargs["loop_target_value"]

        cur_val = context.get(key)
        tgt_val = _resolve_context_vars(target, context)

        if _cmp(cur_val, op, tgt_val):
            # 조건이 참 → 루프 body 실행
            return StepActionResult(status="success")
        # 조건이 거짓 → 루프 종료
        return StepActionResult(status="loop_exit")

    except KeyError as e:
        return StepActionResult(status="error",
                                message=f"loop_control: missing arg {e}")


@register_action(
    id="get_last_cached_frame",
    display_name="Get Last Cached Frame",
    category="Image",
    description="Gets the most recent frame from the controller's cache, populated by the background GrabWorker.",
    arguments=[
        ActionArgument("output_context_key", "Output Key", PARAM_TYPE_CONTEXT_KEY_OUTPUT,
                       "Context key to store the frame.", default_value="frame", required=True),
        ActionArgument("wait_for_new_frame", "Wait for New Frame?", PARAM_TYPE_BOOL,
                       "If True, waits a short period for a new frame to arrive after this action starts.",
                       default_value=True),
    ]
)
def execute_get_last_cached_frame(
        controller: CameraController,
        context: Dict[ContextKey, Any],
        *,
        runner: Optional[Any] = None,
        **kwargs
) -> ActionResult:
    """[최종 수정] 캐시된 프레임을 가져와 context에 저장하고, UI에도 표시합니다."""
    active_ctrl, err = _get_active_controller(controller, context)
    if err: return err
    if not active_ctrl: return _ar_fail("No active controller resolved.")

    key = str(_resolve_context_vars(kwargs.get("output_context_key"), context))
    wait = bool(_convert_value(_resolve_context_vars(kwargs.get("wait_for_new_frame", True), context)))

    if not _live_view_active(active_ctrl):
        return _ar_fail("This action requires a persistent grab session (live view).")

    if wait:
        time.sleep(0.1)

    frame = active_ctrl.get_last_np_frame()

    if frame is None:
        return _ar_fail("Failed to get a cached frame.")

    context[key] = frame

    # ★★★ [수정] execute_grab_frames와 동일한 UI 업데이트 로직 추가 ★★★
    if runner and hasattr(runner, 'test_frame_grabbed'):
        runner.test_frame_grabbed.emit(active_ctrl.cam_id, frame)

    return _ar_success(f"Retrieved cached frame into ctx['{key}'] and displayed.", {"shape": frame.shape})

@register_action(
    id="wait_for_stable_frame",
    display_name="Wait For Stable Frame",
    category="Image",
    description="Waits for the frame dimension to stabilize to the largest observed height.",
    arguments=[
        ActionArgument("output_context_key", "Output Key", PARAM_TYPE_CONTEXT_KEY_OUTPUT,
                       "Context key to store the stable frame.", default_value="frame", required=True),
        ActionArgument("stability_count", "Stability Count", PARAM_TYPE_INT,
                       "Number of consecutive frames that must have the target shape.", default_value=3, min_value=2),
        ActionArgument("poll_interval_ms", "Poll Interval (ms)", PARAM_TYPE_INT,
                       "How often to check for a new frame.", default_value=50, min_value=10),
        ActionArgument("timeout_ms", "Timeout (ms)", PARAM_TYPE_INT,
                       "Maximum time to wait for a stable frame.", default_value=3000, min_value=500),
        # Timeout을 3초로 늘림
    ]
)
def execute_wait_for_stable_frame(controller: CameraController, context: Dict[ContextKey, Any],
                                  **kwargs) -> ActionResult:
    active_ctrl, err = _get_active_controller(controller, context)
    if err: return err

    runner = kwargs.get("runner")

    key = str(_resolve_context_vars(kwargs.get("output_context_key"), context))
    stability_count = int(_resolve_context_vars(kwargs.get("stability_count", 3), context))
    poll_interval_s = int(_resolve_context_vars(kwargs.get("poll_interval_ms", 50), context)) / 1000.0
    timeout_s = int(_resolve_context_vars(kwargs.get("timeout_ms", 3000), context)) / 1000.0

    if not _live_view_active(active_ctrl):
        return _ar_fail("This action requires a persistent grab session (live view).")

    start_time = time.monotonic()

    # [핵심 수정] 가장 큰 높이를 가진 shape을 목표로 설정
    target_shape: Optional[Tuple[int, ...]] = None
    stable_counter = 0

    while time.monotonic() - start_time < timeout_s:
        # ✨ Stop 체크
        if runner and getattr(runner, "_stop_requested", False):
            return _ar_fail("Stopped by user.", {"user_aborted": True})

        frame = active_ctrl.get_last_np_frame()
        time.sleep(poll_interval_s)

        if frame is None:
            continue

        current_shape = frame.shape

        # 목표 shape이 없거나, 더 큰 높이의 프레임이 나타나면 목표를 갱신
        if target_shape is None or current_shape[0] > target_shape[0]:
            logger.debug(f"New target shape detected: {current_shape} (was {target_shape}). Resetting counter.")
            target_shape = current_shape
            stable_counter = 0

        # 현재 프레임이 목표 shape과 일치하는지 확인
        if current_shape == target_shape:
            stable_counter += 1
        else:
            # 목표와 다른 shape이 들어오면 카운터 리셋
            stable_counter = 0

        if stable_counter >= stability_count:
            logger.info(f"Frame shape stabilized at {target_shape} after {stable_counter} checks.")
            context[key] = frame
            return _ar_success(f"Stable frame {target_shape} retrieved into context['{key}']", {"shape": target_shape})

    return _ar_fail("Frame shape did not stabilize within timeout.",
                    {"timeout_ms": int(timeout_s * 1000), "required": stability_count})


@register_action(
    id="detect_scrambled_image",
    display_name="Detect Scrambled Image",
    category="Image",
    description="Detects image scrambling by comparing shape, SSIM, and mean difference. Saves faulty images.",
    arguments=[
        ActionArgument("current_frame_key", "Current Frame Key",
                       PARAM_TYPE_CONTEXT_KEY,
                       "Context key for the current frame to be checked.", required=True),
        ActionArgument("reference_frame_key", "Reference Frame Key",
                       PARAM_TYPE_CONTEXT_KEY,
                       "Context key for the known-good reference frame.", required=True),
        ActionArgument("ssim_threshold", "SSIM Threshold",
                       PARAM_TYPE_FLOAT,
                       "Lower bound for Structural Similarity Index (0.0 to 1.0). Lower is more tolerant.",
                       default_value=0.85, min_value=0.0, max_value=1.0, step=0.01),
        ActionArgument("mean_diff_threshold", "Mean-diff Threshold",
                       PARAM_TYPE_FLOAT,
                       "Upper bound for the mean absolute pixel difference (in DN).",
                       default_value=15.0, min_value=0.0, max_value=255.0, step=0.1),
        ActionArgument("ncc_threshold", "NCC Threshold",
                       PARAM_TYPE_FLOAT,
                       "Lower bound for Normalized Cross-Correlation (fallback when SSIM is unavailable).",
                       default_value=0.95, min_value=0.0, max_value=1.0, step=0.01),
        ActionArgument("save_on_error", "Save on Error?", PARAM_TYPE_BOOL,
                       "Save the current frame if it is detected as scrambled or has a shape mismatch.",
                       default_value=True),
        ActionArgument("error_image_path", "Error Image Path", PARAM_TYPE_FILE_SAVE,
                       "Path template for the faulty image. Supports {cam_id}, {timestamp}, {error_type}.",
                       default_value="logs/error_images/{cam_id}/scrambled_{timestamp}_{error_type}.tiff"),
    ]
)
def execute_detect_scrambled_image(controller: CameraController, context: Dict[ContextKey, Any],
                                   **kwargs) -> ActionResult:
    """[최종 수정] SSIM을 이용한 구조적 깨짐 감지에만 집중하고, 불일치 시 이미지를 저장합니다."""
    # 1. 파라미터 파싱
    ck = str(_resolve_context_vars(kwargs.get("current_frame_key"), context))
    rk = str(_resolve_context_vars(kwargs.get("reference_frame_key"), context))
    s_t = float(_resolve_context_vars(kwargs.get("ssim_threshold", 0.85), context))
    save_err = bool(_convert_value(_resolve_context_vars(kwargs.get("save_on_error", True), context)))
    path_tpl = kwargs.get("error_image_path", "")

    # 2. 프레임 획득 및 검증
    if not NUMPY_AVAILABLE: return _ar_fail("NumPy missing")
    cur = context.get(ck)
    ref = context.get(rk)
    if not isinstance(cur, np.ndarray): return _ar_fail(f"Invalid current frame in '{ck}'")
    if not isinstance(ref, np.ndarray):
        context[rk] = cur.copy()
        return _ar_success("Reference frame for scrambling detection stored.")

    # 3. Shape 검사
    if cur.shape != ref.shape:
        err_msg = f"Shape mismatch: current={cur.shape} vs reference={ref.shape}."
        save_msg = _save_error_image_helper(controller, cur, "shape_mismatch", path_tpl) if save_err else ""
        return _ar_fail(err_msg + save_msg)

    # 4. SSIM 계산 (image_utils 사용)
    ssim_result = compare_frames_advanced(ref, cur, metric="SSIM")
    if ssim_result["status"] == "error":
        return _ar_fail(f"SSIM calculation failed: {ssim_result['message']}")

    ssim_val = ssim_result["value"]
    is_scrambled = ssim_val < s_t

    meta = {"ssim": round(ssim_val, 4), "threshold": s_t, "scrambled": is_scrambled}
    if is_scrambled:
        err_msg = f"Scrambled image detected: SSIM {ssim_val:.4f} < threshold {s_t:.4f}."
        save_msg = _save_error_image_helper(controller, cur, "scrambled", path_tpl) if save_err else ""
        return _ar_fail(err_msg + save_msg, meta)

    return _ar_success(f"Image OK: SSIM {ssim_val:.4f} >= threshold {s_t:.4f}.", meta)
# ---------------------------------------------------------------------------


# ─────────────────────────────────────────────────────────────────────────────
#  Frame-loss monitor helper
# ─────────────────────────────────────────────────────────────────────────────
class FrameLossMonitor:
    def __init__(self):
        self.last_block_id: Optional[int] = None
        self.lost_count: int = 0
        self.total_count: int = 0

    def reset(self):
        self.__init__()

    def update(self, bid: int):
        if self.last_block_id is not None:
            gap = bid - self.last_block_id - 1
            if gap > 0:
                self.lost_count += gap
        self.last_block_id = bid
        self.total_count += 1

    @property
    def loss_ratio(self) -> float:
        denom = (self.lost_count + self.total_count)
        return 0.0 if denom == 0 else self.lost_count / denom

@register_action(
    id="reset_frame_loss_monitor",
    display_name="Reset Frame-loss Monitor",
    category="Diagnostics",                            # ← 수정
    description="Initialises or resets BlockID continuity tracker."
)
def reset_frame_loss_monitor(controller: CameraController,
                             context: Dict[ContextKey, Any], **kwargs) -> ActionResult:
    flm = context.get("frame_loss_monitor") or FrameLossMonitor()
    flm.reset()
    context["frame_loss_monitor"] = flm
    return _ar_success("Frame-loss monitor reset.",
                       {"lost": 0, "total": 0})


@register_action(
    id="reset_all_counters",
    display_name="Reset All Counters",
    category="Diagnostics",
    description="Resets all software (FrameLossMonitor) and hardware (Trigger/Stream loss) counters to prepare for a new test.",
)
def execute_reset_all_counters(controller: CameraController, context: Dict[ContextKey, Any], **kwargs) -> ActionResult:
    """
    [최종 완성본]
    하나의 액션으로 소프트웨어 및 모든 관련 하드웨어 카운터를 리셋하여,
    테스트 시작 전 상태를 완벽하게 초기화합니다.
    """
    details = {}
    reset_actions = []

    # 1. 소프트웨어 FrameLossMonitor 리셋
    flm = context.get("frame_loss_monitor") or FrameLossMonitor()
    flm.reset()
    context["frame_loss_monitor"] = flm
    reset_actions.append("Software_FrameLossMonitor")

    # 2. 하드웨어 카운터 리셋
    leader = controller_pool.first_controller()
    if leader:
        dev = getattr(leader.grabber, "device", None)
        stream = getattr(leader.grabber, "stream", None)

        # A. 트리거 손실 카운터 (CycleLostTriggerCount) 리셋
        if dev and "CycleLostTriggerCountReset" in dev.features():
            try:
                dev.execute("CycleLostTriggerCountReset")
                reset_actions.append("HW_TriggerLostCounter")
            except Exception as e:
                logger.warning(f"Failed to reset CycleLostTriggerCount: {e}")

        # B. 스트림 에러 카운터 (ErrorCount) 리셋
        if stream and "ErrorCountReset" in stream.features():
            try:
                stream.execute("ErrorCountReset")
                reset_actions.append("HW_StreamErrorCounters")
            except Exception as e:
                logger.warning(f"Failed to reset Stream ErrorCounts: {e}")

        # ★★★ [핵심 수정] C. 모든 이벤트 카운터 (EventCount) 리셋 ★★★
        if dev and "EventCountResetAll" in dev.features():
            try:
                dev.execute("EventCountResetAll")
                reset_actions.append("HW_EventCounters")
            except Exception as e:
                logger.warning(f"Failed to reset all EventCounts: {e}")

    else:
        logger.warning("No leader controller found, skipping hardware counter reset.")

    if not reset_actions:
        return _ar_success("No resettable counters found.")

    return _ar_success(f"All counters reset: {', '.join(reset_actions)}", {"reset_items": reset_actions})


# in src/core/actions_impl.py

@register_action(
    id="reset_hardware_counters",
    display_name="Reset Hardware Counters",
    category="Diagnostics",
    description="Resets hardware counters like CycleLostTriggerCount and all Stream ErrorCounts to zero before a test.",
)
def execute_reset_hardware_counters(controller: CameraController, context: Dict[ContextKey, Any],
                                    **kwargs) -> ActionResult:
    """
    다음 테스트를 위해 그래버의 모든 주요 하드웨어 카운터를 0으로 리셋합니다.
    """
    leader = controller_pool.first_controller()
    if not leader:
        return _ar_fail("No leader controller found to reset hardware counters.")

    dev = getattr(leader.grabber, "device", None)
    stream = getattr(leader.grabber, "stream", None)

    reset_actions = []

    # 1. 트리거 손실 카운터 리셋
    if dev and "CycleLostTriggerCountReset" in dev.features():
        try:
            dev.execute("CycleLostTriggerCountReset")
            reset_actions.append("CycleLostTriggerCount")
        except Exception as e:
            logger.warning(f"Failed to reset CycleLostTriggerCount: {e}")

    if dev and "EventCountResetAll" in dev.features():
        try:
            dev.execute("EventCountResetAll")
            reset_actions.append("EventCountResetAll")
        except Exception as e:
            logger.warning(f"Failed EventCountResetAll: {e}")


    if dev and "ErrorCountReset" in dev.features():
        try:
            dev.execute("ErrorCountReset")
            reset_actions.append("StreamErrorCounts")
        except Exception as e:
            logger.warning(f"Failed to reset Stream ErrorCounts: {e}")

    # 2. 스트림 에러 카운터 리셋
    if stream and "ErrorCountReset" in stream.features():
        try:
            stream.execute("ErrorCountReset")
            reset_actions.append("StreamErrorCounts")
        except Exception as e:
            logger.warning(f"Failed to reset Stream ErrorCounts: {e}")

    if not reset_actions:
        return _ar_success("No resettable hardware counters found or needed.")

    return _ar_success(f"Hardware counters reset successfully: {', '.join(reset_actions)}")


@register_action(
    id="check_frame_loss",
    display_name="Verify All Losses",
    category="Diagnostics",
    description="The ultimate validation tool for all SW/HW losses, including internal pipeline consistency checks (ACK vs. Processed).",
    arguments=[
        ActionArgument("max_lost_frames", "Max Lost Frames (App)", PARAM_TYPE_INT,
                       "Allowed software-level frame loss (from FrameLossMonitor).", default_value=0),
        ActionArgument("max_lost_triggers", "Max Lost Triggers (HW)", PARAM_TYPE_INT,
                       "Allowed hardware-level trigger loss (CycleLostTriggerCount).", default_value=0),
        ActionArgument("max_stream_errors", "Max Stream Errors (HW)", PARAM_TYPE_INT,
                       "Allowed total hardware-level stream data error count.", default_value=0),
    ]
)
def execute_check_frame_loss(controller: CameraController, context: Dict[ContextKey, Any], **kwargs) -> ActionResult:
    """
    [최종 통합 검증 버전]
    소프트웨어 및 하드웨어의 모든 주요 카운터와 핵심 이벤트를 종합적으로 검증합니다.
    특히, CxpTriggerAck와 CameraTriggerRisingEdge를 직접 비교하여
    트리거 파이프라인의 내부 일관성을 검증합니다.
    """
    errors = []
    details = {}

    # --- 1. 소프트웨어 레벨 프레임 손실 검사 ---
    max_lost_sw = int(_resolve_context_vars(kwargs.get("max_lost_frames", 0), context))
    flm: Optional[FrameLossMonitor] = context.get("frame_loss_monitor")
    if flm:
        details["sw_lost_frames"] = flm.lost_count
        if flm.lost_count > max_lost_sw:
            errors.append(f"SW Frame Loss ({flm.lost_count} > {max_lost_sw})")
    else:
        details["sw_lost_frames"] = "N/A"

    # --- 2. 하드웨어 레벨 카운터 및 이벤트 검사 ---
    leader = controller_pool.first_controller()
    if not leader:
        return _ar_fail("No leader controller found to check hardware counters.")

    dev = getattr(leader.grabber, "device", None)
    stream = getattr(leader.grabber, "stream", None)

    if dev:
        # A. 최종 하드웨어 손실 카운터 확인 (CycleLostTriggerCount)
        max_lost_triggers = int(_resolve_context_vars(kwargs.get("max_lost_triggers", 0), context))
        if "CycleLostTriggerCount" in dev.features():
            lost_triggers = int(dev.get("CycleLostTriggerCount"))
            details["hw_lost_triggers"] = lost_triggers
            if lost_triggers > max_lost_triggers:
                errors.append(f"HW Trigger Loss ({lost_triggers} > {max_lost_triggers})")
        else:
            details["hw_lost_triggers"] = "N/A"

        # B. 파이프라인 내부 일관성 검증 (ACK vs. Processing)
        if "EventSelector" in dev.features() and "EventCount" in dev.features():
            try:
                # B-1. CxpTriggerAck (통신 성공) 카운트 읽기
                dev.set("EventSelector", "CxpTriggerAck")
                ack_count = int(dev.get("EventCount"))
                acknowledged_triggers = ack_count // 2
                details["acknowledged_triggers"] = acknowledged_triggers

                # B-2. CameraTriggerRisingEdge (처리 시작) 카운트 읽기
                dev.set("EventSelector", "CameraTriggerRisingEdge")
                processed_triggers = int(dev.get("EventCount"))
                details["processed_triggers"] = processed_triggers

                # B-3. 두 카운트 직접 비교
                if acknowledged_triggers != processed_triggers:
                    errors.append(
                        f"Pipeline Inconsistency: Camera Acknowledged {acknowledged_triggers} triggers, but only Processed {processed_triggers}.")
            except Exception as e:
                errors.append(f"Failed to verify event counters: {e}")
        else:
            logger.warning("Event counter features not available, skipping pipeline consistency check.")
            details["acknowledged_triggers"] = "N/A"
            details["processed_triggers"] = "N/A"

    if stream:
        # C. 데이터 스트림 에러 검사
        max_stream_errors = int(_resolve_context_vars(kwargs.get("max_stream_errors", 0), context))
        total_stream_errors = 0
        if "ErrorSelector" in stream.features() and "ErrorCount" in stream.features():
            try:
                q_enum = query.enum_entries("ErrorSelector")
                error_enums = stream.get(q_enum, list) or []
                for error_type in error_enums:
                    stream.set("ErrorSelector", error_type)
                    count = int(stream.get("ErrorCount"))
                    if count > 0:
                        details[f"hw_stream_error_{error_type}"] = count
                        total_stream_errors += count
            except Exception as e:
                logger.error(f"Failed to read stream error counters: {e}")

        details["hardware_total_stream_errors"] = total_stream_errors
        if total_stream_errors > max_stream_errors:
            errors.append(f"HW Stream Errors ({total_stream_errors} > {max_stream_errors})")
    else:
        details["hardware_total_stream_errors"] = "N/A"

    # --- 3. 최종 결과 메시지 생성 및 반환 ---
    summary_parts = [
        f"SW_Lost: {details.get('sw_lost_frames', 'N/A')}",
        f"HW_Lost: {details.get('hw_lost_triggers', 'N/A')}",
        f"HW_Stream_Errors: {details.get('hardware_total_stream_errors', 'N/A')}",
        f"ACK'd: {details.get('acknowledged_triggers', 'N/A')}",
        f"Processed: {details.get('processed_triggers', 'N/A')}"
    ]
    summary_message = ", ".join(summary_parts)

    if errors:
        final_message = "Validation FAILED: " + " | ".join(errors)
        return _ar_fail(final_message, details)

    return _ar_success(f"Validation PASSED. [{summary_message}]", details)
# ─────────────────────────────────────────────────────────────────────────────
#  RANDOM-DEFECT PIXEL DETECTOR  (DN·σ 혼합 임계값)
# ─────────────────────────────────────────────────────────────────────────────
@register_action(
    id="detect_random_defect_pixel",
    display_name="Detect Random Defect Pixel",
    category="Image",
    description=(
        "Current vs. previous frame 절대차를 이용해 랜덤 디펙트 픽셀 검출. "
        "σ-기반 자동 임계값·ROI·CSV 로깅 지원."
    ),
    arguments=[
        ActionArgument("current_frame_key",  "Current Frame Key",  PARAM_TYPE_CONTEXT_KEY,
                       "현재 프레임 저장 키", default_value="frame"),
        ActionArgument("previous_frame_key", "Previous Frame Key", PARAM_TYPE_CONTEXT_KEY,
                       "이전 프레임 저장 키(baseline)", default_value="previous_frame"),
        ActionArgument("threshold_dn",    "|Δ| Threshold (DN)", PARAM_TYPE_INT,
                       "절대 DN 임계값 (σ 기준 미사용 시)", default_value=5,
                       min_value=1, max_value=255, step=1),
        ActionArgument("threshold_sigma", "Threshold (σ×)", PARAM_TYPE_FLOAT,
                       "0보다 크면 σ×N 방식으로 자동 계산", default_value=0.0,
                       min_value=0.0, max_value=10.0, step=0.1),
        ActionArgument("fail_on_defect",  "Fail if defect?", PARAM_TYPE_BOOL,
                       "디펙트 발생 시 액션 failure 로 반환", default_value=True),
        ActionArgument("log_details",     "CSV Log Details?", PARAM_TYPE_BOOL,
                       "좌표·Δ DN CSV 로 상세 로깅", default_value=False),
        ActionArgument("max_details_per_frame", "Max Rows/Frame", PARAM_TYPE_INT,
                       "상세 로깅 최대 행 수 (무제한: 0)", default_value=0,
                       min_value=0, max_value=100000),
        ActionArgument("save_mask",       "Save Bool-mask?", PARAM_TYPE_BOOL,
                       "bool 마스크를 context 에 저장", default_value=False),
        ActionArgument("roi",             "ROI x,y,w,h", PARAM_TYPE_STRING,
                       "분석 ROI (빈 값 = 전체)"),
        ActionArgument("count_key",  "Count Key",  PARAM_TYPE_CONTEXT_KEY_OUTPUT,
                       "디펙트 개수 저장 키", default_value="random_defect_count"),
        ActionArgument("coords_key", "Coords Key", PARAM_TYPE_CONTEXT_KEY_OUTPUT,
                       "좌표 리스트 저장 키", default_value="random_defect_coords"),
        ActionArgument("mask_key",   "Mask Key",   PARAM_TYPE_CONTEXT_KEY_OUTPUT,
                       "마스크 저장 키", default_value="random_defect_mask"),
    ]
)
def execute_detect_random_defect_pixel(controller: CameraController, context: Dict[ContextKey, Any],
                                       **kwargs) -> ActionResult:
    """[수정] _ar_success/_ar_fail만 사용 (ActionResult 직접 생성 제거)"""
    # 1) 파라미터
    cur_key = str(_resolve_context_vars(kwargs.get("current_frame_key"), context))
    prev_key = str(_resolve_context_vars(kwargs.get("previous_frame_key"), context))
    fail_on = bool(_convert_value(_resolve_context_vars(kwargs.get("fail_on_defect", True), context)))
    save_err = bool(_convert_value(_resolve_context_vars(kwargs.get("save_on_error", True), context)))
    path_tpl = kwargs.get("error_image_path", "") or ""
    thr_dn = int(_resolve_context_vars(kwargs.get("threshold_dn", 5), context))
    thr_sigma = float(_resolve_context_vars(kwargs.get("threshold_sigma", 0.0), context))

    # 2) 프레임
    if not NUMPY_AVAILABLE:
        return _ar_fail("NumPy missing")
    curr = context.get(cur_key)
    prev = context.get(prev_key)

    if prev is None:
        if isinstance(curr, np.ndarray):
            context[prev_key] = curr.copy()
            return _ar_success("Baseline frame for defect detection stored.")
        return _ar_fail("Initial frame missing, can't store baseline.")

    if not isinstance(curr, np.ndarray):
        return _ar_fail("Valid current frame not found.")

    if curr.shape != prev.shape:
        err_msg = f"Shape mismatch for defect detection: {curr.shape} vs {prev.shape}"
        save_msg = _save_error_image_helper(controller, curr, "shape_mismatch_defect", path_tpl) if save_err else ""
        context[prev_key] = curr.copy()  # 다음 비교를 위해 업데이트
        return _ar_fail(err_msg + save_msg)

    # 3) 임계값/검출
    if thr_sigma > 0.0:
        sigma = float(np.std(prev.astype(np.float32)))
        thr = max(1, int(round(sigma * thr_sigma)))
    else:
        thr = thr_dn

    diff = np.abs(curr.astype(np.int16) - prev.astype(np.int16))
    count = int(np.sum(diff >= thr))
    context[prev_key] = curr.copy()  # baseline 갱신

    meta = {"defect_count": count, "threshold_used": thr}

    if count > 0:
        err_msg = f"Detected {count} random defect pixels."
        save_msg = _save_error_image_helper(controller, curr, "defect_pixel", path_tpl) if save_err else ""
        full_msg = err_msg + save_msg
        return _ar_fail(full_msg, meta) if fail_on else _ar_success(full_msg, meta)

    return _ar_success(f"No random defect pixels found (count={count}).", meta)


# ─────────────────────────── NEW: Read-/Assert-Parameter ──────────────────────────
@register_action(
    id="read_parameter",
    display_name="Read Parameter",
    category="Camera Control",
    description="Reads a camera feature and stores it in context.",
    arguments=[
        ActionArgument("feature", "Feature Name", PARAM_TYPE_CAMERA_PARAM,
                       "GenICam feature to read.", required=True),
        ActionArgument("output_context_key", "Output Key", PARAM_TYPE_CONTEXT_KEY_OUTPUT,
                       "Context key to store the value.", default_value="", required=False)
    ]
)
def execute_read_parameter(controller: CameraController, context: Dict[ContextKey, Any], **kwargs) -> ActionResult:
    feat = str(_resolve_context_vars(kwargs.get("feature"), context))
    out_key = str(_resolve_context_vars(kwargs.get("output_context_key") or feat, context))
    try:
        val = controller.get_param(feat)
        context[out_key] = val
        return _ar_success(f"{feat} → {val}", {out_key: val})
    except Exception as e:
        return _ar_fail(f"Read parameter error: {e}")


@register_action(
    id="assert_feature",
    display_name="Assert Feature",
    category="Camera Control",
    description="Reads a feature and compares it with an expected value.",
    arguments=[
        ActionArgument("feature", "Feature Name", PARAM_TYPE_CAMERA_PARAM,
                       "Feature to read.", required=True),
        ActionArgument("expected", "Expected Value", PARAM_TYPE_STRING,
                       "Expected value (string-compare after read).", required=True)
    ]
)
def execute_assert_feature(controller: CameraController, context: Dict[ContextKey, Any], **kwargs) -> ActionResult:
    feat = str(_resolve_context_vars(kwargs.get("feature"), context))
    exp  = str(_resolve_context_vars(kwargs.get("expected"), context))
    try:
        cur = str(controller.get_param(feat))
        if cur == exp:
            return _ar_success(f"{feat} OK ({cur})")
        return _ar_fail(f"{feat} mismatch: {cur} ≠ {exp}")
    except Exception as e:
        return _ar_fail(f"Assert feature error: {e}")


@register_action(
    id="connect_all_cameras",
    display_name="[DEPRECATED] Connect ALL Cameras",
    category="System",
    description="[DEPRECATED] Use application startup logic instead. This action is no longer recommended."
)
def execute_connect_all_cameras(controller, context, **kwargs):
    # 이 액션은 더 이상 사용하지 않는 것을 권장합니다.
    # 앱 시작 시 한 번만 연결하는 것이 올바른 구조입니다.
    # 만약을 위해 호출되더라도 동작하도록 유지합니다.
    from src.core.camera_controller import CameraController
    from src.core import controller_pool

    logger.warning("connect_all_cameras action is deprecated and was called from a sequence. "
                   "It's recommended to handle connections only at application startup.")

    try:
        CameraController.connect_all()
        cam_ids = controller_pool.list_ids()
        context["cam_ids"] = cam_ids
        return _ar_success(f"Re-connected {len(cam_ids)} cameras.", {"all_cam_ids": cam_ids})
    except Exception as e:
        return _ar_fail(f"connect_all() failed: {e}")


@register_action(
    id="get_controller_ids",
    display_name="Get Controller IDs",
    category="Context",
    description="Gets all connected camera IDs from controller_pool and stores them in context.",
    arguments=[
        ActionArgument("output_key", "Output Key", PARAM_TYPE_CONTEXT_KEY_OUTPUT,
                       "Context key to store the list of camera IDs.", default_value="cam_ids")
    ]
)
def execute_get_controller_ids(controller: CameraController,
                               context: Dict[ContextKey, Any],
                               **kwargs) -> ActionResult:
    """
    controller_pool에서 현재 연결된 모든 카메라 ID를 수집하여
    - cam_ids (기본 키)  : 리스트
    - ALL_CAMS           : 동일 리스트(가독성용)
    - ALL_CAMS|length    : 카메라 수
    를 컨텍스트에 기록한다.
    """
    from src.core import controller_pool

    key = str(_resolve_context_vars(kwargs.get("output_key", "cam_ids"), context))
    try:
        cam_ids = controller_pool.ids()
        context[key] = cam_ids
        # 추가 저장 ─ 포맷 치환용
        context["ALL_CAMS"] = cam_ids
        context["ALL_CAMS|length"] = len(cam_ids)
        return _ar_success(f"Found {len(cam_ids)} connected controllers.",
                           {key: cam_ids, "ALL_CAMS|length": len(cam_ids)})
    except Exception as exc:
        return _ar_fail(f"Failed to get controller IDs: {exc}")

@register_action(
    id="foreach_controller",
    display_name="Foreach Controller",
    category="Control",
    description=("Iterates cam_id list.  Each call sets "
                 "`current_cam` & returns loop_continue / loop_exit."),
    arguments=[
        ActionArgument("list_key",    "List key",    PARAM_TYPE_CONTEXT_KEY,
                       "Context key with cam_id list.", default_value="cam_ids"),
        ActionArgument("index_key",   "Index key",   PARAM_TYPE_CONTEXT_KEY_OUTPUT,
                       "Key that stores current index.", default_value="cam_iter_idx"),
        ActionArgument("current_key", "Current key", PARAM_TYPE_CONTEXT_KEY_OUTPUT,
                       "Key that stores current cam_id.", default_value="current_cam"),
    ]
)
def execute_foreach_controller(controller, context, **kwargs):
    list_key    = kwargs.get("list_key", "cam_ids")
    index_key   = kwargs.get("index_key", "cam_iter_idx")
    current_key = kwargs.get("current_key", "current_cam")

    cam_list = context.get(list_key) or []
    if not isinstance(cam_list, (list, tuple)):
        return _ar_fail(f"Context[{list_key}] is not a list.")

    idx = int(context.get(index_key, 0))
    if idx < len(cam_list):
        cam_id = cam_list[idx]
        context[current_key] = cam_id
        context[index_key]   = idx + 1     # advance
        return {"status": "loop_continue",
                "message": f"{idx+1}/{len(cam_list)} → {cam_id}",
                "current_cam": cam_id}
    else:
        context[index_key] = 0             # reset for next run
        return {"status": "loop_exit", "message": "foreach complete"}

# src/core/actions_impl.py  (가장 아래쪽에 추가)
@register_action(
    id="endforeach_controller",
    display_name="End Foreach Controller",
    category="Control",
    description="Marks the end of a foreach_controller loop block."
)
def execute_endforeach_controller(controller, context, **kw):
    return _ar_success("EndForeach reached.")
@register_action(
    id="broadcast_software_trigger",
    display_name="Broadcast SW-Trigger",
    category="Camera Control",
    description=("보드-트리거(StartCycle / CxpTriggerMessageSend)가 "
                 "없거나 실패하면 각 카메라의 trigger_software_safe()로 "
                 "폴백합니다."),
    arguments=[
        ActionArgument("list_key", "List key", PARAM_TYPE_CONTEXT_KEY,
                       "Context key with cam-ID list (optional).",
                       default_value="", required=False),
        ActionArgument("timeout", "Timeout (s)", PARAM_TYPE_FLOAT,
                       "wait_until_done() per command.",
                       default_value=0.5, min_value=0.05,
                       max_value=10.0, step=0.05, required=False),
    ],
)
def execute_broadcast_software_trigger(controller, context, **kw):
    from src.core import controller_pool

    list_key = kw.get("list_key") or ""
    timeout  = float(_resolve_context_vars(kw.get("timeout", 0.5), context))

    cam_ids = context.get(list_key) if list_key else list(controller_pool.controllers)
    if not cam_ids:
        return _ar_fail("No cameras available for broadcast")

    # ───────────────────── ① Grabber-remote 보드 트리거 시도 ─────────────────────
    sent_by_board = False
    for cid in cam_ids:
        ctrl = controller_pool.get(cid)
        if ctrl is None:
            continue
        g_remote = getattr(ctrl.grabber, "remote", None)
        if g_remote is None:
            continue
        for cmd in ("StartCycle", "CxpTriggerMessageSend"):
            try:
                node = g_remote.getNode(cmd)
                if node and node.isWritable():
                    node.execute(); node.wait_until_done(timeout)
                    sent_by_board = True
                    break
            except Exception:
                continue
        if sent_by_board:
            break                     # 한 번 성공하면 전체가 동일한 LinkTrigger0 을 받음

    # ───────────────────── ② (보드 실패 시) 카메라 폴백 ────────────────────────
    ok, failed = {}, {}
    if not sent_by_board:
        for cid in cam_ids:
            ctrl = controller_pool.get(cid)
            if ctrl is None:
                failed[cid] = "not registered"
                continue
            try:
                ctrl.trigger_software_safe(timeout=timeout)
                ok[cid] = "SW-trigger OK"
            except Exception as exc:
                failed[cid] = str(exc)

    # ───────────────────── ③ 결과 정리 ─────────────────────────────────────────
    if sent_by_board:
        return _ar_success(f"Board trigger OK → {len(cam_ids)} cam(s).")

    if failed:
        return _ar_fail(f"Trigger NG on {len(failed)}/{len(cam_ids)} cam(s)",
                        {"failed": failed, "ok": ok})
    return _ar_success(f"SW-trigger broadcast → {len(ok)} cam(s) OK",
                       {"ok": ok})
def _set_first(dev, candidates, value):
    """candidates 리스트 중 dev.features() 에 존재하는 첫 노드에 value 를 설정."""
    for name in candidates:
        if name in dev.features():
            dev.set(name, value)
            return name
    raise RuntimeError(f"no matching feature for {candidates}")

# ─── 헬퍼 ──────────────────────────────────────────────────────────
def _set(nm: _NodeMap, names: Sequence[str], value: Any) -> Optional[str]:
    """candidates 중 첫 노드에 value를 쓰고, 사용한 노드명을 반환."""
    for n in names:
        if n in nm.features():
            nm.set(n, value)
            return n
    return None


def _exec(nm: _NodeMap, names: Sequence[str], timeout: float = 0.5) -> Optional[str]:
    """
    candidates 중 첫 Command 노드를 execute 하고 wait_until_done() (있으면) 호출.
    사용한 노드명을 반환, 실패 시 None.
    """
    for n in names:
        if n in nm.features():
            nm.execute(n)
            if hasattr(nm, "wait_until_done"):
                nm.wait_until_done(timeout)
            return n
    return None
# src/core/actions_impl.py  (핵심 부분만)

def _safe_set(node_map, key, value):
    """Node 가 있으면 set(), 없으면 False 리턴."""
    if key not in node_map.features():
        logger.warning("[TrigInit] '%s' not in features() – skipped", key)
        return False
    try:
        node_map.set(key, value)
        return True
    except GenTLException as ge:
        logger.warning("[TrigInit] set(%s=%s) failed: %s – skipped", key, value, ge)
        return False
@register_action(
    id="configure_internal_trigger",
    display_name="Configure Internal Trigger",
    category="Grabber",
    description=(
        "Device0 CIC 설정: RC/Source 설정 후 주기 노드(CycleTriggerPeriod / CyclePeriodUs / "
        "CycleTargetPeriod / CycleMinimumPeriod)로 주기 쓰기. Min 메타로 클램프 + read-back 검증.\n"
        "옵션으로 ExposureReadoutOverlap을 True로 설정(기본). 주기 노드가 없으면 fixed-mode로 처리."
    ),
    arguments=[
        ActionArgument(
            "cycle_trigger_source",
            "Cycle Trigger Source",
            PARAM_TYPE_ENUM,
            "Device::CycleTriggerSource 값",
            default_value="Immediate",
            options=["Immediate","StartCycle","C2C",
                     "LIN1","LIN2","LIN3","LIN4","LIN5","LIN6","LIN7","LIN8",
                     "QDC1","QDC2","QDC3","QDC4",
                     "MDV1","MDV2","MDV3","MDV4",
                     "DIV1","DIV2","DIV3","DIV4"],
        ),
        ActionArgument(
            "cycle_period_us",
            "Cycle Period (µs)",
            PARAM_TYPE_FLOAT,
            "요청 주기(µs). 하드웨어 Min 이상으로 클램프하여 적용.",
            default_value=20000.0, min_value=1.0, step=1.0,
        ),
        ActionArgument(
            "set_overlap_true",
            "Set ExposureReadoutOverlap",
            PARAM_TYPE_BOOL,
            "트리거 설정 시 ExposureReadoutOverlap을 True로 설정할지 여부.",
            default_value=True
        ),
    ],
)
def execute_configure_internal_trigger(
    controller: CameraController,
    context: Dict[ContextKey, Any],
    *,
    cycle_trigger_source: str = "Immediate",
    cycle_period_us: float = 20000.0,
    set_overlap_true: bool = True,
    **__,
) -> ActionResult:
    dev = getattr(controller.grabber, "device", None)
    if dev is None:
        return _ar_fail("Grabber Device0 모듈이 없습니다.")

    try:
        feats = dev.features()

        # 0) (선행) ExposureReadoutOverlap = True (가능할 때)
        if set_overlap_true and "ExposureReadoutOverlap" in feats:
            try:
                dev.set("ExposureReadoutOverlap", True)
            except Exception:
                with suppress(Exception):
                    dev.set("ExposureReadoutOverlap", "On")

        # 1) RC 모드
        if "CameraControlMethod" in feats:
            dev.set("CameraControlMethod", "RC")

        # 2) Source
        if "CycleTriggerSource" in feats:
            dev.set("CycleTriggerSource", cycle_trigger_source)

        # 3) 주기 노드 선택
        period_node = None
        for cand in ("CycleTriggerPeriod", "CyclePeriodUs", "CycleTargetPeriod", "CycleMinimumPeriod"):
            if cand in feats:
                period_node = cand
                break

        # 4) 주기 노드가 없으면 fixed-mode
        if period_node is None:
            min_cur = float(dev.get("CycleMinimumPeriod")) if "CycleMinimumPeriod" in feats else 0.0
            msg = (f"CIC configured → src={cycle_trigger_source}, mode=fixed, "
                   f"applied={min_cur:.1f}µs; no writable period node; CIC runs at CycleMinimumPeriod")
            controller.logger.info("[%s] %s", controller.cam_id, msg)
            return _ar_success(msg, {"mode": "fixed", "applied_us": min_cur})

        # 5) Min 메타로 클램프
        requested = float(cycle_period_us)
        applied = requested
        try:
            from egrabber import query
            q_min = query.info(period_node, "Min")
            min_limit = dev.get(q_min, float) if q_min else None
            if isinstance(min_limit, (int, float)):
                applied = max(requested, float(min_limit))
        except Exception:
            pass

        # 6) 쓰기 + read-back
        dev.set(period_node, applied)
        read_back = float(dev.get(period_node))

        msg = (f"CIC configured → src={cycle_trigger_source}, mode=variable, "
               f"set={applied:.1f}µs, read={read_back:.1f}µs via {period_node} | ERO={'On' if set_overlap_true else 'NoChange'}")
        controller.logger.info("[%s] %s", controller.cam_id, msg)
        return _ar_success(msg, {
            "mode": "variable",
            "period_node": period_node,
            "requested_us": requested,
            "applied_us": applied,
            "read_back_us": read_back,
            "exposure_readout_overlap": bool(set_overlap_true),
        })

    except GenTLException as ge:
        return _ar_fail(f"CIC 설정 실패(GenTL): {ge}")
    except Exception as exc:
        return _ar_fail(f"CIC 설정 실패: {exc}")




@register_action(
    id="trigger_software_direct",
    display_name="Trigger Software (Direct)",
    category="Camera Control",
    description="Executes the 'TriggerSoftware' command on the active controller.",
)
def execute_trigger_software_direct(controller: CameraController, context: Dict[ContextKey, Any],
                                    **kwargs) -> ActionResult:
    """
    현재 활성 컨트롤러에 대해 직접 'TriggerSoftware' 커맨드를 실행합니다.
    _get_active_controller를 사용하여 foreach 루프를 지원합니다.
    """
    # Foreach 루프 내에서 올바른 컨트롤러를 선택
    active_controller, err = _get_active_controller(controller, context)
    if err:
        return err
    if active_controller is None:
        return _ar_fail("No active controller resolved for trigger_software_direct.")

    try:
        # XML에 명시된 'TriggerSoftware' Command를 직접 실행
        active_controller.execute_command("TriggerSoftware")
        return _ar_success(f"[{active_controller.cam_id}] 'TriggerSoftware' command sent directly.")
    except Exception as e:
        logger.error(f"[{active_controller.cam_id}] Direct software trigger failed: {e}", exc_info=True)
        return _ar_fail(f"Direct software trigger failed: {e}")

@register_action(
    id="device_send_linktrigger0",
    display_name="Send LinkTrigger0 (Device)",
    category="Camera Control",
    description=(
            "Finds a Coaxlink grabber and sends a LinkTrigger0 pulse via its Device-module."
    ),
)
def execute_device_send_linktrigger0(controller: CameraController, context: Dict[ContextKey, Any],
                                     **kwargs) -> ActionResult:
    """
    호스트 명령으로 그래버에서 LinkTrigger0를 발생시키는 액션.
    controller_pool을 순회하여 Device 모듈을 가진 첫 번째 컨트롤러를 찾아 실행합니다.
    """
    from src.core.controller_pool import controllers

    grabber_controller = None
    # controller_pool의 모든 컨트롤러를 확인하여 Device 모듈이 있는 것을 찾음
    for ctrl in controllers.values():
        if getattr(ctrl.grabber, "device", None) is not None:
            grabber_controller = ctrl
            break

    if not grabber_controller:
        return _ar_fail("No grabber controller with a 'device' module found in the pool.")

    try:
        # ① Selector / Source 세팅
        grabber_controller.set_device_param("CxpTriggerMessageSelector", "LinkTrigger0")
        grabber_controller.set_device_param("CxpTriggerMessageSource", "HostCommandRisingEdge")

        # ② 트리거 메시지 전송
        grabber_controller.execute_device_command("CxpTriggerMessageSend", timeout=0.5)

        return _ar_success(f"LinkTrigger0 sent via grabber '{grabber_controller.cam_id}'")

    except Exception as e:
        logger.error(f"Device trigger error on grabber '{getattr(grabber_controller, 'cam_id', 'N/A')}': {e}",
                     exc_info=True)
        return _ar_fail(f"Device trigger error: {e}")
@register_action(
    id="set_trigger_source_auto",
    display_name="Auto Set TriggerSource",
    category="Camera Control",
    description="Pick best TriggerSource automatically (LinkTrigger0 > Line0 > Software).",
)
def execute_set_trigger_source_auto(controller, context, **_):
    cam_nm = getattr(controller, "params", None)
    gbr_nm = getattr(controller.grabber, "remote", None)

    # 1) TriggerSelector = ExposureStart 선행
    for nm in (cam_nm, gbr_nm):
        if nm and "TriggerSelector" in nm.features():
            with suppress(Exception):
                nm.set("TriggerSelector", "ExposureStart")

    preferred = ("LinkTrigger0", "LineTrigger0", "Software")
    for nm in (cam_nm, gbr_nm):
        if not nm or "TriggerSource" not in nm.features():
            continue
        node = nm.getNode("TriggerSource")
        if not node or not node.isWritable():
            continue
        avail = [e.getSymbolic() for e in node.getEnumEntries() if e.isAvailable()]
        for cand in preferred:
            if cand in avail:
                node.setValue(cand)
                return _ar_success(f"TriggerSource = {cand}")
    return _ar_fail("No suitable TriggerSource candidate")


WANTED_TRIGGER_SELECTORS: Tuple[str, ...] = (
    "FrameStart",
    "ExposureStart",
    "AcquisitionStart",
)

# ───────────────────────────────────────────────────────────────
def _auto_select_trigger(
    ctrl,
    wanted: Sequence[str] = WANTED_TRIGGER_SELECTORS   # ← 상수 재사용
) -> bool:
    """
    TriggerSelector 가 있는 카메라에서 *가능* 한 값을 골라줍니다.
    - wanted 시퀀스의 우선순위대로 시도
    - 성공하면 True, 모두 실패하면 False
    """
    nm = getattr(ctrl, "params", None)
    if nm is None or "TriggerSelector" not in nm.features():
        return False                                   # Selector 자체가 없으면 skip
    if "TriggerSelector" in nm.features():
        for cand in wanted:
            try:
                nm.set("TriggerSelector", cand)
                # 선택 직후 TriggerMode 가 write 가능해졌는지 검사
                if ctrl.is_writeable("TriggerMode"):
                    return True
            except Exception:
                continue
    return False

@register_action(
    id="setup_camera_for_hw_trigger",
    display_name="Setup Camera for HW Trigger",
    category="Camera Control",
    description="Safely configures a camera to receive hardware triggers from the grabber (e.g., LinkTrigger0).",
    arguments=[
        ActionArgument("source", "Trigger Source", PARAM_TYPE_CAMERA_PARAM, "e.g., LinkTrigger0", default_value="LinkTrigger0")
    ]
)
def execute_setup_camera_for_hw_trigger(controller: CameraController, context: Dict[ContextKey, Any], **kwargs) -> ActionResult:
    active_controller, err = _get_active_controller(controller, context)
    if err: return err

    source = str(_resolve_context_vars(kwargs.get('source'), context))
    try:
        active_controller.setup_for_hardware_trigger(source=source)
        return _ar_success(f"[{active_controller.cam_id}] HW trigger setup complete for source '{source}'.")
    except Exception as e:
        return _ar_fail(f"[{active_controller.cam_id}] Failed to setup for HW trigger: {e}")


@register_action(
    id="execute_grabber_hw_trigger",
    display_name="Execute Grabber HW Trigger",
    category="Grabber",
    description="[Leader-Only] Sends a single hardware trigger pulse from the grabber.",
)
def execute_grabber_hw_trigger(controller: CameraController, context: Dict[ContextKey, Any], **kwargs) -> ActionResult:
    """
    [수정]
    setup_grabber_for_hw_trigger와 동일하게, 시스템의 첫 번째 컨트롤러(그래버)에서만
    트리거 명령(StartCycle)을 실행하여 중복 트리거 발생을 원천적으로 차단합니다.
    """
    # 현재 실행중인 Runner가 리더(첫번째 카메라)가 아니면 스킵
    all_ids = controller_pool.ids()
    if not all_ids or context.get('cam_id') != all_ids[0]:
        return _ar_success("Skipped (not leader camera).")

    leader_controller = controller_pool.first_controller()
    if not leader_controller:
        return _ar_fail("No leader controller found to send trigger.")

    dev_module = getattr(leader_controller.grabber, 'device', None)
    if not dev_module:
        return _ar_fail(f"[{leader_controller.cam_id}] Grabber has no Device module.")

    try:
        # 'StartCycle' 명령이 있는 경우 실행
        if 'StartCycle' in dev_module.features():
            dev_module.execute('StartCycle')
            return _ar_success(f"HW trigger (StartCycle) sent via '{leader_controller.cam_id}'.")
        else:
            return _ar_fail("'StartCycle' command not found in grabber's device module.")
    except Exception as e:
        return _ar_fail(f"Grabber HW trigger failed: {e}", exc_info=True)
@register_action(
    id="configure_trigger",
    display_name="Configure Trigger",
    category="Camera Control",
    description="Safely sets TriggerMode / TriggerSource / Activation.",
    arguments=[
        ActionArgument("mode", "Trigger Mode",
                       "On / Off value.",
                       PARAM_TYPE_ENUM,
                       options=["On", "Off"],
                       default_value="On"),
        ActionArgument("source", "Trigger Source",
                       "Source enumeration value.",
                       PARAM_TYPE_CAMERA_PARAM,
                       default_value="LinkTrigger0"),
        ActionArgument("activation", "Trigger Activation",
                       "Edge polarity.",
                       PARAM_TYPE_ENUM,
                       options=["RisingEdge", "FallingEdge"],
                       default_value="RisingEdge"),
    ],
)
def execute_configure_trigger(controller: CameraController, context: Dict[ContextKey, Any], **kwargs) -> ActionResult:
    """
    [최종 수정] 컨트롤러의 캡슐화된 메소드를 호출하고, 실패 시 상세한 예외 메시지를 반환합니다.
    """
    active_controller, err = _get_active_controller(controller, context)
    if err:
        return err

    mode = str(_resolve_context_vars(kwargs.get('mode'), context))
    source = str(_resolve_context_vars(kwargs.get('source'), context))
    activation = str(_resolve_context_vars(kwargs.get('activation'), context))

    try:
        # 이 부분이 실제 카메라 하드웨어를 제어하는 호출입니다.
        active_controller.configure_trigger(mode=mode, source=source, activation=activation)

        # 성공 시
        return _ar_success(f"[{active_controller.cam_id}] Trigger configured: Mode={mode}, Source={source}")

    except (ParameterSetError, CommandExecutionError, ParameterError) as e:
        # [핵심 수정] Parameter 관련 예외 발생 시, 예외 메시지(e)를 명확히 포함하여 반환합니다.
        error_message = f"[{active_controller.cam_id}] Trigger configuration parameter error: {e}"
        logger.error(error_message, exc_info=True)
        return _ar_fail(error_message)
    except Exception as e:
        # [핵심 수정] 기타 모든 예외에 대해서도 상세 내용을 포함합니다.
        error_message = f"[{active_controller.cam_id}] Unexpected error during trigger configuration: {e}"
        logger.error(error_message, exc_info=True)
        return _ar_fail(error_message)


@register_action(
    id="run_trigger_integrity_test",
    display_name="Run Trigger Integrity Test",
    category="Diagnostics",                            # ← 수정
    description="[Leader-Only] Runs a synchronized, long-run HW trigger "
                "test on all connected cameras, failing on any frame loss.",
    arguments=[
        ActionArgument("expected_triggers", "Expected Triggers", PARAM_TYPE_INT,
                       "Total number of triggers to test.",
                       default_value=1000, min_value=1),
        ActionArgument("poll_interval_ms", "Poll Interval (ms)", PARAM_TYPE_INT,
                       "Interval for checking status.",
                       default_value=2000, min_value=100),
    ],
)
def execute_run_trigger_integrity_test(controller: CameraController, context: Dict[ContextKey, Any],
                                       **kwargs) -> ActionResult:
    """
    [신규 액션]
    복잡한 트리거 테스트 로직을 캡슐화한 액션입니다.
    리더 카메라만 이 액션의 실제 로직을 실행하며, 다른 카메라들은 리더가 끝날 때까지 대기 상태와 유사하게 성공을 반환합니다.
    """
    all_ids = controller_pool.ids()
    leader_id = all_ids[0] if all_ids else None

    # 이 액션은 리더만 실행합니다.
    if context.get('cam_id') != leader_id:
        return _ar_success(f"Deferring to leader '{leader_id}' for trigger test.")

    leader_controller = controller_pool.first_controller()
    if not leader_controller:
        return _ar_fail("No leader controller to run the test.")

    try:
        expected_triggers = int(_resolve_context_vars(kwargs.get("expected_triggers", 1000), context))
        poll_interval_sec = int(_resolve_context_vars(kwargs.get("poll_interval_ms", 2000), context)) / 1000.0

        # CameraController에 새로 추가될 캡슐화된 테스트 메서드 호출
        leader_controller.run_synchronized_trigger_test(
            expected_triggers=expected_triggers,
            poll_interval=poll_interval_sec
        )
        return _ar_success(f"Trigger integrity test PASSED for {expected_triggers} triggers across all cameras.")

    except Exception as e:
        return _ar_fail(f"Trigger integrity test FAILED: {e}", exc_info=True)
@register_action(
    id="setup_grabber_for_hw_trigger",
    display_name="Setup Grabber for HW Trigger",
    category="Grabber",
    description="[Leader-Only] Maps CycleGenerator→LinkTrigger N and arms CIC. "
                "내부 메소드 실패 시 이 액션에서 직접 Device 노드들 설정.",
    arguments=[
        ActionArgument(
            "line_tool",
            "Line Tool",
            PARAM_TYPE_STRING,
            "Grabber LINx used as source (예: LIN1).",
            default_value="LIN1"
        ),
        ActionArgument(
            "link_trigger",
            "LinkTrigger",
            PARAM_TYPE_STRING,
            "출력할 DeviceLinkTrigger (예: LinkTrigger0).",
            default_value="LinkTrigger0"
        ),
        ActionArgument(
            "edge",
            "Edge",
            PARAM_TYPE_ENUM,
            "DLT/LineTool의 엣지 극성.",
            options=["RisingEdge", "FallingEdge"],
            default_value="RisingEdge"
        ),
        ActionArgument(
            "cycle_trigger_source",
            "Cycle Trigger Source",
            PARAM_TYPE_STRING,
            "Device::CycleTriggerSource 값 (예: StartCycle/Immediate/LIN1…).",
            default_value="StartCycle"
        ),
        ActionArgument(
            "cycle_period_us",
            "Cycle Period (µs)",
            PARAM_TYPE_FLOAT,
            "요청 주기(µs). 하드웨어 Min 이상으로 클램프하여 적용.",
            default_value=20000.0
        ),
    ],
)
def execute_setup_grabber_for_hw_trigger(controller: CameraController,
                                         context: Dict[ContextKey, Any],
                                         **kwargs) -> ActionResult:
    all_ids = controller_pool.ids()
    if not all_ids or context.get("cam_id") != all_ids[0]:
        return _ar_success("Skipped (not leader).")

    leader = controller_pool.first_controller()
    if not leader:
        return _ar_fail("No leader controller present.")

    # ① 우선 내부 캡슐화 메소드 시도 (FW/보드별 분기 포함)
    try:
        ok = leader.enable_internal_grabber_trigger(
            link_trigger=kwargs.get("link_trigger", "LinkTrigger0"),
            min_period_us=float(kwargs.get("cycle_period_us", 20000.0)),
            trigger_activation=kwargs.get("edge", "RisingEdge"),
            cycle_source=kwargs.get("cycle_trigger_source", "StartCycle"),
        )
        if ok:
            return _ar_success(f"[{leader.cam_id}] CIC configured (internal generator).")
    except Exception as e:
        leader.logger.debug("enable_internal_grabber_trigger failed: %s – trying direct device path.", e)

    # ② 내부 메소드가 False/예외일 때, Device 노드를 직접 구성
    try:
        dev = getattr(leader.grabber, "device", None)
        if dev is None:
            return _ar_fail(f"[{leader.cam_id}] Device module unavailable.")

        feats = dev.features()
        # Camera control to RC
        if "CameraControlMethod" in feats:
            dev.set("CameraControlMethod", "RC")

        # DLT 매핑(옵션)
        if "DeviceLinkTriggerToolSelector" in feats:
            dev.set("DeviceLinkTriggerToolSelector", kwargs.get("link_trigger", "LinkTrigger0"))
        if "DeviceLinkTriggerToolSource" in feats:
            # 내부 생성기 사용
            dev.set("DeviceLinkTriggerToolSource", "CycleGenerator0")
        if "DeviceLinkTriggerToolActivation" in feats:
            dev.set("DeviceLinkTriggerToolActivation", kwargs.get("edge", "RisingEdge"))

        # Source + Period
        if "CycleTriggerSource" in feats:
            dev.set("CycleTriggerSource", kwargs.get("cycle_trigger_source", "StartCycle"))

        period_node = None
        for cand in ("CycleTriggerPeriod", "CyclePeriodUs", "CycleTargetPeriod", "CycleMinimumPeriod"):
            if cand in feats:
                period_node = cand
                break

        requested = float(kwargs.get("cycle_period_us", 20000.0))
        applied = requested
        if period_node:
            try:
                from egrabber import query
                q_min = query.info(period_node, "Min")
                min_limit = dev.get(q_min, float) if q_min else None
                if isinstance(min_limit, (int, float)):
                    applied = max(requested, float(min_limit))
            except Exception:
                pass
            dev.set(period_node, applied)
            read_back = float(dev.get(period_node))
            return _ar_success(f"[{leader.cam_id}] CIC configured via device. set={applied:.1f}µs, read={read_back:.1f}µs ({period_node})")
        else:
            # 주기 노드가 정말 없으면 fixed-mode
            min_cur = float(dev.get("CycleMinimumPeriod")) if "CycleMinimumPeriod" in feats else 0.0
            return _ar_success(f"[{leader.cam_id}] CIC fixed-mode. applied={min_cur:.1f}µs (CycleMinimumPeriod).")

    except Exception as e2:
        return _ar_fail(f"[{leader.cam_id}] Grabber HW-trigger setup failed: {e2}", exc_info=True)


@register_action(
    id="grab_frames",
    display_name="Grab Frames",
    category="Image",
    description=(
            "Grabs *frame_count* frames and displays them on the Live View in real-time. "
            "Assumes 'start_grab' has been called previously."
    ),
    arguments=[
        ActionArgument("frame_count", "Frame Count", PARAM_TYPE_INT,
                       "Number of frames to grab.",
                       default_value=1, min_value=1, max_value=10_000, step=1),
        ActionArgument("timeout_ms", "Timeout (ms)", PARAM_TYPE_INT,
                       "Per-frame timeout.",
                       default_value=1000, min_value=10, max_value=60_000, step=10),
        ActionArgument("save_numpy", "Save raw *.npy?", PARAM_TYPE_BOOL,
                       "Save grabbed frames to logs/raw/*.npy",
                       default_value=False),
    ]
)
def execute_grab_frames(
        controller: "CameraController",
        context: Dict[ContextKey, Any],
        *,
        frame_count: int = 1,
        timeout_ms: int = 1_000,
        runner: Optional["SequenceRunner"] = None,
        **__
) -> ActionResult:
    """
    [최종 안정화 버전]
    지정한 수의 프레임을 획득하며, 획득한 각 프레임을 실시간으로 Live View에 표시합니다.
    """
    t0 = time.monotonic()

    active, err = _get_active_controller(controller, context)
    if err: return err
    if active is None: return _ar_fail("No active controller resolved for grab_frames.")
    if not active.is_grabbing(): return _ar_fail(f"[{active.cam_id}] Cannot grab frames: grabber is not active.")

    last_frame: Optional[np.ndarray] = None
    grabbed_count = 0

    try:
        for i in range(frame_count):
            if runner and runner._stop_requested:
                logger.warning(f"Grab frames interrupted by stop request after {grabbed_count} frames.")
                break

            frame = active.get_next_frame(timeout_ms=timeout_ms)
            if frame is None:
                raise FrameTimeoutError(f"Timeout while grabbing frame {i + 1}/{frame_count}")

            last_frame = frame
            grabbed_count += 1

            if runner and hasattr(runner, 'test_frame_grabbed'):
                runner.test_frame_grabbed.emit(active.cam_id, frame)
                time.sleep(0.001)

        if last_frame is not None:
            context['frame'] = last_frame.copy()

        result_meta = {
            "frames_grabbed": grabbed_count,
            "total_requested": frame_count,
            "execution_time_ms": round((time.monotonic() - t0) * 1000),
        }
        message = (f"Grab interrupted after {grabbed_count} frames." if runner and runner._stop_requested
                   else f"Successfully grabbed {grabbed_count} of {frame_count} frames.")
        return _ar_success(message, result_meta)

    except Exception as exc:
        logger.error(f"grab_frames failed after acquiring {grabbed_count} frames: {exc}", exc_info=True)
        if last_frame is not None:
            context['frame'] = last_frame.copy()
        return _ar_fail(f"grab_frames error after {grabbed_count} frames: {exc}")

_RP_KEY_PREFIX = "_repeat_"

def _rp_key(block_id: str, suffix: str) -> str:
    """컨텍스트용 키 조합 헬퍼"""
    return f"{_RP_KEY_PREFIX}{block_id}_{suffix}"

@register_action(
    id="repeat_block_start",
    display_name="Repeat Block (Start)",
    category="Flow",
    description=(
            "Repeat Block 진입점.\n"
            "• 루프 상태를 초기화하고, 선택적으로 시작 로그를 출력합니다."
    ),
    arguments=[
        ActionArgument("mode", "Mode", PARAM_TYPE_ENUM, "반복 방식", options=["count", "condition", "list"],
                       default_value="count"),
        ActionArgument("count", "Repeat Count", PARAM_TYPE_INT, "mode=count 일 때 반복 횟수", default_value=1, min_value=1),
        ActionArgument("condition_expr", "Condition Expr", PARAM_TYPE_STRING, "mode=condition일 때 평가할 표현식", required=False),
        ActionArgument("list_key", "List Key", PARAM_TYPE_CONTEXT_KEY, "mode=list일 때 순회할 리스트가 담긴 컨텍스트 키", default_value="items", required=False),
        ActionArgument("index_key", "Index Key", PARAM_TYPE_CONTEXT_KEY_OUTPUT, "현재 반복 횟수를 저장할 사용자 정의 컨텍스트 키.",
                       required=False),
        ActionArgument("log_on_start", "Log on Start?", PARAM_TYPE_BOOL, "True일 경우, 시작 시 로그 메시지를 출력합니다.",
                       default_value=False, required=False),
        ActionArgument("log_message", "Log Message", PARAM_TYPE_STRING, "출력할 로그 메시지 템플릿. {loop_index} 등을 사용할 수 있습니다.",
                       default_value="--- Loop Start: Iteration #{loop_index} ---", required=False),
        ActionArgument("block_id", "Block ID", PARAM_TYPE_STRING, "동일 Flow 내에서 유일해야 함 (자동 생성 가능)", required=False),
    ]
)
def execute_repeat_block_start(controller, context, **kw):
    """
    지정한 횟수만큼 반복하는 루프 블록의 진입점입니다. (최종 단순화 버전)
    """
    bid = kw.get("block_id") or str(uuid4())
    user_index_key = kw.get("index_key")

    context[_rp_key(bid, "index")] = 0
    context[_rp_key(bid, "bid")] = bid

    total = int(_resolve_context_vars(kw.get("count", 1), context))
    context[_rp_key(bid, "total")] = total

    # 선택적 index_key가 제공된 경우에만 컨텍스트에 저장
    if user_index_key:
        context[user_index_key] = 0
        context[_rp_key(bid, "user_index_key")] = user_index_key

    success_message = f"Repeat-Block[{bid}] initialised for {total} iterations."
    return _ar_success(success_message, {"block_id": bid})


@register_action(
    id="repeat_block_end",
    display_name="Repeat Block (End)",
    category="Flow",
    description=(
            "Repeat-Block 종료점.\n"
            "내부 카운터를 증가시키고, 조건을 검사하여 loop_continue 또는 loop_exit 을 반환합니다."
    ),
    arguments=[
        ActionArgument("block_id", "Block ID", PARAM_TYPE_STRING,
                       "해당 루프의 시작 블록(repeat_block_start)과 동일한 ID여야 합니다.", required=True),
        # [수정] 혼란을 주던 loop_start_label 인자 완전 삭제
        ActionArgument("current_item_key", "Current Item Key",
                       PARAM_TYPE_CONTEXT_KEY_OUTPUT,
                       "mode=list 일 때 현재 아이템을 저장할 키",
                       default_value="current_item"),
    ]
)
def execute_repeat_block_end(controller, context, **kw) -> StepActionResult:
    """
    루프의 끝을 처리합니다. 카운터를 증가시키고, 루프 지속/종료를 결정하며, 진행 상황을 보고합니다.
    """
    bid = kw.get("block_id")
    if not bid:
        return StepActionResult(status="error", message="Repeat Block End requires a 'block_id'.")

    internal_index_key = _rp_key(bid, "index")
    total_key = _rp_key(bid, "total")

    if total_key not in context:
        return StepActionResult(status="error",
                                message=f"Repeat-Block[{bid}] not initialised. Ensure a 'Repeat Block (Start)' action with the same block_id comes first.")

    current_index = int(context.get(internal_index_key, 0))
    total = int(context.get(total_key, 0))
    next_index = current_index + 1

    loop_continue = next_index < total

    if loop_continue:
        context[internal_index_key] = next_index
        user_index_key = context.get(_rp_key(bid, "user_index_key"))
        if user_index_key:
            context[user_index_key] = next_index

        message = f"Continuing loop '{bid}'. Iteration {next_index + 1} of {total}."
        return StepActionResult(status="loop_continue", block_id=bid, message=message)
    else:
        # 루프 관련 컨텍스트 변수 정리
        user_index_key = context.get(_rp_key(bid, "user_index_key"))
        if user_index_key:
            context.pop(user_index_key, None)
        for suffix in ["index", "total", "user_index_key", "bid"]:
            context.pop(_rp_key(bid, suffix), None)

        message = f"Finished loop '{bid}' after {total} iterations."
        return StepActionResult(status="loop_exit", block_id=bid, message=message)

def _realloc_grabber_buffers(grabber, *, height: int, count: Optional[int] = None) -> None:
    # count가 주어지면 사용, 없으면 현재 announced 개수 유지 시도
    try:
        get_cnt = getattr(grabber, "get_announced_buffer_count",
                          getattr(grabber, "announced_buffer_count", None))
        cur = int(get_cnt()) if callable(get_cnt) else 0
    except Exception:
        cur = 0
    buf_cnt = int(count or cur or 64)   # 기본 32
    grabber.realloc_buffers(buf_cnt, height)



def _safe_reconfig_linescan(ctrl: CameraController,
                            scan_len: int, buf_height: int):
    if buf_height <= 0:
        buf_height = scan_len

    g = ctrl.grabber
    was_grabbing = getattr(ctrl, "is_grabbing", lambda: False)()
    if was_grabbing:
        ctrl.stop_grab()

        # --- [수정된 부분 시작] ---
        # 수집 중지 후 파라미터 변경 시 발생할 수 있는 경쟁 상태를 방지하기 위해
        # 스트림이 유휴 상태가 될 때까지 최대 0.5초간 대기합니다.
        ctrl.wait_until_idle(timeout=0.5)
        # --- [수정된 부분 끝] ---

    # Remote / Stream 분리
    g.stream.set("ScanLength", scan_len)
    g.stream.set("BufferHeight", buf_height)

    # 버퍼 재할당
    _realloc_grabber_buffers(g, height=buf_height)

    if was_grabbing:
        ctrl.start_grab()

    return scan_len, buf_height


@register_action(
    id="configure_linescan_dimensions",
    display_name="Configure Linescan Dimensions",
    category="Grabber",
    description="Sets ScanLength & BufferHeight **safely**. Call Start-Grab afterwards.",
    arguments=[
        ActionArgument("scan_length", "Scan Length", PARAM_TYPE_INT,
                       "Logical image height (lines).",
                       required=True, default_value=1024),
        ActionArgument("buffer_height", "Buffer Height", PARAM_TYPE_INT,
                       "Physical buffer height (0 → same as ScanLength).",
                       required=False, default_value=1024),
    ]
)
def execute_configure_linescan_dimensions(controller: CameraController,
                                          context: Dict[ContextKey, Any],
                                          **kwargs) -> ActionResult:
    active_ctrl, err = _get_active_controller(controller, context)
    if err:
        return err

    try:
        scan_length = int(_resolve_context_vars(kwargs["scan_length"], context))
        buffer_height = int(_resolve_context_vars(kwargs.get("buffer_height", 0), context))
        if buffer_height <= 0:
            buffer_height = scan_length

        g = active_ctrl.grabber
        if not g:
            return _ar_fail("Grabber is not initialized. Cannot configure dimensions.")

        # 1) 완전 정지 + 유휴 대기
        if active_ctrl.is_grabbing():
            active_ctrl.logger.info(f"[{active_ctrl.cam_id}] Stopping grab to reconfigure linescan dimensions.")
            active_ctrl.stop_grab(flush=True)
            active_ctrl.wait_until_idle(timeout=1.5)

        # 2) 스트림 파라미터 설정
        stream_module = getattr(g, "stream", None)
        if not stream_module:
            return _ar_fail("Required GenTL module 'stream' is not available.")

        active_ctrl.logger.debug(f"Setting ScanLength to {scan_length} on stream module.")
        stream_module.set("ScanLength", scan_length)

        active_ctrl.logger.debug(f"Setting BufferHeight to {buffer_height} on stream module.")
        stream_module.set("BufferHeight", buffer_height)

        # 3) 안전 재할당 (직접 realloc 금지)
        target_cnt = int(getattr(active_ctrl, "_buf_target", 64))
        active_ctrl.logger.info(f"[{active_ctrl.cam_id}] Safe realloc buffers: {target_cnt} (auto-size)")
        active_ctrl._realloc_buffers_safe(target_cnt, 0, refill=True)

        msg = (f"Linescan dimensions configured: ScanLength={scan_length}, BufferHeight={buffer_height}. "
               "Run 'Start Grab' to apply.")
        active_ctrl.logger.info(f"[{active_ctrl.cam_id}] {msg}")
        return _ar_success(msg, {"scan_length": scan_length, "buffer_height": buffer_height})

    except Exception as exc:
        return _ar_fail(f"configure_linescan_dimensions encountered an unexpected error: {exc}", exc_info=True)

