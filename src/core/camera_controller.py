#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import logging, threading, time, subprocess, os, json
from typing import Optional, List, Dict, Any, Tuple, Literal
import numpy as np
import math
from contextlib import contextmanager
from datetime import datetime
from contextlib import suppress
from src.core import controller_pool
# ──────────────────────────────────────────────
#  1) eGrabber (실제 / 더미)  +  싱글턴
# ──────────────────────────────────────────────
try:
    from egrabber import (
        EGenTL, EGrabber, Buffer,
        BUFFER_INFO_BASE, BUFFER_INFO_WIDTH,
        BUFFER_INFO_DELIVERED_IMAGEHEIGHT, BUFFER_INFO_DATA_SIZE,
        INFO_DATATYPE_PTR, INFO_DATATYPE_SIZET,
        TimeoutException, GenTLException,
        query, Coaxlink
    )
    EURESYS_AVAILABLE = True

    # ──────────────────────────────────────────────
    #  글로벌 EGenTL 싱글턴 (안정판)
    # ──────────────────────────────────────────────
    _gentl_singleton: Optional[EGenTL] = None

    def _new_gentl() -> EGenTL:
        """CTI 경로가 깨져도 최소한 객체는 리턴하도록 2단 시도."""
        try:
            return EGenTL(Coaxlink())  # 표준
        except Exception:
            return EGenTL()  # fallback


    def _gentl_is_healthy(g) -> bool:
        """
        빌드마다 존재하는 '가벼운' API 한 가지라도 호출에 성공하면 정상으로 본다.
        호출 자체가 없는 경우도 *정상* 취급 (AttributeError 허용).
        """
        try:
            for probe in (
                    lambda: getattr(g, "get_num_interfaces")() if hasattr(g, "get_num_interfaces") else None,
                    lambda: len(g.interfaces()) if callable(getattr(g, "interfaces", None)) else None,
                    lambda: len(g.interfaces) if hasattr(g, "interfaces") else None,
                    lambda: getattr(g, "interface_count")() if callable(getattr(g, "interface_count", None)) else None,
                    lambda: getattr(g, "version")() if callable(getattr(g, "version", None)) else None,
            ):
                try:
                    probe();
                    return True
                except AttributeError:
                    continue  # 메서드가 없으면 다른 probe 시도
            return True  # 모든 probe 가 AttributeError → 기능은 없지만 죽진 않음
        except GenTLException as e:
            return e.gc_err == 0  # 0==SUCCESS → 정상, 그 외 == 오류
        except Exception:
            return False


    def _get_gentl() -> EGenTL:
        """깨졌으면 폐기 후 새로 만든다."""
        global _gentl_singleton
        if _gentl_singleton is None or not _gentl_is_healthy(_gentl_singleton):
            try:
                # 가능하면 기존 객체 정리 (3.x 에서만 close() 존재)
                getattr(_gentl_singleton, "close", lambda: None)()
            except Exception:
                pass
            _gentl_singleton = _new_gentl()
        return _gentl_singleton


except ImportError:  # ───── 시뮬레이션 모드 ─────
    EURESYS_AVAILABLE = False

    class EGenTL:                         # 더미
        def get_num_interfaces(self): return 0
        def get_interface_id(self, idx): raise IndexError
        @contextmanager
        def open_interface(self, iface_id): yield DummyInterface()

    class EGrabber:  pass
    class Buffer:    ...                  # 생략 (더미)
    class TimeoutException(Exception): pass
    class GenTLException(Exception):   pass
    query = None

    _get_gentl = lambda: EGenTL()         # 중복 생성 OK

class DummyInterface:
    def get_num_devices(self): return 0
    def get_device_id(self, idx): raise IndexError

from src.core.camera_exceptions import (
    CameraError, CameraConnectionError, CameraNotConnectedError, GrabberError,
    GrabberNotActiveError, FrameAcquisitionError, FrameTimeoutError,
    GrabberStartError, GrabberStopError, ParameterError, ParameterSetError,
    ParameterGetError, CommandExecutionError
)

logger = logging.getLogger(__name__)

from PyQt5.QtCore import QObject, QThread, pyqtSignal, QTimer
LEGACY_MAP: Dict[str, List[Tuple[str, Any]]] = {}

import math

def _align_up(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a
# ────────────────────────────────────────────────────────────────────
#  Memento CLI 경로 탐색
# ────────────────────────────────────────────────────────────────────
def _find_memento_cli() -> Optional[str]:
    exe = "memento.exe" if os.name == "nt" else "memento"
    env = os.environ.get("EURESYS_MEMENTO_PATH")
    if env and os.path.isfile(env):
        return env
    for p in os.environ.get("PATH", "").split(os.pathsep):
        f = os.path.join(p, exe)
        if os.path.isfile(f):
            return f
    for f in [
        r"C:\Program Files\Euresys\Memento\bin\x86_32\memento.exe",
        r"C:\Program Files\Euresys\Memento\bin\memento.exe",
        r"C:\Program Files\Euresys\eGrabber\bin\memento.exe",
    ]:
        if os.path.isfile(f):
            return f
    return None

def _run_memento_cmd(args: List[str]) -> None:
    cli = _find_memento_cli()
    if not cli:
        raise FileNotFoundError("Memento CLI not found")
    subprocess.check_call([cli, *args],
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)
# ──────────────────────────────────────────────────────
#  (1) 디버그용 – 실제 EGenTL 객체가 어떤 API를 갖고 있는지 확인
# ──────────────────────────────────────────────────────
def debug_print_gentl_dir():
    """
    현재 설치된 eGrabber Python-API 가 노출하는 속성·메서드를 한눈에 출력.
    - main 스크립트나 IPython에서 임시로 호출해 보세요.
    """
    try:
        g = _get_gentl()
        print("EGenTL dir =", dir(g))
    except Exception as e:
        print("EGenTL dir() 확인 실패:", e)
# ──────────────────────────────────────────────────────
#  (2) 인터페이스 열거 – API 버전별로 안전하게 대응
# ──────────────────────────────────────────────────────
def _iter_gentl_interfaces(gentl):
    """
    yield (iface_index:int, iface_obj)  ← with-context 로 이미 감싸진 상태
    대응 가능한 API 패턴
    ─────────────────────────────────────────────────────────────
      ① get_num_interfaces / get_interface_id   (eGrabber 1.x~2.x)
      ② interfaces() 메서드                      (3.0 ~ 3.1.x)
      ③ interfaces  프로퍼티(list-like)          (3.2+)
      ④ enumerate_interfaces() 메서드            (드문 파생 build)
      ⑤ interface_ids()  메서드                 (또 다른 파생 build)
    """
    # ① legacy API: get_num_interfaces / get_interface_id
    if hasattr(gentl, "get_num_interfaces"):
        for i in range(gentl.get_num_interfaces()):
            iface_id = gentl.get_interface_id(i)
            with gentl.open_interface(iface_id) as iface:
                yield i, iface
        return

    # ② 3.0 – 3.1 : interfaces() 메서드
    if callable(getattr(gentl, "interfaces", None)):
        for i, iface in enumerate(gentl.interfaces()):       # list[Interface]
            with iface:
                yield i, iface
        return

    # ③ 3.2+ : interfaces 프로퍼티 (list-like)
    if hasattr(gentl, "interfaces"):
        for i, iface in enumerate(gentl.interfaces):         # property
            with iface:
                yield i, iface
        return

    # ④ enumerate_interfaces()  → interface ID 시퀀스
    if callable(getattr(gentl, "enumerate_interfaces", None)):
        for i, iface_id in enumerate(gentl.enumerate_interfaces()):
            with gentl.open_interface(iface_id) as iface:
                yield i, iface
        return

    # ⑤ interface_ids()  → interface ID 시퀀스
    if callable(getattr(gentl, "interface_ids", None)):
        for i, iface_id in enumerate(gentl.interface_ids()):
            with gentl.open_interface(iface_id) as iface:
                yield i, iface
        return

    # ⑥ 마지막 수단: EGenTL 객체 자체가 iterable 한 경우
    try:
        for i, iface in enumerate(gentl):
            with iface:
                yield i, iface
        return
    except TypeError:
        pass

    # 전부 실패 시 명시적 예외
    raise AttributeError("EGenTL API: 인터페이스 열거 메서드를 찾을 수 없습니다.")
# ------------------------------------------------------------------
#  안전하게 벤더/모델/시리얼을 읽어오는 헬퍼
# ------------------------------------------------------------------
def _safe_get_dev_info(grabber, key: str) -> str:
    """
    - eGrabber 1.x/2.x : grabber.get_info(key)
    - Coaxlink 파이썬 API : grabber.remote.get(key)
    - 모두 실패        : 'Unknown'
    """
    # ① 표준 get_info()
    if hasattr(grabber, "get_info"):
        try:
            val = grabber.get_info(key)
            if val:
                return str(val)
        except Exception:
            pass

    # ② GenICam remote 노드
    try:
        if hasattr(grabber, "remote") and grabber.remote:
            return str(grabber.remote.get(key))
    except Exception:
        pass

    return "Unknown"

def _probe_attr(obj, *candidates, default="Unknown"):
    """obj 에서 candidates 중 처음으로 존재하는 속성/메서드를 호출/읽어 반환."""
    for name in candidates:
        if hasattr(obj, name):
            attr = getattr(obj, name)
            try:
                return attr() if callable(attr) else attr
            except Exception:
                pass
    return default

# ───────────── Force-WO (read forbidden but writable) candidates ─────────────
# 일부 장비/빌드에서 AccessMode/Readable 플래그가 틀리게 노출되는 노드들.
# 여기 지정하면: 읽기 실패 시 WO로 강제 판정하고, 필요한 경우 enum 엔트리도 제공.
FORCE_WO_ENUMS: dict[str, list[str]] = {
    "BalanceWhiteAuto": ["Off", "Once", "Continuous"],
    "BalanceRatioSelector": ["Red", "Green", "Blue"],
}
FORCE_WO_NUMS: set[str] = set()  # 수치형 WO 후보 있으면 이름 추가

# ---------------------------------------------------------------------
#  helper: grabber 가 가진 CameraPort 개수 추정
# ---------------------------------------------------------------------
def _num_camera_ports(grabber) -> int:
    """
    Coaxlink 계열 보드에서 사용 가능한 Camera Port 개수를 반환한다.
    드라이버/펌웨어 버전에 따라 노드 이름이 조금씩 달라질 수 있으므로
    여러 후보를 순차적으로 시도한다.
    """
    for key in ("NumCameraPorts", "CameraPortCount", "CameraPorts"):
        try:
            return int(grabber.remote.get(key))
        except Exception:
            pass
    return 1      # 알 수 없으면 1개라고 가정
# ----------------------------------------------------------------------
#  Grabber 1개 안에 몇 개의 Visible-Device(Port)가 존재하는지 조사
# ----------------------------------------------------------------------
def _iter_visible_devices(grabber):
    """
    yield (port_idx:int, model:str, serial:str, vendor:str)
    - VisibleDeviceSelector   : Coaxlink 3.x+
    - PortSelector / PortId   : 구 버전/다른 Producer
    """
    remote = getattr(grabber, "remote", None)
    if remote is None:
        return

    # ① Coaxlink 3.x : VisibleDeviceSelector (Enumeration)
    if "VisibleDeviceSelector" in remote.features():
        enum = remote.getEnum("VisibleDeviceSelector")   # ← egrabber.query.enum_entries(...)
        for idx, entry in enumerate(enum):
            remote.set("VisibleDeviceSelector", entry)
            yield idx, remote.get("DeviceModelName", str), \
                       remote.get("DeviceSerialNumber", str), \
                       remote.get("DeviceVendorName", str)
    # ② Fallback : PortSelector / PortId
    elif "PortSelector" in remote.features():
        max_ports = int(remote.get("PortCount"))
        for idx in range(max_ports):
            remote.set("PortSelector", idx)
            yield idx, remote.get("DeviceModelName", str), \
                       remote.get("DeviceSerialNumber", str), \
                       remote.get("DeviceVendorName", str)


#  CameraController 내부 ─────────────────────────────────────────────
from egrabber.discovery import EGrabberDiscovery, EGrabberCameraInfo, EGrabberInfo
from egrabber.generated.constants import DEVICE_ACCESS_READONLY as RO
import egrabber.generated.constants as C
OFFLINE_CODES = {
    C.GC_ERR_NOT_AVAILABLE,
    C.GC_ERR_INVALID_ADDRESS,
    C.GC_ERR_CUSTOM_DRIVER_NOT_AVAILABLE,
    C.GC_ERR_ERROR,  # == -1
}


def _is_online(cam_info: EGrabberCameraInfo) -> bool:
    """RO 플래그로 ‘살짝’ 열어 본다."""
    for flags in (C.DEVICE_ACCESS_CONTROL, RO):
        for remote in (True, False):
            try:
                with EGrabber(cam_info, device_open_flags=flags,
                              remote_required=remote):
                    return True
            except GenTLException as ge:
                if getattr(ge, "gc_err", None) in OFFLINE_CODES:
                    continue
            except Exception:
                # 파이썬 레벨 에러(속성 누락 등)는 무시
                pass
    return False


def _enum_entries_safe(dev, node: str) -> List[str]:
    try:
        if hasattr(dev, "get_enum_entries"):
            return list(dev.get_enum_entries(node))
        if hasattr(dev, "getEnum"):
            return list(dev.getEnum(node))
        if hasattr(dev, "getNode"):
            n = dev.getNode(node)
            if n:
                return [e.getSymbolic() for e in n.getEnumEntries() if e.isAvailable()]
    except Exception:
        pass
    return []

from src.ui.grab_worker import GrabWorker
_grab_worker: Optional[GrabWorker] = None
DEFAULT_STATS_INTERVAL = 0.5

def read_lost_triggers_legacy(dev) -> int:
    """
    FW < 4.38 보드에서 Trigger 누락을 추산하기 위한 에러 카운터 합산.
    ErrorSelector : {DidNotReceiveTriggerAck, CameraTriggerOverrun}
    """
    lost = 0
    if {"ErrorSelector", "ErrorCount"}.issubset(dev.features()):
        for sel in ("DidNotReceiveTriggerAck", "CameraTriggerOverrun"):
            try:
                dev.set("ErrorSelector", sel)
                lost += int(dev.get("ErrorCount"))
            except Exception:
                continue
    return lost

# ────────── Stream-Port Error 카운터 기반 Frame-Loss ──────────
NEW_ERROR_ENUMS = (
    "StreamPacketSizeError", "StreamPacketFifoOverflow",
    "CameraTriggerOverrun", "DidNotReceiveTriggerAck",
    "TriggerPacketRetryError", "InputStreamFifoHalfFull",
    "InputStreamFifoFull", "ImageHeaderError",
    "MigAxiWriteError", "MigAxiReadError",
    "PacketWithUnexpectedTag", "StreamPacketArbiterError",
    "StartOfScanSkipped", "PrematureEndOfScan",
    "ExternalTriggerReqsTooClose",
    "StreamPacketCrcError0", "StreamPacketCrcError1",
    "StreamPacketCrcError2", "StreamPacketCrcError3",
    "StreamPacketCrcError4", "StreamPacketCrcError5",
    "StreamPacketCrcError6", "StreamPacketCrcError7",
)
_DMA_BLOCK_BYTES = 1024
# ────────── Stream-Port Error 카운터 기반 Frame-Loss (개선판) ──────────
def read_frame_loss_stream(dev) -> int:
    """
    StreamPort-Errors 섹션에서 ErrorSelector/E​rrorCount를
    **모든 enum 엔트리**에 대해 누적해 반환한다.
    ErrorSelector 또는 ErrorCount 노드가 없으면 0.
    """
    lost = 0
    if {"ErrorSelector", "ErrorCount"}.issubset(dev.features()):
        # ▶︎ ErrorSelector 의 전체 enum 항목을 런타임에 확보
        for sel in _enum_entries_safe(dev, "ErrorSelector"):
            try:
                dev.set("ErrorSelector", sel)
                lost += int(dev.get("ErrorCount"))
            except Exception:
                continue     # 읽기 불가·권한 없음 등은 무시
    return lost



class CameraController(QObject):
    """
    개별 카메라 하드웨어를 제어하고 상태를 관리하는 핵심 클래스.
    """

    camera_connected    = pyqtSignal()
    camera_disconnected = pyqtSignal()
    frame_ready         = pyqtSignal(np.ndarray)
    reshaped_frame_ready = pyqtSignal(np.ndarray)

    # ──────────────────── ❶ __init__ 전체 (NEW 필드 포함) ────────────────────
    def __init__(self, enable_param_cache: bool = False):
        QObject.__init__(self)
        self.logger = logging.getLogger(f"{__name__}.{id(self)}")

        # low-level handles & flags
        self.grabber: Optional[EGrabber] = None
        self._device: Optional[Any] = None  # ★ NEW – 내부 Device-module
        self.params: Optional[Any] = None
        self.connected: bool = False
        self.grabbing: bool = False
        self._lock = threading.RLock()
        self._last_buffer: Optional[Any] = None
        self._last_np_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        # identifiers
        self.cam_id: str = f"cam_{id(self)}"
        self.serial_number: str = "N/A"
        self.grabber_index: int = -1

        # SDK helpers
        self.gentl = _get_gentl()

        # NEW – last grabbed buffer cache
        self._last_buffer: Optional[Any] = None

        # parameter-cache option
        self.enable_param_cache = enable_param_cache
        self._param_cache: Dict[str, Any] = {}
        self._param_cache_timestamp: Dict[str, float] = {}
        self.cache_timeout_sec = 1.0

        # runtime stats
        self._stats_interval: float = DEFAULT_STATS_INTERVAL
        now = time.time()
        self.stats = {
            "frame_count": 0,
            "error_count": 0,
            "last_frame_time": now,
            "start_time": 0.0,
            "fps": 0.0,
            "frame_loss": 0.0,
        }
        self._last_stats_update = now

        # misc flags / workers
        self._memento_live: bool = False
        self._grab_thr: Optional["_GrabThread"] = None
        self._live_view: bool = False
        self._quiet_until: float = 0.0  # epoch seconds

        self._bp_wd_stop = threading.Event()
        self._bp_recover = threading.Lock()
        self._bp_last_fix = 0.0

    def begin_quiet(self, ms: int, *, pause_acquisition: bool = False, pause_trigger: bool = False) -> None:
        with self._lock:
            self._quiet_until = time.time() + max(0, int(ms)) / 1000.0
            self._quiet_pause = {"acq": False, "trig": False}

            if pause_acquisition and self.is_grabbing():
                with suppress(Exception):
                    self.execute_command("AcquisitionStop", timeout=0.3)
                self._quiet_pause["acq"] = True

            if pause_trigger and self.params and "TriggerMode" in self.params.features():
                try:
                    if str(self.params.get("TriggerMode")) != "Off":
                        self.params.set("TriggerMode", "Off")
                        self._quiet_pause["trig"] = True
                except Exception:
                    pass

    def _maybe_leave_quiet(self) -> None:
        if self._is_quiet():
            return
        qp = getattr(self, "_quiet_pause", None)
        if not qp:
            return
        # 트리거 복귀
        if qp.get("trig"):
            with suppress(Exception):
                if self.params and "TriggerMode" in self.params.features():
                    self.params.set("TriggerMode", "On")
        # 획득 복귀
        if qp.get("acq") and not self.is_grabbing():
            with suppress(Exception):
                self.grabber.start()
                self.grabbing = True
        self._quiet_pause = None

    def _is_quiet(self) -> bool:
        """
        Quiet 윈도우인지 여부.
        """
        return time.time() < getattr(self, "_quiet_until", 0.0)

    def set_last_buffer(self, buffer: Any) -> None:
        """
        [스레드 안전] 실행 중 가장 마지막에 획득한 **eg.Buffer** 객체를 저장합니다.
        주로 예외 발생 시 오류 이미지 저장을 위해 사용됩니다.
        """
        with self._frame_lock:
            self._last_buffer = buffer


    def get_last_buffer(self) -> Optional[Any]:
        """
        [스레드 안전] `set_last_buffer()`로 저장된 **eg.Buffer** 를 반환합니다.
        """
        with self._frame_lock:
            return self._last_buffer

    def set_last_np_frame(self, frame: np.ndarray) -> None:
        """
        [스레드 안전] 가장 최근에 성공적으로 디코딩된 NumPy 프레임을 캐시에 저장합니다.
        GrabWorker가 이 메서드를 호출합니다.
        """
        with self._frame_lock:
            self._last_np_frame = frame

    def get_last_np_frame(self) -> Optional[np.ndarray]:
        """
        [스레드 안전] 캐시된 최신 NumPy 프레임의 복사본을 반환합니다.
        액션(get_last_cached_frame 등)이 이 메서드를 호출합니다.
        """
        with self._frame_lock:
            return self._last_np_frame.copy() if self._last_np_frame is not None else None

    def start_live_view(self, *, buffer_count: int = 64, max_fps: int = 30) -> None:
        """
        Enable real-time preview via GrabWorker.
        소비자(GrabWorker)를 먼저 준비하고, 그 다음 충분한 버퍼로 그래빙을 시작한다.
        """
        if not self.connected:
            self.logger.warning("[%s] Cannot start live view: not connected.", self.cam_id)
            return
        if self._live_view:
            self.logger.debug("[%s] Live view is already active.", self.cam_id)
            return

        self._live_view = True

        # 1) 소비자(GrabWorker) 먼저
        self._ensure_grab_worker(max_fps=max_fps)

        # 2) 충분한 버퍼로 그래빙 시작 (최소 64)
        if not self.is_grabbing():
            self.start_grab(buffer_count=max(buffer_count, 64), min_buffers=64)

        self.logger.info("[%s] Live-view session started.", self.cam_id)

    def stop_live_view(self) -> None:
        """
        [최종 안정화 버전] 프리뷰를 비활성화하고, 하드웨어가 완전히 멈출 때까지 기다려
        경쟁 상태를 방지하며 안정적으로 수집을 종료합니다.
        """
        if not self._live_view:
            return

        with self._lock:
            self._live_view = False

            # GrabWorker를 먼저 분리하여 추가적인 프레임 요청을 막습니다.
            self._detach_from_grab_worker()

            # 그래빙 중인 경우에만 중지 로직을 실행합니다.
            if self.is_grabbing():
                # stop_grab은 중지 '명령'을 보내는 역할을 합니다.
                self.stop_grab(flush=True)

                # ★★★ [핵심 수정] 하드웨어가 실제로 멈출 때까지 여기서 대기합니다. ★★★
                # 이 코드를 통해, 이 함수가 반환될 때 카메라는 확실히 유휴 상태임이 보장됩니다.
                try:
                    self.wait_until_idle(timeout=1.0)  # 최대 1초 대기
                except Exception as e:
                    self.logger.warning(f"[{self.cam_id}] Exception while waiting for idle state: {e}")

        self.logger.info(f"[{self.cam_id}] Live-view session reliably stopped and camera is idle.")

    @property
    def device(self):
        """Grabber Device-module 핸들(read-only). None일 수 있음."""
        return self._device

    # ─────────────────────────────────────────────────────────────
    #  Legacy grab control aliases
    # ─────────────────────────────────────────────────────────────
    def grab_start(
            self,
            buffer_count: Optional[int] = None,
            *,
            min_buffers: Optional[int] = None,
            align_bytes: int = 4096,
            force_realloc: bool = False,
    ) -> None:
        """
        **레거시 호환용** : 내부적으로 ``start_grab()`` 을 그대로 호출합니다.

        예전 코드에서
            ctrl.grab_start(...)
        로 부르던 부분을 그대로 유지할 수 있습니다.
        """
        return self.start_grab(
            buffer_count,
            min_buffers=min_buffers,
            align_bytes=align_bytes,
            force_realloc=force_realloc,
        )

    def flush_buffers(self, *, block: bool = False, timeout: float = 0.5) -> None:
        """
        DMA FIFO / pending-buffer 큐 비움.
        ⚠️ 그래빙/라이브뷰 중엔 입력 풀을 0으로 만들 수 있으므로 **skip**.
        """
        if not self.grabber:
            return
        if self._live_view or self.is_grabbing():
            self.logger.debug("[%s] flush_buffers skipped (live_view/grabbing)", self.cam_id)
            return

        try:
            if hasattr(self.grabber, "flush_buffers"):
                self.grabber.flush_buffers()
            elif hasattr(self.grabber, "dma_write_flush"):
                self.grabber.dma_write_flush()
        except Exception as exc:
            self.logger.debug("flush_buffers ignored error: %s", exc)

        if block:
            deadline = time.time() + timeout
            while self._pending_buffer_count() > 0 and time.time() < deadline:
                time.sleep(0.001)

    def wait_until_idle(self, timeout: float = 0.3) -> None:
        """
        Stream 큐가 비거나 timeout 초가 지날 때까지 대기.
        grab_stop 후 곧바로 param set 할 때 race condition 방지용.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self.grabber.stream.get_info("QueuedBuf") == 0:
                    return
            except Exception:
                return
            time.sleep(0.005)

    def stop_grab_safe(self, *, flush: bool = True, revoke: bool = True, wait_ms: int = 200) -> None:
        """
        Grabber가 여전히 grabbing 중이면 완전 정지.
        추가: 드레인 정지 선행, idle 보장 강화.
        """
        # 드레인 먼저 중지
        with suppress(Exception):
            self._stop_safety_drain()

        if not self.is_grabbing():
            return

        with suppress(Exception):
            self.execute_command("AcquisitionStop", timeout=0.2)

        self.stop_grab(flush=flush)

        if revoke and hasattr(self.grabber, "revokeAllBuffers"):
            with suppress(Exception):
                self.grabber.revokeAllBuffers()

        # 내부 큐 소거 확인
        deadline = time.time() + wait_ms / 1000.0
        while time.time() < deadline:
            try:
                if self.grabber.stream.get_info("QueuedBuf") == 0:
                    break
            except Exception:
                break
            time.sleep(0.01)

    def grab_stop(self) -> None:
        """
        **레거시 호환용** : 내부적으로 ``stop_grab()`` 을 그대로 호출합니다.

        예전 코드에서
            ctrl.grab_stop()
        로 부르던 부분을 그대로 유지할 수 있습니다.
        """
        self.stop_grab()

    # ───────────────── Internal GrabWorker (singleton) Management ─────────────────
    def _ensure_grab_worker(self, *, max_fps: int = 30) -> None:
        """Create or update the global GrabWorker when live-view is ON."""
        global _grab_worker
        if _grab_worker is None or not _grab_worker.isRunning():
            _grab_worker = GrabWorker(controllers=[self], max_fps=max_fps)
            _grab_worker.start()
            self.logger.debug("GrabWorker started (singleton)")
        elif self not in _grab_worker.controllers:
            _grab_worker.add_controller(self)
            self.logger.debug("GrabWorker » added %s", self.cam_id)


    def _detach_from_grab_worker(self) -> None:
        """Remove this controller from GrabWorker; stop thread if empty."""
        global _grab_worker
        if _grab_worker and self in _grab_worker.controllers:
            _grab_worker.remove_controller(self)
            if not _grab_worker.controllers:
                _grab_worker.stop()
                _grab_worker = None
                self.logger.debug("GrabWorker terminated (no controllers)")



    def setup_camera_for_hw_trigger(
            self,
            trigger_selector: Literal["ExposureStart", "FrameStart"] = "ExposureStart",
            trigger_source: Literal["LinkTrigger0", "Line1"] = "LinkTrigger0",
            activation: Literal["RisingEdge", "FallingEdge"] = "RisingEdge",
    ) -> None:
        self._check_connected()
        dev = self.params
        log = self.logger
        log.info(
            "[%s] 카메라 HW-Trigger 설정 시작  Selector=%s  Source=%s",
            self.cam_id, trigger_selector, trigger_source,
        )
        with suppress(Exception):
            self.execute_command("AcquisitionStop")
        time.sleep(0.05)
        if "TriggerSelector" in dev.features():
            dev.set("TriggerSelector", trigger_selector)
        if "TriggerMode" in dev.features():
            dev.set("TriggerMode", "Off")
        if "TriggerSource" in dev.features():
            dev.set("TriggerSource", trigger_source)
        if "TriggerActivation" in dev.features():
            dev.set("TriggerActivation", activation)
        if "TriggerMode" in dev.features():
            dev.set("TriggerMode", "On")
        log.info("[%s] 카메라 HW-Trigger 설정 완료", self.cam_id)

    # ────────────────────────────── Memento control ──────────────────────
    def toggle_memento(self, enable: bool, ring_mb: int = 256) -> None:
        with self._lock:  # Thread-safe
            if not EURESYS_AVAILABLE:
                self.logger.info(f"[{self.cam_id}] toggle_memento ignored – eGrabber unavailable.")
                return
            if enable == self._memento_live:
                return
            if self.grabber_index < 0:
                self.logger.warning(f"[{self.cam_id}] Grabber index unknown – toggle skipped.")
                return
            try:
                cmd = [
                    "ringbuffer",
                    "--enable" if enable else "--disable",
                    f"--grabber={self.grabber_index}",
                ]
                if enable:
                    cmd.append(f"--ringbuffer={ring_mb}m")
                _run_memento_cmd(cmd)
                self._memento_live = enable
                self.logger.info(f"[{self.cam_id}] Memento Live {'ON' if enable else 'OFF'} (grabber={self.grabber_index})")
            except FileNotFoundError:
                self.logger.warning(f"[{self.cam_id}] Memento CLI not found – toggle skipped.")
            except subprocess.CalledProcessError as e:
                self.logger.warning(f"[{self.cam_id}] Memento toggle failed: {e}")

    def connect_camera_by_info(self, cam_info: EGrabberCameraInfo) -> bool:
        from egrabber.generated.constants import DEVICE_ACCESS_CONTROL
        with self._lock:
            if self.connected:
                return True
            try:
                self.grabber = EGrabber(cam_info,
                                        device_open_flags=DEVICE_ACCESS_CONTROL)

                self.params = self.grabber.remote
                self._device = getattr(self.grabber, "device", None)  # ★ 수정

                gi = cam_info.grabbers[0]
                self.serial_number = gi.deviceSerialNumber or \
                                     f"SN_{gi.interface_index}_{gi.device_index}"
                self.cam_id = f"{gi.deviceModelName}-{self.serial_number}"
                self.grabber_index = int(
                    getattr(self.grabber, "get_info", lambda k: -1)("GrabberIndex") or -1
                )
                self.connected = True
                self.logger.info("[%s] CONNECT OK (DEVICE_ACCESS_CONTROL)", self.cam_id)
                return True
            except Exception as e:
                self.logger.error("connect_camera_by_info failed: %s", e, exc_info=True)
                self.disconnect_camera()
                return False


    # ------------------------------------------------------------------
    #  노드 존재 여부를 결정 – _find_nodemap() 재사용
    # ------------------------------------------------------------------
    def _feature_exists(self, name: str) -> bool:
        """
        어떤 레벨(nodemap)에 있든 *name* 을 노출하면 True.
        """
        return self._find_nodemap(name) is not None



    def setup_grabber_for_hw_trigger(
            self,
            *,  # keyword-only
            line_tool: str = "LIN1",
            link_trigger: str = "LinkTrigger0",
            edge: str = "RisingEdge",
            camera_control_method: str = "RC",
            cycle_trigger_source: str = "Immediate",
            cycle_period_us: float = 3360.0,
            idempotent: bool = True,
            verify: bool = True,
    ) -> None:
        from src.core.camera_exceptions import GrabberError

        dev = getattr(self.grabber, "device", None)
        if dev is None:  # 멀티-카메라 세컨드
            self.logger.warning("[%s] grabber.device unavailable – CIC setup skipped (shared grabber)",
                                self.cam_id)
            return

        feat = dev.features()
        f_exists = feat.__contains__

        def safe_set(n, v):
            try:
                if not f_exists(n):
                    return False
                if idempotent and str(dev.get(n)) == str(v):
                    return True
                dev.set(n, v)
                return True
            except Exception:  # 읽기 전용 등
                return False

        # ── ① FW 4.38+ : CycleTriggerSource 가 존재 ──────────────────
        if f_exists("CycleTriggerSource"):
            safe_set("CameraControlMethod", camera_control_method)
            if f_exists("CxpTriggerMessageSelector"):
                safe_set("CxpTriggerMessageSelector", link_trigger)
                safe_set("CxpTriggerMessageSource", "CycleTrigger")

            use_cg = cycle_trigger_source.lower() in ("immediate", "startcycle", "cyclegenerator0")
            if f_exists("DeviceLinkTriggerToolSelector"):
                safe_set("DeviceLinkTriggerToolSelector", link_trigger)
                safe_set("DeviceLinkTriggerToolSource",
                         "CycleGenerator0" if use_cg else line_tool)
                safe_set("DeviceLinkTriggerToolActivation", edge)

            safe_set("CycleTriggerSource", cycle_trigger_source)
            if use_cg and f_exists("CycleTriggerPeriod"):
                safe_set("CycleTriggerPeriod", float(cycle_period_us))

            # optional sanity
            if verify and f_exists("CycleLostTriggerCount"):
                if int(dev.get("CycleLostTriggerCount")):
                    raise GrabberError(f"[{self.cam_id}] non-zero CycleLostTriggerCount")

            self.logger.info("[%s] CIC ready (FW≥4.38, CTS=%s, Period=%.3f µs)",
                             self.cam_id, cycle_trigger_source, cycle_period_us)
            return

        # ── ② 구형 FW : DLT Toolbox 만 존재 ────────────────────────────
        dlt_nodes = {"DeviceLinkTriggerToolSelector",
                     "DeviceLinkTriggerToolSource",
                     "DeviceLinkTriggerToolActivation"}
        if dlt_nodes.issubset(feat):
            safe_set("DeviceLinkTriggerToolSelector", link_trigger)
            safe_set("DeviceLinkTriggerToolSource", line_tool)
            safe_set("DeviceLinkTriggerToolActivation", edge)
            self.logger.info("[%s] DLT mapping %s→%s (%s) – CIC not available",
                             self.cam_id, line_tool, link_trigger, edge)
            return

        # ── ③ 아무 경로도 없으면 예외 ─────────────────────────────────
        raise GrabberError(f"[{self.cam_id}] Neither CIC nor DLT nodes present – check firmware")


    # ───────────────────── NEW: CIC Counter 헬퍼 & 롱런 시험 ───────────────
    def reset_cic_counters(self) -> None:
        """CycleLostTriggerCount 등을 0으로 리셋."""
        if self._feature_exists("CycleLostTriggerCountReset"):
            self.grabber.device.run("CycleLostTriggerCountReset")

    def read_cic_lost(self) -> int:
        """현재 누락된 트리거 수(노드 없으면 0)."""
        if self._feature_exists("CycleLostTriggerCount"):
            return int(self.grabber.device.get("CycleLostTriggerCount"))
        return 0

    def trigger_integrity_test(
            self,
            expected_triggers: int,
            poll_interval: float = 5.0,
    ) -> None:
        """
        멀티-카메라 환경에서 HW-Trigger 롱런 무-누락 시험.
        누락 발생 시 GrabberError 예외로 즉시 중단.
        """
        from src.core.camera_exceptions import GrabberError

        self.reset_cic_counters()
        self.start_grab()
        recv = 0
        t_last = time.time()

        while recv < expected_triggers:
            recv = self.get_received_frame_count()
            if time.time() - t_last >= poll_interval:
                lost = self.read_cic_lost()
                self.logger.info("[%s] progress %d/%d, lost=%d",
                                 self.cam_id, recv, expected_triggers, lost)
                if lost:
                    raise GrabberError(
                        f"[{self.cam_id}] 롱런 시험 실패 – 누락 {lost} (수신 {recv})"
                    )
                t_last = time.time()
            time.sleep(0.002)

        self.logger.info("[%s] 롱런 시험 통과 – 누락 0 (수신 %d)", self.cam_id, recv)

    def get_received_frame_count(self) -> int:
        """Frames successfully grabbed since the last start."""
        return int(self.stats.get("frame_count", 0))

    # ───────────────────── PATCH: enum helper (3.7+ safe) ────────────────
    def _enum_entries(self, node_name: str) -> List[str]:
        """주어진 Enumeration 노드의 모든 엔트리 이름을 반환."""
        return self.grabber.device.get_enum_entries(node_name)


    def stop_grab_flush(self) -> None:
        """즉시 중단 + flush_buffers() ― 사용 시 크래시 위험 주의!"""
        self.stop_grab(flush=True)

    def discover_cameras(self) -> List[Dict[str, Any]]:
        """
        카메라-centric 스캔 (EGrabberDiscovery.cameras 활용).
        """
        with self._lock:
            cams: List[Dict[str, Any]] = []
            try:
                disc = EGrabberDiscovery(self.gentl)
                disc.discover(find_cameras=True)

                # cameras 는 len/[] 만 지원 → range() 로 인덱스 반복
                for idx in range(len(disc.cameras)):
                    cam_info: EGrabberCameraInfo = disc.cameras[idx]

                    # 첫 grabber(=포트0) 로부터 인덱스·모델·시리얼 추출
                    gi = cam_info.grabbers[0]           # type: EGrabberInfo

                    online = _is_online(cam_info)       # 빠른 alive check
                    self.logger.debug("[%s] cam#%d (%d/%d/%d) online=%s",
                                      self.cam_id, idx,
                                      gi.interface_index, gi.device_index,
                                      gi.stream_index, online)

                    cams.append({
                        "camera_info": cam_info,
                        "iface_idx":   gi.interface_index,
                        "dev_idx":     gi.device_index,
                        "stream_idx":  gi.stream_index,
                        "vendor":      gi.deviceVendorName,
                        "model":       gi.deviceModelName,
                        "serial":      gi.deviceSerialNumber,
                        "online":      online,
                    })
            except Exception as e:
                self.logger.error("[%s] discovery failed: %s",
                                  self.cam_id, e, exc_info=True)

            self.logger.info("[%s] Discovered %d camera(s) (raw)",
                             self.cam_id, len(cams))
            return cams

    @staticmethod
    def connect_all(enable_param_cache: bool = False) -> List['CameraController']:
        controller_pool.flush(disconnect=True)
        try:
            discovery = EGrabberDiscovery(_get_gentl())
            discovery.discover(find_cameras=True)
            camera_infos = [discovery.cameras[i] for i in range(len(discovery.cameras))]
        except Exception as e:
            logger.error(f"Camera discovery failed: {e}", exc_info=True)
            return []

        connected_controllers: List[CameraController] = []
        for cam_info in camera_infos:
            if _is_online(cam_info):
                ctrl = CameraController(enable_param_cache=enable_param_cache)
                if ctrl.connect_camera_by_info(cam_info):
                    controller_pool.register(ctrl)
                    connected_controllers.append(ctrl)
        return connected_controllers
    def setup_for_hardware_trigger(self, source: str = "LinkTrigger0") -> None:
        """
        Legacy API wrapper. 외부 코드 호환성을 위해 유지.
        """
        self.setup_camera_for_hw_trigger(
            trigger_selector="ExposureStart",
            trigger_source=source,
            activation="RisingEdge",
        )
    # ------------------------------------------------------------------
    #  Grabber CIC (내부 Trigger Generator) 활성화 – 안정판
    # ------------------------------------------------------------------
    def _enable_internal_grabber_trigger(
            self,
            *,
            link_trigger: str = "LinkTrigger0",
            min_period_us: float = 20000.0,
            cycle_source: str = "Immediate",
            trigger_activation: str = "RisingEdge",
            verify: bool = True,
    ) -> bool:
        """
        Grabber(Device-module) CIC 활성화.
        • 주기 노드 후보: CycleTriggerPeriod / CyclePeriodUs / CycleTargetPeriod / CycleMinimumPeriod
        • Min 메타로 클램프 후 set, read-back 검증
        """
        dev = getattr(self.grabber, "device", None)
        if dev is None:
            self.logger.warning("[%s] grabber.device unavailable – CIC enable skipped", self.cam_id)
            return False

        feats = dev.features()
        exists = feats.__contains__

        # 기본 제어/매핑 (가능한 노드만)
        try:
            if exists("CameraControlMethod"):
                dev.set("CameraControlMethod", "RC")
            if exists("CxpTriggerMessageSelector"):
                dev.set("CxpTriggerMessageSelector", link_trigger)
            if exists("CxpTriggerMessageSource"):
                dev.set("CxpTriggerMessageSource", "CycleTrigger")
            if exists("DeviceLinkTriggerToolSelector"):
                dev.set("DeviceLinkTriggerToolSelector", link_trigger)
            if exists("DeviceLinkTriggerToolSource"):
                dev.set("DeviceLinkTriggerToolSource", "CycleGenerator0")
            if exists("DeviceLinkTriggerToolActivation"):
                dev.set("DeviceLinkTriggerToolActivation", trigger_activation)
            if exists("CycleTriggerSource"):
                dev.set("CycleTriggerSource", cycle_source)
        except Exception as exc:
            self.logger.debug("[%s] CIC pre-config write failed: %s", self.cam_id, exc)

        # 주기 노드 선택
        period_node = None
        for cand in ("CycleTriggerPeriod", "CyclePeriodUs", "CycleTargetPeriod", "CycleMinimumPeriod"):
            if exists(cand):
                period_node = cand
                break

        if not period_node:
            self.logger.info("[%s] CIC fixed-mode (no period node).", self.cam_id)
            return True  # 최소 동작은 가능

        requested = float(min_period_us)
        applied = requested
        # Min 메타로 클램프
        try:
            from egrabber import query
            q_min = query.info(period_node, "Min")
            min_limit = dev.get(q_min, float) if q_min else None
            if isinstance(min_limit, (int, float)):
                applied = max(requested, float(min_limit))
        except Exception:
            pass

        # 쓰기 + 검증
        try:
            dev.set(period_node, applied)
            read_back = float(dev.get(period_node))
            if verify and read_back < applied - 1e-3:
                self.logger.warning("[%s] CIC period read-back < applied (%.3f < %.3f) via %s",
                                    self.cam_id, read_back, applied, period_node)
            self.logger.info("[%s] CIC enabled → node=%s set=%.1fµs read=%.1fµs src=%s act=%s",
                             self.cam_id, period_node, applied, read_back, cycle_source, trigger_activation)
            return True
        except Exception as exc:
            self.logger.warning("[%s] CIC period write failed via %s: %s", self.cam_id, period_node, exc)
            return False

    def enable_internal_grabber_trigger(self, **kwargs) -> bool:
        return self._enable_internal_grabber_trigger(**kwargs)


    def execute_grabber_cycle_trigger(self) -> None:
        """
        Grabber(Device-module) 쪽에서 **하나의 트리거 펄스**를 발생시킨다.
        • FW ≥ 4.38  :  StartCycle  노드
        • 구형 FW     :  CxpTriggerMessageSend  +  LinkTrigger0
        """
        from src.core.camera_exceptions import CommandExecutionError

        dev = getattr(self.grabber, "device", None)
        if not dev:
            raise CommandExecutionError("grabber.device is None")

        # 신형 FW – StartCycle
        if "StartCycle" in dev.features():
            dev.execute("StartCycle")
            self.logger.debug("[%s] Device.StartCycle executed", self.cam_id)
            return

        # 구형 FW – CXP Trigger Message
        if "CxpTriggerMessageSend" in dev.features():
            with suppress(Exception):
                if "CxpTriggerMessageID" in dev.features():
                    dev.set("CxpTriggerMessageID", 0)
            dev.execute("CxpTriggerMessageSend")
            self.logger.debug("[%s] CxpTriggerMessageSend executed", self.cam_id)
            return

        raise CommandExecutionError("No cycle-trigger command available in Device-module")

    def connect_camera(self) -> bool:
        """Connect to the first available grabber (generic path)."""
        with self._lock:
            if self.connected:
                self.logger.warning("[%s] Already connected.", self.cam_id)
                return True
            try:
                self.logger.debug("[%s] Initializing EGrabber…", self.cam_id)
                self.grabber = EGrabber(self.gentl)
                if not getattr(self.grabber, "remote", None):
                    raise CameraConnectionError("remote interface unavailable")

                # GenICam nodemaps
                self.params = self.grabber.remote
                self._device = getattr(self.grabber, "device", None)  # ★ 수정
                if self._device is None:
                    with suppress(GenTLException, AttributeError):
                        self._device = self.grabber._get_device_module()  # type: ignore
                if self._device is None:
                    self.logger.warning("[%s] Device-module missing ⇒ CIC 제한", self.cam_id)

                # IDs
                self.serial_number = self.get_device_serial_number()
                self.cam_id = self.serial_number or self.cam_id
                gi = getattr(self.grabber, "get_info", lambda k: -1)
                self.grabber_index = int(gi("GrabberIndex") or -1)

                # state
                self.connected, self.grabbing = True, False
                controller_pool.register(self)
                self.camera_connected.emit()
                self.logger.info("[%s] CONNECT OK", self.cam_id)
                return True

            except Exception as e:
                self.logger.error("[%s] connect_camera failed: %s",
                                  self.cam_id, e, exc_info=True)
                self.disconnect_camera()
                raise CameraConnectionError(f"[{self.cam_id}] {e}") from e



    def configure_trigger(self, mode: str, source: str,
                          activation: str = "RisingEdge") -> None:
        self._check_connected()
        p = self.params
        self.logger.info("[%s] 트리거 설정 시작: Mode=%s, Source=%s", self.cam_id, mode, source)

        if "TriggerSelector" in p.features() and self.is_writeable("TriggerSelector"):
            avail = self.get_enumeration_entries("TriggerSelector")
            selector = next((s for s in ("FrameStart", "ExposureStart") if s in avail),
                            avail[0] if avail else None)
            if selector:
                self.set_param("TriggerSelector", selector)

        if self.is_writeable("TriggerMode"):
            self.set_param("TriggerMode", "Off")
        else:
            raise ParameterSetError("TriggerMode를 Off로 설정 불가")

        if mode == "On":
            if self.is_writeable("TriggerSource"):
                if source not in self.get_enumeration_entries("TriggerSource"):
                    raise ParameterSetError(
                        f"[{self.cam_id}] TriggerSource '{source}' not supported "
                        f"({self.get_enumeration_entries('TriggerSource')})"
                    )
                self.set_param("TriggerSource", source)
            if self.is_writeable("TriggerActivation"):
                self.set_param("TriggerActivation", activation)

        if self.is_writeable("TriggerMode"):
            self.set_param("TriggerMode", mode)

        self.logger.info("[%s] 트리거 설정 완료", self.cam_id)

    def execute_command(self, node_name: str, timeout: float = 1.0) -> None:
        self._check_connected()

        node_maps = [self.params, getattr(self.grabber, "device", None), getattr(self.grabber, "remote", None)]
        long_ops_delay_ms = {
            "UserSetLoad": 10000, "UserSetSave": 8000, "DeviceReset": 15000, "Reboot": 15000,
            "LUTGenerate": 8000, "LUTSave": 6000, "LUTLoad": 6000, "FlatFieldCorrection": 10000,
            "StartCalibration": 10000,
        }

        for nm in filter(None, node_maps):
            # Node-API 우선
            if hasattr(nm, "getNode"):
                with suppress(Exception):
                    node = nm.getNode(node_name)
                    if node and node.isFeature() and node.isWritable() and hasattr(node, "execute"):
                        node.execute()
                        if timeout > 0 and hasattr(node, 'wait_until_done'):
                            node.wait_until_done(timeout)
                        self.logger.info("[%s] Command '%s' executed via Node-API on %s.",
                                         self.cam_id, node_name, type(nm).__name__)
                        if node_name in long_ops_delay_ms:
                            # ★ Quiet + 트리거/획득 일시 정지
                            self.begin_quiet(long_ops_delay_ms[node_name],
                                             pause_acquisition=True, pause_trigger=True)
                        return

            # Raw execute
            if hasattr(nm, "execute") and hasattr(nm, "features") and node_name in nm.features():
                try:
                    nm.execute(node_name)
                    self.logger.info("[%s] Command '%s' executed via raw execute() on %s.",
                                     self.cam_id, node_name, type(nm).__name__)
                    if node_name in long_ops_delay_ms:
                        self.begin_quiet(long_ops_delay_ms[node_name],
                                         pause_acquisition=True, pause_trigger=True)
                    return
                except Exception:
                    pass

        raise CommandExecutionError(
            f"[{self.cam_id}] Command '{node_name}' not found or not writable in any available nodemap."
        )

    def _reset_gentl_instance(self):
        """
        [신규 메소드] GenTL 드라이버 인스턴스를 강제로 해제하고 재생성하여,
        드라이버 레벨의 상태를 완벽하게 초기화합니다. Reboot 문제 해결의 핵심입니다.
        """
        global _gentl_singleton
        with self._lock:
            try:
                if self.gentl and hasattr(self.gentl, 'close'):
                    self.gentl.close()
                _gentl_singleton = None # 싱글턴 참조를 끊어 재생성을 유도
                self.gentl = _get_gentl() # 새로운 GenTL 인스턴스를 가져옴
                logger.info(f"[{self.cam_id}] GenTL instance has been forcefully reset.")
            except Exception as e:
                logger.error(f"[{self.cam_id}] Failed to reset GenTL instance: {e}", exc_info=True)

    def disconnect_camera(self) -> None:
        """
        [수정됨] 연결을 해제하고 이 컨트롤러가 사용하던 EGrabber 리소스만 정리합니다.
        애플리케이션의 전역 GenTL 인스턴스는 변경하지 않습니다.
        """
        with self._lock:
            if not self.connected:
                return

            try:
                if self.is_grabbing() or self._live_view:
                    self.stop_live_view()
            finally:
                self.grabber = None
                self.params = None
                self._device = None
                self.connected = False
                self.grabbing = False

                # 전역 GenTL 리셋 금지
                # self._reset_gentl_instance()

                self.logger.info(f"[{self.cam_id}] Camera resources released.")

    def send_one_trigger(self, *, prefer_device: bool = True) -> str:
        """
        단발 트리거를 전송 (3단 폴백):
          1) Device.StartCycle  (그래버 CIC)
          2) Device.CxpTriggerMessageSend (필요 시 ID/Selector/Source 세팅)
          3) Camera TriggerSoftware (safe wrapper)
        반환: "device_startcycle" | "device_cxpmsg" | "camera_sw" | "failed:…"
        """
        dev = getattr(self.grabber, "device", None)

        if prefer_device and dev and hasattr(dev, "features"):
            feats = set()
            with suppress(Exception): feats = set(dev.features())

            if "StartCycle" in feats:
                try:
                    dev.execute("StartCycle")
                    return "device_startcycle"
                except Exception:
                    pass

            if "CxpTriggerMessageSend" in feats:
                with suppress(Exception):
                    if "CxpTriggerMessageID" in feats:
                        dev.set("CxpTriggerMessageID", 0)
                    if "CxpTriggerMessageSelector" in feats:
                        dev.set("CxpTriggerMessageSelector", "LinkTrigger0")
                    if "CxpTriggerMessageSource" in feats:
                        dev.set("CxpTriggerMessageSource", "HostCommandRisingEdge")
                with suppress(Exception):
                    dev.execute("CxpTriggerMessageSend")
                    return "device_cxpmsg"

        try:
            self.trigger_software_safe()
            return "camera_sw"
        except Exception as e:
            return f"failed:{e}"

    def diagnose_timeout_counters(self) -> Dict[str, Any]:
        """
        -1011 타임아웃 근본원인 분석 스냅샷을 반환.
        예:
        {
          "fifo": {"in":1, "out":0},
          "device": {"lost_triggers":0, "ack":1000, "processed":1000},
          "stream_errors": {"...": count, ..., "total": N}
        }
        """
        snap = {"fifo": {"in": -1, "out": -1}, "device": {}, "stream_errors": {"total": 0}}

        g = getattr(self, "grabber", None)
        s = getattr(g, "stream", None) if g else None
        d = getattr(g, "device", None) if g else None

        # FIFO
        with suppress(Exception):
            snap["fifo"]["in"]  = int(s.get_info("InputFifo"))
        with suppress(Exception):
            snap["fifo"]["out"] = int(s.get_info("OutputFifo"))

        # Device counters: lost / ack / processed
        if d and hasattr(d, "features"):
            feats = set()
            with suppress(Exception): feats = set(d.features())

            if "CycleLostTriggerCount" in feats:
                with suppress(Exception):
                    snap["device"]["lost_triggers"] = int(d.get("CycleLostTriggerCount"))

            if {"EventSelector", "EventCount"} <= feats:
                with suppress(Exception):
                    d.set("EventSelector", "CxpTriggerAck")
                    ack = int(d.get("EventCount")) // 2  # rising+falling
                    snap["device"]["ack"] = ack
                with suppress(Exception):
                    d.set("EventSelector", "CameraTriggerRisingEdge")
                    snap["device"]["processed"] = int(d.get("EventCount"))

        # Stream error counters (전부 합산)
        if s and hasattr(s, "features") and "ErrorSelector" in s.features() and "ErrorCount" in s.features():
            try:
                if query:
                    q_enum = query.enum_entries("ErrorSelector")
                    enums = s.get(q_enum, list) or []
                else:
                    enums = []
                tot = 0
                for e_name in enums:
                    with suppress(Exception):
                        s.set("ErrorSelector", e_name)
                        cnt = int(s.get("ErrorCount"))
                        if cnt > 0:
                            snap["stream_errors"][e_name] = cnt
                            tot += cnt
                snap["stream_errors"]["total"] = tot
            except Exception:
                pass

        return snap
    # ────────────────────────────── Grab control ─────────────────────────
    def is_grabbing(self) -> bool:
        """
        Check if the camera is currently grabbing frames.
        """
        with self._lock:
            if not self.connected or not self.grabber:
                return False
            # EGrabber의 is_grabbing 메서드 호출, 없으면 self.grabbing 반환
            return getattr(self.grabber, "is_grabbing", lambda: self.grabbing)()
    def get_grabber(self):
        """
        Return the underlying EGrabber instance.

        Added for backward-compatibility with legacy action modules that
        expect ctrl.get_grabber() to exist.
        """
        with self._lock:
            return self.grabber

    def _pending_buffer_count(self) -> int:
        """
        현재 DataStream 쪽에 큐잉된 버퍼 개수 (best-effort).
        기존 코드가 self.remote 를 보던 버그를 수정.
        """
        s = getattr(self.grabber, "stream", None)
        if not s or not hasattr(s, "get_info"):
            return 0
        for k in ("QueuedBuf", "NewBufferEventUnreadCount", "NewBuffer"):
            try:
                return int(s.get_info(k))
            except Exception:
                continue
        return 0

    def _start_safety_drain(self) -> None:
        """
        소비자가 전혀 없을 때만 출력 FIFO를 아주 느리게 비움.
        • 빈 큐에서는 pop 호출 금지
        • 주기 완화(기본 80ms)로 드라이버 로그/CPU 낭비 감소
        """
        if self._live_view or self._is_safety_drain_running():
            return
        import threading, time
        self._drain_stop = threading.Event()

        def _drain():
            PERIOD = 0.08
            while not self._drain_stop.is_set():
                try:
                    if self._stream_has_output():
                        self._flush_one_buffer()
                    else:
                        # 입력 풀 고갈 시 보강 (팝 없이)
                        with suppress(Exception):
                            self.ensure_stream_ready(min_input_fifo=1, refill_if_zero=True)
                except Exception:
                    pass
                time.sleep(PERIOD)

        self._drain_thr = threading.Thread(target=_drain, name="SafetyDrain", daemon=True)
        self._drain_thr.start()

    def _stop_safety_drain(self):
        thr = getattr(self, "_drain_thr", None)
        ev = getattr(self, "_drain_stop", None)
        if ev:
            with suppress(Exception): ev.set()
        if thr and getattr(thr, "is_alive", lambda: False)():
            with suppress(Exception): thr.join(timeout=0.2)

    def _set_stream_overflow_policy(self) -> None:
        """
        Stream 모듈에 버퍼 오버플로 정책 노드가 있을 때만, '드롭' 성격의 심볼로 설정.
        존재하지 않거나 심볼이 다르면 아무 것도 하지 않음(에러 로그 금지).
        """
        sm = getattr(self.grabber, "stream", None)
        if not sm or not hasattr(sm, "features"):
            return

        candidates = ("StreamBufferHandlingMode", "BufferHandlingMode")
        for node in candidates:
            if node not in sm.features():
                continue

            # 실제 enum 엔트리 읽기 (없는 빌드는 조용히 패스)
            try:
                entries = []
                if hasattr(sm, "getEnum"):
                    entries = list(sm.getEnum(node))
                elif hasattr(sm, "getNode"):
                    nd = sm.getNode(node)
                    if nd:
                        entries = [e.getSymbolic() for e in nd.getEnumEntries() if e.isAvailable()]
            except Exception:
                entries = []

            if not entries:
                continue

            for prefer in ("DropNewest", "DropOldest",
                           "NewestFirst", "NewestOnly",
                           "OldestFirstOverwrite", "OverwriteOldest"):
                if prefer in entries:
                    try:
                        sm.set(node, prefer)
                        self.logger.info("[%s] %s = %s", self.cam_id, node, prefer)
                    except Exception:
                        pass
                    return

    def _read_stream_triplet(self) -> tuple[int, int, int]:
        """
        Stream에서 (input_fifo, output_fifo, announced) 값을 다중 키로 읽는다.
        모르면 -1.
        """
        g = getattr(self, "grabber", None)
        s = getattr(g, "stream", None) if g else None
        if not s or not hasattr(s, "get_info"):
            return -1, -1, -1

        # Input FIFO 후보
        in_keys = ("InputFifo", "InputPool", "InputBufferPool")
        # Output FIFO/Unread 후보
        out_keys = ("OutputFifo", "NewBufferEventUnreadCount", "NewBuffer")
        # Announced 후보
        announced_keys = ("AnnouncedBuffers", "AnnouncedBuf", "BuffersAnnounced", "Announced")

        def _try(keys):
            for k in keys:
                try:
                    v = int(s.get_info(k))
                    return v
                except Exception:
                    continue
            return -1

        in_fifo = _try(in_keys)
        out_fifo = _try(out_keys)
        announced = _try(announced_keys)

        # 일부 빌드에서는 QueuedBuf가 사실상 announce/queue 된 수로 근사됨
        if announced < 0:
            try:
                q = int(s.get_info("QueuedBuf"))
                if q >= 0:
                    announced = q
            except Exception:
                pass

        return in_fifo, out_fifo, announced

    def _announced_buf_count(self) -> int:
        """현재 announce된 버퍼 수. stream의 여러 info 키와 grabber API를 모두 시도. 실패 시 -1."""
        # ① stream 우선
        in_fifo, out_fifo, announced = self._read_stream_triplet()
        if announced >= 0:
            return announced

        # ② grabber 쪽 메서드
        g = getattr(self, "grabber", None)
        if not g:
            return -1
        for name in ("get_announced_buffer_count", "announced_buffer_count", "announcedBuffersCount"):
            fn = getattr(g, name, None)
            if callable(fn):
                try:
                    return int(fn())
                except Exception:
                    pass
        return -1

    def _queue_announced_buffers_any(self) -> tuple[bool, int, int, int]:
        """
        grabber/stream 양쪽 queue/requeue API를 시도하고, 최신 (in,out,announced)를 돌려준다.
        """
        g = getattr(self, "grabber", None)
        s = getattr(g, "stream", None) if g else None
        methods = (
            "queueAnnouncedBuffers", "queue_announced_buffers",
            "requeueAnnouncedBuffers", "requeue_announced_buffers",
            "requeueBuffers", "requeue_buffers",
            "queueAllBuffers", "queue_all_buffers",
            "fill_input_pool", "refill_input_pool",
        )
        success = False
        for obj in (g, s):
            if not obj:
                continue
            for name in methods:
                fn = getattr(obj, name, None)
                if callable(fn):
                    try:
                        fn()
                        success = True
                    except Exception as e:
                        self.logger.debug("[%s] %s.%s() failed: %s",
                                          self.cam_id, type(obj).__name__, name, e)

        in_fifo, out_fifo, announced = self._read_stream_triplet()
        return success, in_fifo, out_fifo, announced

    def _fill_input_pool(self) -> bool:
        """
        announced/반납된 버퍼들을 입력 풀로 재-큐.
        (grabber 와 stream 양쪽 경로 모두 시도)
        """
        ok, in_fifo, out_fifo, announced = self._queue_announced_buffers_any()
        if ok:
            self.logger.debug("[%s] input pool refilled → in=%s, out=%s, announced=%s",
                              self.cam_id, in_fifo, out_fifo, announced)
        # in_fifo를 모르는(-1) 빌드에선 announced>=1이면 ‘존재’로 인정
        return bool(ok or (in_fifo >= 1) or (in_fifo < 0 and announced >= 1))

    def ensure_stream_ready(self, *, min_input_fifo: int = 1, refill_if_zero: bool = True) -> Dict[str, Any]:
        """
        스트림 FIFO 상태 점검/복구.
        재할당 필요 시 'needs_realloc': True 로만 표시 (여기서는 재할당하지 않음).
        """
        info = {"input_fifo": -1, "output_fifo": -1, "refilled": False, "needs_realloc": False}
        g = getattr(self, "grabber", None)
        s = getattr(g, "stream", None) if g else None
        if not s or not hasattr(s, "get_info"):
            self.logger.debug("[%s] No stream.get_info() → skipping FIFO check", self.cam_id)
            return info

        with suppress(Exception):
            info["input_fifo"] = int(s.get_info("InputFifo"))
        with suppress(Exception):
            info["output_fifo"] = int(
                self._get_info_multi(s, ("OutputFifo", "NewBufferEventUnreadCount", "NewBuffer"), -1))

        if refill_if_zero and (info["input_fifo"] < min_input_fifo):
            ok, in_fifo, out_fifo, announced = self._queue_announced_buffers_any()
            info["refilled"] = bool(ok)
            if in_fifo >= 0: info["input_fifo"] = in_fifo
            if out_fifo >= 0: info["output_fifo"] = out_fifo
            # ‘진짜 0’로 확정(in_fifo==0)이고 announced도 0/미만이면 재할당 필요
            if (in_fifo == 0) and (announced <= 0):
                info["needs_realloc"] = True

        return info

    def arm_trigger_path(self,
                         *,
                         camera_selector: str = "ExposureStart",
                         camera_source: str   = "LinkTrigger0",
                         camera_activation: str = "RisingEdge",
                         cic_source: str = "Immediate",   # 또는 "StartCycle"
                         link_trigger: str = "LinkTrigger0",
                         period_us: float = 20000.0) -> Dict[str, Any]:
        """
        트리거 경로를 '카메라 ←(LinkTrigger0)← 그래버 CIC'로 안전하게 무장.
        - 카메라: TriggerMode/Source/Activation
        - 그래버 Device: CIC 매핑 + 소스/주기 (Min 클램프)
        실패하는 노드는 조용히 건너뛴다.
        """
        applied = {"camera": {}, "device": {}, "notes": []}

        # 1) 카메라 트리거 세팅
        try:
            self.configure_trigger(mode="On", source=camera_source, activation=camera_activation)
            applied["camera"].update({"selector": camera_selector, "source": camera_source, "activation": camera_activation})
        except Exception as e:
            applied["notes"].append(f"camera trigger config skipped/failed: {e}")

        # 2) 그래버 CIC
        dev = getattr(self.grabber, "device", None)
        if not dev or not hasattr(dev, "features"):
            applied["notes"].append("grabber.device unavailable")
            return applied

        feats = set()
        with suppress(Exception): feats = set(dev.features())

        def _maybe_set(name, val):
            if name in feats:
                with suppress(Exception):
                    dev.set(name, val)
                    applied["device"][name] = val
                    return True
            return False

        _maybe_set("CameraControlMethod", "RC")
        _maybe_set("DeviceLinkTriggerToolSelector", link_trigger)
        _maybe_set("DeviceLinkTriggerToolSource",   "CycleGenerator0")
        _maybe_set("DeviceLinkTriggerToolActivation", "RisingEdge")

        _maybe_set("CycleTriggerSource", cic_source)
        period_node = next((n for n in ("CycleTriggerPeriod", "CyclePeriodUs",
                                        "CycleTargetPeriod", "CycleMinimumPeriod")
                            if n in feats), None)
        if period_node:
            applied_val = float(period_us)
            try:
                if query:
                    qmin = query.info(period_node, "Min")
                    minv = dev.get(qmin, float) if qmin else None
                    if isinstance(minv, (int, float)):
                        applied_val = max(applied_val, float(minv))
            except Exception:
                pass
            with suppress(Exception):
                dev.set(period_node, applied_val)
                applied["device"][period_node] = float(dev.get(period_node))
        else:
            with suppress(Exception):
                applied["device"]["CycleMinimumPeriod"] = float(dev.get("CycleMinimumPeriod"))

        return applied

    def start_grab(self,
                   buffer_count: Optional[int] = None, *,
                   min_buffers: Optional[int] = None,
                   align_bytes: int = 4096,
                   force_realloc: bool = False) -> None:
        """
        안정화된 시작 시퀀스 (최종판)
          ① 필요 시 안전 재할당(완전 유휴화) → _realloc_buffers_safe()
          ② 스트림 오버플로 정책 Drop 계열로
          ③ 입력 풀 보장(리필/보강) + 짧은 폴링 리트라이
          ④ in_fifo=0 '확정'이면 차단, '모름'(-1)인 경우 announced>=1이면 통과
             (모름/-1 이고 announced 도 -1이면 경고만하고 진행: 드라이버가 info 미제공)
          ⑤ start() 후 세이프티 드레인/백프레셔 워치독 가동
          ⑥ 통계 초기화
        """
        with self._lock:
            self._check_connected()

            if self.is_grabbing():
                self.logger.debug("[%s] Start grab ignored: already grabbing.", self.cam_id)
                return

            # 요청 버퍼 수 계산 (최소 4)
            req_buffers = max(int(min_buffers or buffer_count or 8), 4)
            self._buf_target = req_buffers

            # 재할당 필요 여부 판단
            need_realloc = force_realloc
            if not need_realloc:
                try:
                    get_cnt = getattr(self.grabber, "get_announced_buffer_count",
                                      getattr(self.grabber, "announced_buffer_count", lambda: 0))
                    if int(get_cnt()) < req_buffers:
                        need_realloc = True
                except Exception:
                    need_realloc = True

            # ① 필요 시 안전 재할당
            if need_realloc:
                self.logger.info("[%s] Reallocating %d buffers (safe path)...", self.cam_id, req_buffers)
                self._realloc_buffers_safe(req_buffers, 0, refill=False)
                time.sleep(0.03)

            # ② 오버플로 정책(가능할 때 Drop 계열) – 과포화 시 프레임 스톨 방지
            with suppress(Exception):
                self._set_stream_overflow_policy()

            # ③ 입력 풀 보장(+ 짧은 폴링 리트라이, 최대 150ms)
            poll_deadline = time.monotonic() + 0.15  # 최대 150ms 대기
            ok = False
            in_fifo = out_fifo = announced = -1

            while True:
                _ok, in_fifo, out_fifo, announced = self._queue_announced_buffers_any()
                ok = ok or _ok
                # 판정 조건: (1) in_fifo>=1 이거나 (2) in_fifo 모름인데 announced>=1 이면 충분
                if (in_fifo >= 1) or (in_fifo < 0 and announced >= 1):
                    break
                if time.monotonic() >= poll_deadline:
                    break
                time.sleep(0.01)

            # announce/queue가 너무 적다고 판단되면 한 번 더 재할당+리필 시도
            if (in_fifo <= 0 and announced <= 0):
                self.logger.info("[%s] Input probe weak → safe realloc(%d,0)+refill(retry)",
                                 self.cam_id, self._buf_target)
                self._realloc_buffers_safe(int(getattr(self, "_buf_target", req_buffers)), 0, refill=True)
                time.sleep(0.03)
                _ok, in_fifo, out_fifo, announced = self._queue_announced_buffers_any()
                ok = ok or _ok

            # ④ 최종 게이트 (완화 논리)
            #  A) in_fifo가 0으로 '확정'되면 실패 (진짜 비었음)
            #  B) in_fifo가 '모름'이고 announced도 '모름'이면 '경고만' 찍고 통과(드라이버가 info 미제공)
            if (in_fifo >= 0 and in_fifo == 0):
                raise GrabberStartError(
                    f"[{self.cam_id}] Input pool confirmed empty (in_fifo=0) – cannot start acquisition."
                )
            if (in_fifo < 0 and announced < 0):
                self.logger.warning(
                    "[%s] Input pool state unknown (in_fifo=-1, announced=-1) – proceeding defensively.",
                    self.cam_id)

            # ⑤ 실제 시작
            self.grabber.start()
            self.grabbing = True

            # 소비자(GrabWorker/동기 pop)가 없으면 얇은 드레인 스타트(빈 큐 pop 금지 버전)
            if not self._live_view:
                with suppress(Exception):
                    self._start_safety_drain()

            # 백프레셔 워치독 시작(GrabWorker 유무와 무관하게 안전망으로 동작)
            with suppress(Exception):
                self._start_backpressure_watchdog()

            # ⑥ 통계 초기화
            now = time.time()
            self.stats.update(frame_count=0, error_count=0, last_frame_time=now,
                              start_time=now, fps=0.0, frame_loss=0.0)
            self._last_stats_update = now

            self.logger.info("[%s] Grabbing started (buffers=%d, in_fifo=%s, out_fifo=%s, announced=%s)",
                             self.cam_id, req_buffers, in_fifo, out_fifo, announced)

    def _gate_triggers(self, on: bool) -> None:
        """
        카메라/그래버 트리거 경로를 원자적으로 ON/OFF.
        - 카메라: TriggerMode On/Off
        - 그래버 Device CIC: Start/Stop 또는 Source=Disabled 등 가능한 경로 우선 적용
        실패해도 조용히 넘어가되, 가능한 많이 닫는다(OFF) / 연다(ON).
        """
        dev = getattr(self.grabber, "device", None)
        # 1) 카메라 TriggerMode
        try:
            if self.params and "TriggerMode" in self.params.features():
                cur = str(self.params.get("TriggerMode"))
                tgt = "On" if on else "Off"
                if cur != tgt:
                    self.params.set("TriggerMode", tgt)
        except Exception:
            pass

        if not dev or not hasattr(dev, "features"):
            return

        feats = set()
        try:
            feats = set(dev.features())
        except Exception:
            pass

        # 2) 그래버 CIC/링크 트리거
        try:
            if on:
                # 가능한 Start 우선
                if "StartCycle" in feats:
                    dev.execute("StartCycle")
                elif "CycleTriggerSource" in feats:
                    # Immediate 등 기존 설정을 살리되 Disabled였다면 Immediate로
                    try:
                        src = str(dev.get("CycleTriggerSource"))
                        if src.lower() in ("disabled", "off", "none"):
                            dev.set("CycleTriggerSource", "Immediate")
                    except Exception:
                        dev.set("CycleTriggerSource", "Immediate")
            else:
                # OFF: Stop 우선 → Source=Disabled → Message Blocking
                if "StopCycle" in feats:
                    dev.execute("StopCycle")
                elif "CycleTriggerSource" in feats:
                    dev.set("CycleTriggerSource", "Disabled")
                # 선택: CxpTriggerMessageSource/Selector로 하드 차단(빌드별 상이)
                if "DeviceLinkTriggerToolSource" in feats:
                    try:
                        dev.set("DeviceLinkTriggerToolSource", "Off")
                    except Exception:
                        pass
        except Exception:
            pass

    def _output_fifo_capacity(self) -> int:
        """
        Output FIFO 총 용량 추정. 정보가 없으면 128로 가정.
        """
        s = getattr(self.grabber, "stream", None)
        if s and hasattr(s, "get_info"):
            for k in ("OutputFifoCapacity", "NewBufferEventCapacity", "NewBufferCapacity"):
                try:
                    cap = int(s.get_info(k))
                    if cap > 0:
                        return cap
                except Exception:
                    continue
        return 128

    def _drain_until(self, *, target_out: int, min_in: int = 1, max_ms: int = 50) -> tuple[int, int]:
        """
        Output FIFO를 target_out 이하로, Input FIFO를 min_in 이상으로 만들 때까지 가능한 한 빠르게 드레인.
        반환: (마지막 in, 마지막 out)
        """
        deadline = time.monotonic() + max_ms / 1000.0
        last_in = last_out = -1
        while time.monotonic() < deadline:
            snap = self._fifo_snapshot()
            last_in, last_out = int(snap.get("in", -1)), int(snap.get("out", -1))
            if (last_out >= 0 and last_out <= target_out) and ((last_in < 0) or (last_in >= min_in)):
                break
            # 새 버퍼가 있으면 0ms pop 반복 → 자동 재큐로 input 풀 보강
            if self._stream_has_output():
                try:
                    with Buffer(self.grabber, timeout=0):
                        pass
                    continue
                except Exception:
                    pass
            time.sleep(0)  # 양보
        return last_in, last_out

    def _start_backpressure_watchdog(self, period_s: float = 0.008) -> None:
        """
        프리엠프 역압 워치독 (최종판)
        - in==0&&out>=1 즉시 게이팅(OFF) + 집중 드레인
        - out이 용량의 75%를 넘으면 선제 게이팅(OFF) + 집중 드레인
        - 재개(ON)는 in>=SAFE 또는 announced>=SAFE일 때만 (히스테리시스)
        - 최후 안전망: 여전히 in=0 && out 높음 → 재할당+재시작
        """
        t = getattr(self, "_bp_wd_thr", None)
        if t and t.is_alive():
            return

        self._bp_wd_stop = threading.Event()
        self._bp_recover = threading.Lock()
        self._bp_last_fix = 0.0

        cap = self._output_fifo_capacity()
        OUT_PREEMPT_HI = max(8, int(cap * 3 / 4))  # 75%에서 선제 게이팅
        OUT_RELIEF_LO = max(2, int(cap * 1 / 8))  # 12.5% 이하로 낮출 때까지 집중 드레인
        IN_SAFE = 2  # 재개 허용 입력 수위
        EMERG_BUDGET = 32  # 스파이크 누를 때 pop 예산 상향
        PANIC_COOLDOWN = 0.8
        GATE_HOLD_MS = 10  # 최소 게이트 OFF 유지시간

        def _loop():
            trig_gated = False
            gate_off_until = 0.0
            in0_since = 0.0

            while not self._bp_wd_stop.is_set():
                try:
                    if not (self.connected and self.is_grabbing() and self.grabber):
                        time.sleep(period_s);
                        continue

                    snap = self._fifo_snapshot()
                    in_fifo = int(snap.get("in", -1))
                    out_fifo = int(snap.get("out", -1))
                    now = time.monotonic()

                    # 0) out 고수위 즉시 완화
                    if out_fifo >= OUT_PREEMPT_HI:
                        with suppress(Exception):
                            self._emergency_backpressure_relief(hwm=OUT_PREEMPT_HI, budget=EMERG_BUDGET)

                    # 1) 즉시 게이팅 트리거 (사전/사후)
                    need_gate_off = False
                    if in_fifo == 0 and out_fifo >= 1:
                        # 사후 감지 (이미 in=0)
                        if in0_since == 0.0:
                            in0_since = now
                        if not trig_gated and (now - in0_since) * 1000.0 >= 0:
                            need_gate_off = True
                    else:
                        in0_since = 0.0
                    # 사전 감지 (out이 75% 넘음)
                    if not trig_gated and out_fifo >= OUT_PREEMPT_HI:
                        need_gate_off = True

                    if need_gate_off:
                        with suppress(Exception):
                            self._gate_triggers(False)
                        trig_gated = True
                        gate_off_until = now + (GATE_HOLD_MS / 1000.0)

                        # 집중 드레인: out을 낮은 수위까지 끌어내리고 input을 보강
                        with suppress(Exception):
                            self._fill_input_pool()
                        with suppress(Exception):
                            self.ensure_stream_ready(min_input_fifo=1, refill_if_zero=True)
                        with suppress(Exception):
                            self._drain_until(target_out=OUT_RELIEF_LO, min_in=1, max_ms=50)

                    # 2) 게이트 OFF 상태 → 재개 조건 판단(히스테리시스)
                    if trig_gated and now >= gate_off_until:
                        snap2 = self._fifo_snapshot()
                        in2 = int(snap2.get("in", -1))
                        announced = self._announced_buf_count()
                        # 입력이 충분히 회복되었거나(out이 낮고 input이 안전 수위),
                        # in 정보를 모르는 빌드에서는 announced로 대체 판정
                        ok_to_resume = ((in2 >= IN_SAFE) or (in2 < 0 and announced >= IN_SAFE))
                        if ok_to_resume:
                            with suppress(Exception):
                                self._gate_triggers(True)
                            trig_gated = False

                    # 3) 최후 안전망
                    if in_fifo == 0 and out_fifo >= OUT_PREEMPT_HI:
                        if (now - getattr(self, "_bp_last_fix", 0.0)) >= PANIC_COOLDOWN:
                            with self._bp_recover:
                                with suppress(Exception): self.execute_command("AcquisitionStop", timeout=0.2)
                                with suppress(Exception): self.stop_grab(flush=True)
                                with suppress(Exception): self.wait_until_idle(timeout=0.8)

                                new_cnt = max(int(getattr(self, "_buf_target", 64)) * 2, 128)
                                self._buf_target = new_cnt
                                self._realloc_buffers_safe(new_cnt, 0, refill=True)

                                with suppress(Exception): self.grabber.start(); self.grabbing = True
                                with suppress(Exception): self._fill_input_pool()
                                with suppress(Exception): self.ensure_stream_ready(min_input_fifo=1,
                                                                                   refill_if_zero=True)

                                self._bp_last_fix = time.monotonic()
                except Exception:
                    pass

                time.sleep(period_s)

        th = threading.Thread(target=_loop, name=f"BPWatchdog-{self.cam_id}", daemon=True)
        self._bp_wd_thr = th
        th.start()

    def _stop_backpressure_watchdog(self) -> None:
        th = getattr(self, "_bp_wd_thr", None)
        ev = getattr(self, "_bp_wd_stop", None)
        if ev:
            with suppress(Exception): ev.set()
        if th and getattr(th, "is_alive", lambda: False)():
            with suppress(Exception): th.join(timeout=0.5)
        self._bp_wd_thr = None

    def stop_grab(self, *, flush: bool = True) -> None:
        """
        안정화된 정지 시퀀스:
          ① 세이프티 드레인/백프레셔 워치독 중지
          ② DataStream 정지(가능 시 DSStopFlags), AcquisitionStop, grabber.stop()
          ③ 필요 시 flush/revoke
          ④ 상태 플래그/로그
        """
        with self._lock:
            if not self.is_grabbing():
                self.logger.debug("[%s] Stop ignored: not currently grabbing.", self.cam_id)
                return

            # ① 보조 루프 먼저 중지
            with suppress(Exception):
                self._stop_safety_drain()
            with suppress(Exception):
                self._stop_backpressure_watchdog()

            self.logger.debug("[%s] Stopping acquisition (flush=%s)...", self.cam_id, flush)

            # ② 스트림 정지(DataStream stop flags 지원 시 우선)
            try:
                import egrabber as _eg
                ds_stop_flags = getattr(_eg, "DSStopFlags", None)
                stream = getattr(self.grabber, "stream", None)
                if ds_stop_flags and stream:
                    # flush=True면 즉시 중단(ASYNC_ABORT), 아니면 GRACEFUL
                    flags = ds_stop_flags.ASYNC_ABORT if flush else ds_stop_flags.ASYNC_GRACEFUL
                    stream.stop(flags)
            except Exception as exc:
                self.logger.warning("[%s] DataStream stop failed: %s", self.cam_id, exc, exc_info=False)

            # AcquisitionStop / grabber.stop
            with suppress(Exception):
                self.execute_command("AcquisitionStop", timeout=0.5)
            with suppress(Exception):
                self.grabber.stop()

            # ③ flush / revoke(필요 시)
            if flush:
                with suppress(Exception):
                    if hasattr(self.grabber, "flush_buffers"):
                        self.grabber.flush_buffers()
            with suppress(Exception):
                getattr(self.grabber, "revokeAllBuffers", lambda: None)()

            # 상태 플래그
            self.grabbing = False
            self.logger.info("[%s] Grabbing stopped (flush=%s).", self.cam_id, flush)

    def _bytes_per_pixel_from_pixfmt(self, pix_fmt: str) -> int:
        # unpacked 기준 근사. (packed 10/12bpp는 실제 피치가 달라질 수 있으나,
        # 많은 카메라가 16bit로 내보내므로 2바이트로 취급해도 안전한 편)
        m = {
            "Mono8": 1, "BayerRG8": 1, "BayerGB8": 1, "BayerGR8": 1, "BayerBG8": 1,
            "RGB8": 3, "BGR8": 3,
            "Mono10": 2, "Mono12": 2, "Mono14": 2, "Mono16": 2,
            "BayerRG10": 2, "BayerGB10": 2, "BayerGR10": 2, "BayerBG10": 2,
            "BayerRG12": 2, "BayerGB12": 2, "BayerGR12": 2, "BayerBG12": 2,
            "BayerRG16": 2,
        }
        return m.get((pix_fmt or "").strip(), 1)

    def _estimate_line_pitch_bytes(self) -> int:
        """
        라인 피치(바이트) 추정(최신값 보장):
        ① stream 후보 노드 → ② device 후보 노드 → ③ Width×Bpp(+64B 정렬)
        """
        # 스트림/디바이스 후보 노드
        s = getattr(self.grabber, "stream", None)
        d = getattr(self.grabber, "device", None)

        stream_candidates = ("PixelProcessorLinePitch", "LinePitch", "DeliveredLinePitch", "Pitch")
        device_candidates = ("PixelProcessorLinePitch", "LinePitch", "DeliveredLinePitch", "Pitch")

        for nm, cands in ((s, stream_candidates), (d, device_candidates)):
            if nm and hasattr(nm, "features"):
                feats = set()
                with suppress(Exception):
                    feats = set(nm.features())
                for n in cands:
                    if n in feats:
                        try:
                            val = int(nm.get(n))
                            if val > 0:
                                return val
                        except Exception:
                            pass

        # Fallback: Width×Bpp (+ 64B 정렬 권장)
        try:
            w = int(self.get_param("Width", force_reload=True))  # ← 캐시 무시
        except Exception:
            w = 0
        try:
            pf = str(self.get_param("PixelFormat", force_reload=True))
        except Exception:
            pf = "Mono8"
        bpp = self._bytes_per_pixel_from_pixfmt(pf)
        pitch = max(1, w * bpp)
        # 64바이트 정렬(많은 보드가 64B stride 사용)
        pitch = _align_up(pitch, 64)
        return pitch



    def _calc_dma_blocks_for_pitch(self, pitch_bytes: int) -> int:
        """
        1KB 블록 × N = 버퍼 파트 크기 를 만들 때,
        파트 크기가 라인 피치의 정수배가 되도록 N을 계산.
        N*1024 % pitch == 0 && N*1024 >= pitch
        """
        block = _DMA_BLOCK_BYTES
        # 기본: 한 줄을 담을 수 있을 만큼 확보
        n = (pitch_bytes + block - 1) // block
        # 정수배가 될 때까지 증가
        while (n * block) % pitch_bytes != 0:
            n += 1
        return n

    def _compute_part_bytes(self, pitch: int, *,
                            lines_hint: int = 32,
                            gran_bytes: int = 2048,
                            max_part_bytes: int = 1 * 1024 * 1024) -> int:
        """
        part_bytes 를 계산:
          • part_bytes 는 pitch 의 배수
          • part_bytes 는 gran_bytes(2KB) 의 배수 (보드가 2KB 블록으로 올림)
          • 기본 라인수(lines_hint)를 시작점으로, 필요 시 step으로 증가
        """
        # 먼저 최소 라인 수
        lines = max(1, int(lines_hint))

        # pitch*N 을 gran 로 올림했을 때도 pitch 의 배수 유지되게 만들기:
        # 조건: align_up(pitch*lines, gran) % pitch == 0
        # 이를 보장하려면 lines 가 (gran / gcd(gran, pitch)) 의 배수면 충분.
        step_lines = gran_bytes // math.gcd(gran_bytes, pitch)
        # lines 를 step_lines 배수로 상향
        if lines % step_lines != 0:
            lines = ((lines + step_lines - 1) // step_lines) * step_lines

        part = _align_up(pitch * lines, gran_bytes)

        # 상한 초과 시 줄이기 (그래도 최소 1step 유지)
        if part > max_part_bytes:
            # 가능한 최대 라인수로 재계산
            max_lines = max(step_lines, (max_part_bytes // pitch) // step_lines * step_lines)
            part = _align_up(pitch * max_lines, gran_bytes)

        return max(pitch, part)  # 최소 한 줄은 담기

    def _realloc_buffers_safe(self, count: int, height: int = 0, *, refill: bool = True) -> None:
        """
        안전 재할당:
          • 완전 정지 → revoke/flush
          • 두 번째 인자 = '버퍼 파트 크기(바이트)'. 보드의 2KB 블록/올림을 고려해 산출.
          • height>0 가 들어오면 'lines_hint'로 간주(보수적으로 반영).
        """
        if not getattr(self, "_realloc_lock", None):
            import threading
            self._realloc_lock = threading.Lock()

        with self._realloc_lock:
            with suppress(Exception):
                self._stop_safety_drain()
            if self.is_grabbing():
                with suppress(Exception): self.execute_command("AcquisitionStop", timeout=0.3)
                with suppress(Exception): self.stop_grab(flush=True)
                with suppress(Exception): self.wait_until_idle(timeout=1.0)

            g = getattr(self, "grabber", None)
            if not g:
                raise GrabberError("grabber is None")

            with suppress(Exception):
                getattr(g, "revokeAllBuffers", lambda: None)()
            with suppress(Exception):
                getattr(g, "flush_buffers", lambda: None)()

            # ── 최신 피치 산출
            pitch = int(self._estimate_line_pitch_bytes())

            # granularity = 2048B 로 가정 (로그가 2048*… 로 찍힘)
            gran = 2048
            lines_hint = int(height) if int(height) > 0 else 32
            part_bytes = self._compute_part_bytes(pitch, lines_hint=lines_hint, gran_bytes=gran)

            # 방어적 체크: 드라이버가 추가로 올려도 여전히 pitch 배수 보장되도록 한 번 더 보정
            # (align_up 후 pitch로 나눠떨어지지 않으면 step_lines 만큼 라인수 추가)
            if (part_bytes % pitch) != 0:
                step_lines = gran // math.gcd(gran, pitch)
                part_bytes = self._compute_part_bytes(pitch,
                                                      lines_hint=lines_hint + step_lines,
                                                      gran_bytes=gran)

            g.realloc_buffers(int(count), int(part_bytes))
            self.logger.info("[%s] realloc_buffers: count=%d, pitch=%d, gran=%d, part_bytes=%d",
                             self.cam_id, count, pitch, gran, part_bytes)

            time.sleep(0.05)
            if refill:
                with suppress(Exception): self._fill_input_pool()
                with suppress(Exception): self.ensure_stream_ready(min_input_fifo=1, refill_if_zero=True)

    def _flush_one_buffer(self) -> None:
        """
        Output FIFO에 새 버퍼가 있을 때만 1개 pop하여 드레인.
        ★ 빈 큐면 pop 자체를 하지 않아 -1011 타임아웃 스팸을 원천 차단한다.
        """
        try:
            if not self._stream_has_output():
                return
            # timeout=0 → 즉시 반환, 버퍼 없으면 TimeoutException (위에서 이미 가드)
            with Buffer(self.grabber, timeout=0) as _:
                pass
        except (TimeoutException, AttributeError):
            pass

    def _is_safety_drain_running(self) -> bool:
        t = getattr(self, "_drain_thr", None)
        try:
            return bool(t and t.is_alive())
        except Exception:
            return False

    # ───────────────────── Stream Output readiness probe (신규) ─────────────────────
    def _stream_has_output(self) -> bool:
        s = getattr(self.grabber, "stream", None)
        if not s or not hasattr(s, "get_info"):
            return False
        try:
            # 여러 키로 '읽기'만 한다. 능동 pop은 하지 않는다.
            for k in ("OutputFifo", "NewBufferEventUnreadCount", "NewBuffer"):
                try:
                    if int(s.get_info(k)) > 0:
                        return True
                except Exception:
                    continue
            return False
        except Exception:
            return False

    # CameraController 내부 (기존 메서드들 인근에 추가)
    def _get_info_multi(self, module, keys: tuple[str, ...], default: int = -1) -> int:
        """
        GenTL get_info 키가 빌드마다 다른 문제를 우회하기 위한 다중 키 조회.
        어느 한 키라도 성공하면 정수로 반환, 전부 실패하면 default.
        """
        if not module or not hasattr(module, "get_info"):
            return default
        for k in keys:
            try:
                return int(module.get_info(k))
            except Exception:
                continue
        return default

    def _fifo_snapshot(self) -> dict:
        snap = {"in": -1, "out": -1, "queued": -1}
        s = getattr(self.grabber, "stream", None)
        if not s or not hasattr(s, "get_info"):
            return snap
        with suppress(Exception): snap["in"] = int(s.get_info("InputFifo"))
        with suppress(Exception): snap["out"] = int(
            self._get_info_multi(s, ("OutputFifo", "NewBufferEventUnreadCount", "NewBuffer"), -1))
        with suppress(Exception): snap["queued"] = int(s.get_info("QueuedBuf"))
        return snap

    def _emergency_backpressure_relief(self, *, hwm: int = 24, budget: int = 8) -> bool:
        s = getattr(self.grabber, "stream", None)
        if not s or not hasattr(s, "get_info"):
            return False
        try:
            out = self._get_info_multi(s, ("OutputFifo", "NewBufferEventUnreadCount", "NewBuffer"), -1)
        except Exception:
            out = -1

        if out >= 0 and out < hwm:
            return False

        # ★ out 정도에 따라 예산을 동적으로 확대(최대 64)
        dyn_budget = max(budget, min(64, (0 if out < 0 else (out - hwm + 4))))
        did = 0
        while did < int(dyn_budget):
            try:
                with Buffer(self.grabber, timeout=0):
                    did += 1
            except TimeoutException:
                break
            except Exception:
                break

        if did:
            self.logger.warning("[%s] Emergency drain %d (out=%s, hwm=%d, dyn=%d)",
                                self.cam_id, did, out, hwm, dyn_budget)
        return bool(did)

    def get_next_frame(
            self,
            timeout_ms: int = 3000,
            *,
            count_timeout_error: bool = True,
    ) -> Optional[np.ndarray]:
        """
        안정화된 프레임 획득 (최종판)
          • Quiet 윈도우: 평소엔 pop 금지, 단 출력 FIFO가 고수위면 '비상 드레인'으로 역압 해소
          • 빈 큐면 pop 시도 자체 금지(드라이버 -1011 로그 원천 차단)
          • GC_ERR_TIMEOUT 시 입력 풀 보강 + 짧은 백오프 재시도
          • 성공 시 통계/시그널 업데이트
        Returns:
          np.ndarray | None (Quiet/재시도 끝 → 프레임 없으면 None)
        """
        # ── 사전 조건 ─────────────────────────────────────────────
        if not self.is_grabbing():
            raise GrabberNotActiveError(f"[{self.cam_id}] Cannot get frame: not grabbing.")

        # Quiet 종료 시점 복구(있으면) – 트리거/획득 원복
        with suppress(Exception):
            if hasattr(self, "_maybe_leave_quiet"):
                self._maybe_leave_quiet()

        # Quiet 윈도우: 일반 pop 중지하되, Output FIFO가 고수위면 비상 드레인으로 역압 해소
        if self._is_quiet():
            with suppress(Exception):
                if hasattr(self, "_emergency_backpressure_relief"):
                    # out ≥ 24이면 최대 8개만 즉시 드레인하여 입력 풀 회복
                    self._emergency_backpressure_relief(hwm=24, budget=8)
            time.sleep(min(0.005, max(0, timeout_ms) / 1000.0))
            return None

        # ── 내부 변환기: Buffer → NumPy ──────────────────────────
        def _buffer_to_numpy(buf) -> np.ndarray:
            from ctypes import POINTER, c_ubyte, cast

            ptr = buf.get_info(BUFFER_INFO_BASE, INFO_DATATYPE_PTR)
            w = int(buf.get_info(BUFFER_INFO_WIDTH, INFO_DATATYPE_SIZET))
            h = int(buf.get_info(BUFFER_INFO_DELIVERED_IMAGEHEIGHT, INFO_DATATYPE_SIZET))
            sz = int(buf.get_info(BUFFER_INFO_DATA_SIZE, INFO_DATATYPE_SIZET))

            # DeliveredWidth 우선
            try:
                import egrabber as _eg
                DIW = getattr(_eg, "BUFFER_INFO_DELIVERED_IMAGEWIDTH", None)
                if DIW is not None:
                    dw = int(buf.get_info(DIW, INFO_DATATYPE_SIZET))
                    if dw > 0: w = dw
            except Exception:
                pass

            if not all((ptr, w, h)) or sz <= 0 or (w * h) == 0:
                raise FrameAcquisitionError(
                    f"Invalid buffer metadata: ptr={ptr}, w={w}, h={h}, sz={sz}"
                )

            view = cast(ptr, POINTER(c_ubyte * sz)).contents
            buf_u8 = np.frombuffer(view, dtype=np.uint8, count=sz)

            # 픽셀포맷 추정 (기본 Mono8)
            pix_fmt = (self.get_param("PixelFormat", force_reload=False) or "Mono8").strip()
            pf_map = {
                "Mono8": (np.uint8, 1), "Mono10": (np.uint16, 1), "Mono12": (np.uint16, 1),
                "Mono16": (np.uint16, 1),
                "BayerRG8": (np.uint8, 1), "BayerGB8": (np.uint8, 1),
                "BayerGR8": (np.uint8, 1), "BayerBG8": (np.uint8, 1),
                "RGB8": (np.uint8, 3), "BGR8": (np.uint8, 3),
                "BayerRG10": (np.uint16, 1), "BayerGB10": (np.uint16, 1),
                "BayerGR10": (np.uint16, 1), "BayerBG10": (np.uint16, 1),
                "BayerRG12": (np.uint16, 1), "BayerGB12": (np.uint16, 1),
                "BayerGR12": (np.uint16, 1), "BayerBG12": (np.uint16, 1),
                "Mono14": (np.uint16, 1), "BayerRG16": (np.uint16, 1),
            }
            dt, ch = pf_map.get(pix_fmt, (np.uint8, 1))

            row_bytes_guess = max(sz // h, 1)
            need_per_row = w * dt().itemsize * ch
            row_bytes = row_bytes_guess

            # stride 부족 관대 처리(need의 5% 또는 64bytes 허용)
            if row_bytes < need_per_row:
                deficit = need_per_row - row_bytes
                allow = min(64, int(need_per_row * 0.05))
                if deficit > allow:
                    raise FrameAcquisitionError(
                        f"Row bytes too small: row={row_bytes}, need={need_per_row}"
                    )

            # 리쉐이프 경로
            if ch == 1 and dt == np.uint8:
                rows = buf_u8[:row_bytes * h].reshape(h, row_bytes)
                line = rows[:, :min(row_bytes, need_per_row)]
                if line.shape[1] < need_per_row:
                    out = np.zeros((h, w), dtype=np.uint8)
                    filled = min(line.shape[1], w)
                    out[:, :filled] = line[:, :filled]
                    return out
                return line[:, :w].copy()

            if ch == 1 and dt == np.uint16:
                row_even = (row_bytes // 2) * 2
                used = min(row_even, need_per_row)
                buf_even = buf_u8[:used * h]
                row_pix = used // 2
                frame16 = np.frombuffer(buf_even, dtype=np.uint16).reshape(h, row_pix)
                if row_pix < w:
                    out = np.zeros((h, w), dtype=np.uint16)
                    out[:, :row_pix] = frame16
                    return out
                return frame16[:, :w].copy()

            if ch == 3 and dt == np.uint8:
                rows = buf_u8[:row_bytes * h].reshape(h, row_bytes)
                used = min(row_bytes, need_per_row)
                rgb_bytes = rows[:, :used]
                w_used = used // 3
                rgb = rgb_bytes.reshape(h, w_used, 3)
                if w_used < w:
                    out = np.zeros((h, w, 3), dtype=np.uint8)
                    out[:, :w_used, :] = rgb
                    rgb = out
                if "BGR" in pix_fmt:
                    rgb = rgb[..., ::-1]
                return rgb.copy()

            # fallback: 8bit gray
            rows = buf_u8[:row_bytes * h].reshape(h, row_bytes)
            line = rows[:, :min(row_bytes, w)].copy()
            if line.shape[1] < w:
                out = np.zeros((h, w), dtype=np.uint8)
                out[:, :line.shape[1]] = line
                line = out
            return line

        # ── Safety-drain 일시정지 ────────────────────────────────
        drain_was_running = False
        try:
            drain_was_running = self._is_safety_drain_running()
            if drain_was_running:
                with suppress(Exception):
                    self._stop_safety_drain()

            # ----------- Buffer pop (빈 큐 프리체크 + 최대 3회 재시도) ----------
            retries = 3
            backoff_ms = max(1, min(10, timeout_ms // 10 or 1))  # 1~10ms
            local_to = max(0, int(timeout_ms))

            while True:
                # ★ 빈 큐면 pop 자체를 생략하여 -1011 로그 원천 차단
                try:
                    if local_to <= 10 and not self._stream_has_output():
                        time.sleep(backoff_ms / 1000.0)
                        retries -= 1
                        if retries <= 0:
                            return None
                        continue
                except Exception:
                    # 상태 확인 자체가 실패하면 일단 시도는 해보되, 타임아웃 시 복구 루틴으로
                    pass

                try:
                    with self._lock:
                        with Buffer(self.grabber, timeout=local_to) as buf:
                            frame = _buffer_to_numpy(buf)
                    # 성공
                    break

                except TimeoutException:
                    if count_timeout_error:
                        self.stats["error_count"] += 1
                    self.logger.debug("[%s] pop timeout (%d ms). refill+backoff (retries=%d)",
                                      self.cam_id, local_to, retries)

                    # 입력 풀/스트림 상태 보강 + 디버그 스냅샷
                    with suppress(Exception):
                        diag = self.ensure_stream_ready(min_input_fifo=1, refill_if_zero=True)
                        fi, fo, an = (-1, -1, -1)
                        if hasattr(self, "_read_stream_triplet"):
                            fi, fo, an = self._read_stream_triplet()
                        self.logger.debug("[%s] ensure_stream_ready → %s (in=%s, out=%s, announced=%s)",
                                          self.cam_id, diag, fi, fo, an)

                    retries -= 1
                    if retries <= 0:
                        return None
                    time.sleep(backoff_ms / 1000.0)
                    local_to = max(local_to, 15)

            # ----------- 성공 처리 ---------------------------------------------
            self.set_last_np_frame(frame)
            self.stats["frame_count"] += 1
            self.stats["last_frame_time"] = time.time()
            self._update_stats()
            with suppress(Exception):
                self.reshaped_frame_ready.emit(frame)
                self.frame_ready.emit(frame)
            return frame

        except GenTLException as e:
            raise FrameAcquisitionError(f"[{self.cam_id}] GenTL error during frame acquisition: {e}") from e
        except Exception as e:
            raise FrameAcquisitionError(f"[{self.cam_id}] Unexpected error processing frame: {e}") from e
        finally:
            # Safety drain 재개
            if drain_was_running:
                with suppress(Exception):
                    self._start_safety_drain()

    # --- END   OF PATCH src/core/camera_controller.py ----------------------------

    def run_synchronized_trigger_test(
            self,
            expected_triggers: int,
            poll_interval: float = 2.0,
    ):
        """
        멀티-카메라 HW-Trigger 동기화 장기 시험.
        (1) 모든 컨트롤러 CIC 카운터 리셋
        (2) Leader 가 StartCycle -or- 대체 명령 반복 실행
        (3) 누락 트리거 감시
        """
        from src.core.camera_exceptions import GrabberError, CommandExecutionError

        self.logger.info("[%s] Synchronized trigger test → %d cycles", self.cam_id, expected_triggers)

        # ① 풀에서 연결된 컨트롤러 수집
        from src.core import controller_pool
        ctrls = [c for c in controller_pool.controllers.values() if c.is_connected()]
        if not ctrls:
            raise GrabberError("No connected controllers found.")

        # ② CIC 카운터 reset
        for c in ctrls:
            dev = getattr(c.grabber, "device", None)
            if dev and "CycleLostTriggerCountReset" in dev.features():
                with suppress(Exception):
                    dev.execute("CycleLostTriggerCountReset")
            self.logger.debug("[%s] CIC reset done", c.cam_id)

        # ③ Leader 의 grabber.device 확보
        dev = getattr(self.grabber, "device", None)
        if dev is None:
            # 리더가 device 모듈을 못 가진 경우 → 첫 번째로 가진 컨트롤러를 리더로 교체
            for c in ctrls:
                dev = getattr(c.grabber, "device", None)
                if dev:
                    self.logger.info("[%s] Delegating leader role to %s", self.cam_id, c.cam_id)
                    break
        if dev is None:
            raise CommandExecutionError("No grabber.device owns a Cycle generator.")

        # ④ StartCycle 계열 명령 후보
        start_cmds = [n for n in ("StartCycle", "CycleGeneratorStart", "Start") if n in dev.features()]
        if not start_cmds:
            raise CommandExecutionError("Neither StartCycle nor compatible command found in Device-module.")

        trigger_count = 0
        last_log = time.time()

        while trigger_count < expected_triggers:
            # 1) 트리거 전송 (첫 번째 성공하는 명령 사용)
            for cmd in start_cmds:
                try:
                    dev.execute(cmd)
                    break
                except Exception:
                    continue
            else:
                raise CommandExecutionError("Failed to execute any cycle-start command.")

            trigger_count += 1
            time.sleep(0.05)  # 카메라 수신 여유

            # 2) 주기적 상태 확인
            if time.time() - last_log >= poll_interval or trigger_count == expected_triggers:
                total_lost = 0
                status_msg = []
                for c in ctrls:
                    lost = 0
                    dev_c = getattr(c.grabber, "device", None)
                    if dev_c and "CycleLostTriggerCount" in dev_c.features():
                        lost = int(dev_c.get("CycleLostTriggerCount"))
                    total_lost += lost
                    status_msg.append(f"{c.cam_id}: lost={lost}")

                self.logger.info("Progress %d/%d – %s", trigger_count, expected_triggers, "; ".join(status_msg))
                if total_lost:
                    raise GrabberError(f"Trigger lost! total={total_lost}")
                last_log = time.time()

        self.logger.info("Synchronized trigger test PASSED – %d cycles, 0 lost", expected_triggers)

    # --- END   OF PATCH camera_controller.py --------------------------------------

    # ────────────────────────────── Statistics ───────────────────────────
    def _update_stats(self) -> None:
        now = time.time()
        if now - self._last_stats_update < self._stats_interval:
            return
        start = self.stats["start_time"]
        if start > 0:
            elapsed = now - start
            self.stats["fps"] = self.stats["frame_count"] / elapsed if elapsed else 0.0
            total = self.stats["frame_count"] + self.stats["error_count"]
            self.stats["frame_loss"] = self.stats["error_count"] / total if total else 0.0
        self._last_stats_update = now
        self.logger.debug("[%s] Stats: fps=%.2f loss=%.2f%%",
                          self.cam_id, self.stats["fps"],
                          self.stats["frame_loss"] * 100)

    def export_stats(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Export camera statistics as JSON or return as dict.
        """
        with self._lock:  # Thread-safe
            self._update_stats()
            stats_data = {
                "cam_id": self.cam_id,
                "serial_number": self.serial_number,
                "timestamp": datetime.utcnow().isoformat(),
                "connected": self.connected,
                "grabbing": self.grabbing,
                "stats": self.stats.copy()
            }
            if output_path:
                try:
                    with open(output_path, "w") as f:
                        json.dump(stats_data, f, indent=2)
                    self.logger.info(f"[{self.cam_id}] Stats exported to {output_path}")
                except Exception as e:
                    self.logger.error(f"[{self.cam_id}] Failed to export stats: {e}")
            return stats_data

    # ------------------------------------------------------------------
    #  노드맵 탐색 유틸 – camera-remote → interface → device → remote → stream → system
    # ------------------------------------------------------------------
    def _find_nodemap(self, node_name: str):
        """
        Camera-remote(self.params)를 시작으로

            Interface → Device → Remote → Stream → System

        순서로 돌며 `features()` 안에 *node_name* 이 있는 첫 nodemap을
        돌려준다.  없으면 **None**.
        """
        if not self.connected:
            return None

        nm_candidates = [self.params]

        # Grabber가 제공하는 여러 nodemap 속성
        if self.grabber:
            for attr in ("interface", "device", "remote", "stream", "system"):
                nm = getattr(self.grabber, attr, None)
                if nm and nm not in nm_candidates:
                    nm_candidates.append(nm)

        for nm in nm_candidates:
            try:
                if node_name in nm.features():
                    return nm
            except Exception:
                # 일부 nodemap 은 features() 를 안 가질 수도 있다
                continue
        return None



    def set(self, name: str, value: Any, *, verify: bool = True) -> None:
        if not self.connected:
            raise CameraNotConnectedError(f"[{self.cam_id}] not connected")

        node_maps = [self.params, getattr(self.grabber, "remote", None)]

        def _write(nm):
            if hasattr(nm, "getEnum") and isinstance(value, str):
                try:
                    if value not in nm.getEnum(name):
                        raise ParameterError(f"{name} unsupported '{value}' (enum)")
                except Exception:
                    pass
            nm.set(name, value)
            if verify:
                with suppress(Exception):
                    if str(nm.get(name)) != str(value):
                        raise ParameterError(f"{name} verify failed")

        for nm in node_maps:
            if nm and name in nm.features():
                _write(nm)
                self.logger.info("[%s] %s ← %s (%s)",
                                 self.cam_id, name, value,
                                 "camera" if nm is self.params else "grabber")
                return

        # Legacy SFNC-map 보정
        for alt_name, alt_val in LEGACY_MAP.get(name, []):
            for nm in node_maps:
                if nm and alt_name in nm.features():
                    nm.set(alt_name, alt_val)
                    self.logger.debug("[%s] LEGACY_MAP %s → %s=%s",
                                      self.cam_id, name, alt_name, alt_val)
                    return
        raise ParameterError(f"{name} not available in camera/grabber nodemap")

    def grab_single_frame(self, timeout_ms: int = 1000) -> np.ndarray:  # noqa: D401
        """
        SequenceRunner·액션 모듈 호환용 thin-wrapper.
        Live-view 신호(`frame_ready`)도 그대로 전파됩니다.
        """
        return self.get_next_frame(timeout_ms=timeout_ms)

    def start_cycle_when_safe(self,
                              *,
                              min_in: int = 2,
                              max_out_frac: float = 0.125,  # 용량의 12.5% 이하일 때만
                              wait_ms: int = 200) -> bool:
        """
        안전 조건을 만족하면 CIC를 시작(게이트 ON).
        - InputFifo >= min_in  또는 (Unknown 이면 announced>=min_in)
        - OutputFifo <= capacity*max_out_frac
        - 조건이 안되면 조건 성립 또는 timeout 전까지 drain/refill 시도
        """
        dev = getattr(self.grabber, "device", None)
        if dev is None:
            return False

        cap = self._output_fifo_capacity()
        out_lo = max(2, int(cap * max_out_frac))
        deadline = time.monotonic() + (wait_ms / 1000.0)

        # 워치독이 켜져있다면 충돌 방지용으로 일단 게이트 OFF 유지 상태에서 정리
        with suppress(Exception):
            self._gate_triggers(False)

        while time.monotonic() < deadline:
            # out 낮추고 in 보강
            with suppress(Exception):
                self._fill_input_pool()
            with suppress(Exception):
                self.ensure_stream_ready(min_input_fifo=1, refill_if_zero=True)
            with suppress(Exception):
                self._drain_until(target_out=out_lo, min_in=1, max_ms=20)

            # 상태 재평가
            snap = self._fifo_snapshot()
            in_fifo, out_fifo = int(snap.get("in", -1)), int(snap.get("out", -1))
            announced = self._announced_buf_count()

            in_ok = (in_fifo >= min_in) or (in_fifo < 0 and announced >= min_in)
            out_ok = (out_fifo >= 0 and out_fifo <= out_lo) or (out_fifo < 0)  # out을 모르겠으면 일단 허용

            if in_ok and out_ok:
                # 게이트 ON → StartCycle 또는 CycleTriggerSource Enable
                src_sym = "Immediate"  # 필요에 따라 "StartCycle"과 조합
                with suppress(Exception):
                    if "CycleTriggerSource" in dev.features():
                        dev.set("CycleTriggerSource", src_sym)
                with suppress(Exception):
                    if "StartCycle" in dev.features():
                        dev.execute("StartCycle")

                # 카메라 TriggerMode ON (워치독 게이트-ON과 일치)
                with suppress(Exception):
                    if self.params and "TriggerMode" in self.params.features():
                        self.params.set("TriggerMode", "On")

                self.logger.info("[%s] CIC safely started (in=%s, out=%s, cap=%d)", self.cam_id, in_fifo, out_fifo, cap)
                return True

            time.sleep(0.005)

        self.logger.warning("[%s] start_cycle_when_safe: timed out (in=%s, out=%s, cap=%d)",
                            self.cam_id, in_fifo, out_fifo, cap)
        return False

    def configure_grabber_cycle_trigger(
            self, *,
            trigger_source: str = "Immediate",
            cycle_period_us: float = 3360.0,  # 3.36 ms (Min 이상으로 클램프)
            dlt_index: int = 1,
            arm_only: bool = True,  # ← 설정만(arm), 자동 시작 금지
    ) -> None:
        """
        CIC(내부 트리거) 설정만 수행하고, '시작(StartCycle)'은 절대 하지 않는다.
        - 항상 Disabled로 먼저 잠그고 난 뒤 나머지 노드를 세팅(사전 게이팅)
        - period는 보드 Min 이상으로 클램프
        - DLT 매핑은 그대로
        """
        dev = getattr(self.grabber, "device", None)
        if dev is None:
            raise AttributeError("Device-module (grabber.device) unavailable for cycle-trigger.")

        feats = dev.features()
        required_nodes_cycle = {"CycleGeneratorSelector", "CycleTriggerSource",
                                "CycleTriggerActivation", "CyclePeriodUs"}
        if not required_nodes_cycle.issubset(feats):
            raise ParameterError("CycleGenerator related nodes are missing from grabber.device.")

        # 0) 반드시 먼저 'Disabled'로 잠근다(사전 게이팅)
        try:
            dev.set("CycleTriggerSource", "Disabled")
        except Exception:
            pass

        # 1) 보드 Min 이상으로 주기 클램프
        applied = float(cycle_period_us)
        try:
            if query:
                qmin = query.info("CyclePeriodUs", "Min")
                minv = dev.get(qmin, float) if qmin else None
                if isinstance(minv, (int, float)):
                    applied = max(applied, float(minv))
        except Exception:
            pass

        # 2) 코어 파라미터 설정
        dev.set("CycleGeneratorSelector", 0)
        dev.set("CycleTriggerActivation", "RisingEdge")
        dev.set("CyclePeriodUs", applied)

        # DLT 매핑
        required_nodes_dlt = {"DeviceLinkTriggerToolSelector", "DeviceLinkTriggerToolSource"}
        if not required_nodes_dlt.issubset(feats):
            raise ParameterError("DeviceLinkTriggerTool related nodes are missing from grabber.device.")

        dev.set("DeviceLinkTriggerToolSelector", int(dlt_index))
        dev.set("DeviceLinkTriggerToolSource", "CycleGenerator0")

        # 3) 'arm_only'면 Disabled 상태 유지. (시작은 별도 API에서 안전조건 하에만)
        #    만약 arm_only=False로 주더라도 여기서 즉시 Enable 하지 말고,
        #    안전 시작 함수에서 재개하도록 강제.
        self.logger.info("[%s] CIC armed (source kept Disabled), period=%.3f µs, DLT=%d",
                         self.cam_id, applied, dlt_index)

    # ------------------------------------------------------------------ #
    #  Parameter setter (액션·API가 호출하는 공개 메서드)
    # ------------------------------------------------------------------ #
    def set_param(self, node_name: str, value: Any) -> None:
        """
        Public wrapper for setting parameters with safety checks.
        """
        with self._lock:
            self._check_connected()

            # For critical parameters that affect image size/format,
            # we must stop the grab if not in a persistent live-view session.
            critical_params = {"width", "height", "offsetx", "offsety", "pixelformat", "binninghorizontal", "binningvertical"}
            if node_name.lower() in critical_params and self.is_grabbing() and not self._live_view:
                self.logger.info("[%s] [Safety] Stopping grab before changing '%s'", self.cam_id, node_name)
                try:
                    self.stop_grab(flush=True)
                    time.sleep(0.1)
                except Exception as e:
                    self.logger.warning("[%s] Failed to stop grab for parameter change: %s", self.cam_id, e)

            self.set(node_name, value, verify=False)

            if self.enable_param_cache:
                self._param_cache[node_name] = value
                self._param_cache_timestamp[node_name] = time.time()

            self.logger.info("[%s] Param '%s' ← %s", self.cam_id, node_name, value)
    # ------------------------------------------------------------------
    #  안전한 파라미터 읽기
    # ------------------------------------------------------------------
    def get_param(self, node_name: str, force_reload: bool = False):
        """
        • `_find_nodemap` 으로 실제 노드 위치를 찾아 읽는다.
        • 캐시 사용 여부/갱신 로직은 기존과 동일.
        """
        with self._lock:
            self._check_connected()

            # ── 캐시 확인 ───────────────────────────────────────────
            if self.enable_param_cache and not force_reload:
                cached = self._param_cache.get(node_name)
                age    = time.time() - self._param_cache_timestamp.get(node_name, 0)
                if cached is not None and age < self.cache_timeout_sec:
                    return cached

            # ── 실제 nodemap 검색 ─────────────────────────────────
            nm = self._find_nodemap(node_name)
            if nm is None:
                raise ParameterError(f"[{self.cam_id}] Node '{node_name}' not found in any nodemap")

            try:
                val = nm.get(node_name)

                if self.enable_param_cache:
                    self._param_cache[node_name] = val
                    self._param_cache_timestamp[node_name] = time.time()

                return val

            except (GenTLException, TypeError) as e:
                raise ParameterError(
                    f"[{self.cam_id}] Failed to read '{node_name}': {e}"
                ) from e

    def get_device_vendor_name(self) -> str:
        with self._lock:
            try:
                # 수정: getattr로 get_info 안전하게 접근
                vendor = getattr(self.grabber, "get_info", lambda key: None)("DeviceVendorName")
                self.logger.info(f"[{self.cam_id}] Device vendor: {vendor}")
                return vendor if vendor is not None else "N/A"
            except Exception as e:
                self.logger.warning(f"[{self.cam_id}] Failed to get device vendor name: {e}")
                return "N/A"

    def get_device_model_name(self) -> str:
        with self._lock:
            try:
                # 수정: getattr로 get_info 안전하게 접근
                model = getattr(self.grabber, "get_info", lambda key: None)("DeviceModelName")
                self.logger.info(f"[{self.cam_id}] Device model: {model}")
                return model if model is not None else "N/A"
            except Exception as e:
                self.logger.warning(f"[{self.cam_id}] Failed to get device model name: {e}")
                return "N/A"

    def get_device_serial_number(self) -> str:
        with self._lock:
            # ① Grabber info → ② 원격 노드 순으로 시도
            serial = getattr(self.grabber, "get_info", lambda k: None)("DeviceSerialNumber")
            if not serial and self.params:
                try:
                    serial = self.params.get("DeviceSerialNumber")
                except Exception:
                    pass
            return serial or "N/A"

    # ────────────────────────────── 수정 ①
    def is_connected(self) -> bool:
        """스레드 안전 + 즉시 값 반환(락 불필요)."""
        return self.connected

    def _check_connected(self) -> None:
        if not self.connected:
            raise CameraNotConnectedError(f"[{self.cam_id}] Camera not connected.")

    def list_all_features(self, only_available: bool = True) -> List[str]:
        with self._lock:  # Thread-safe
            self._check_connected()
            if not self.params:
                self.logger.warning(f"[{self.cam_id}] No params object => cannot list features")
                return []
            try:
                feats = self.params.features(available_only=only_available)
                return feats
            except GenTLException as e:
                self.logger.error(f"[{self.cam_id}] Failed to list features: {e}", exc_info=True)
                return []
            except Exception as e:
                self.logger.error(f"[{self.cam_id}] Unexpected error listing features: {e}", exc_info=True)
                return []

    def get_enumeration_entries(self, node_name: str) -> List[str]:
        with self._lock:  # Thread-safe
            self._check_connected()
            if not self.params or not query:
                self.logger.warning(f"[{self.cam_id}] No params or query module => cannot get enum for {node_name}")
                return []
            try:
                q_enum = query.enum_entries(node_name, available_only=True)
                entries = self.params.get(q_enum, list)
                self.logger.debug(f"[{self.cam_id}] [ENUM] {node_name} => {entries}")
                return entries if entries is not None else []
            except GenTLException as e:
                self.logger.debug(f"[{self.cam_id}] get_enumeration_entries({node_name}) failed: {e}")
                return []
            except Exception as e:
                self.logger.warning(f"[{self.cam_id}] Unexpected error in get_enumeration_entries({node_name}): {e}")
                return []

    # ──────────────────────────────────────────────
    # Device-module 파라미터 읽기 / 쓰기 / Command
    # ──────────────────────────────────────────────
    def set_device_param(self, node_name: str, value) -> None:
        """
        Grabber(Device-module) GenICam 노드에 값을 쓰는 헬퍼.
        """
        dev = getattr(self.grabber, "device", None)
        if dev is None:
            raise AttributeError("grabber.device is None (Device-module not supported?)")

        if node_name not in dev.features():
            raise ParameterError(f"Node '{node_name}' not found in Device-module.")

        dev.set(node_name, value)
        self.logger.info(f"[{self.cam_id}] [Device] {node_name} ← {value}")

    def get_device_param(self, node_name: str):
        """
        Grabber(Device-module) 노드 값을 읽는 헬퍼.
        """
        dev = getattr(self.grabber, "device", None)
        if dev is None:
            raise AttributeError("grabber.device is None (Device-module not supported?)")

        if node_name not in dev.features():
            raise ParameterError(f"Node '{node_name}' not found in Device-module.")

        val = dev.get(node_name)
        self.logger.debug(f"[{self.cam_id}] [Device] {node_name} ⇒ {val}")
        return val

    def trigger_software_safe(self, *, cxp_id: int = 0) -> None:
        self._check_connected()
        original = self.params.get("TriggerSource")
        try:
            if original != "Software":
                self.params.set("TriggerSource", "Software")
            self.execute_software_trigger(cxp_id=cxp_id)
            self.logger.info("[%s] Software trigger sent (safe)", self.cam_id)
        finally:
            if original != "Software":
                with suppress(Exception):
                    self.params.set("TriggerSource", original)

    def execute_device_command(self, node_name: str, timeout: float = 0.5) -> None:
        """
        Device-module에 존재하는 *Command* 노드를 실행한다.
        """
        dev = getattr(self.grabber, "device", None)
        if dev is None:
            raise AttributeError("grabber.device is None (Device-module not supported?)")

        node = dev.getNode(node_name)
        if not (node and node.isValid() and node.isWritable()):
            raise CommandExecutionError(f"Device command '{node_name}' not found or not writable.")

        node.execute()
        if timeout > 0 and hasattr(node, 'wait_until_done'):
            node.wait_until_done(timeout)

        self.logger.info(f"[{self.cam_id}] Device-command '{node_name}' executed.")

    def is_writeable(self, node_name: str) -> bool:
        with self._lock:  # Thread-safe
            self._check_connected()
            if not self.params or not query:
                self.logger.debug(f"[{self.cam_id}] Cannot check writeable for {node_name}: No params or query")
                return False
            try:
                q_ = query.writeable(node_name)
                return bool(self.params.get(q_, bool))
            except GenTLException as e:
                self.logger.debug(f"[{self.cam_id}] is_writeable({node_name}) failed: {e}")
                return False
            except Exception as e:
                self.logger.warning(f"[{self.cam_id}] Unexpected error in is_writeable({node_name}): {e}")
                return False

    def get_parameter_metadata(self, name: str) -> Dict[str, Any]:
        with self._lock:  # Thread-safe
            self._check_connected()
            if not self.params:
                raise ParameterError(f"[{self.cam_id}] Camera parameters not initialized.")
            metadata = {}
            try:
                value = self.params.get(name)
                metadata["value"] = value
                self.logger.debug(f"[{self.cam_id}] Metadata for {name}: value={value}")
            except Exception as e:
                metadata["value"] = None
                metadata["error"] = f"Failed to get value: {str(e)}"
                self.logger.warning(f"[{self.cam_id}] Failed to get value for {name}: {e}")
                return metadata
            try:
                metadata["is_writeable"] = self.is_writeable(name)
                metadata["access"] = "RW" if metadata["is_writeable"] else "RO"
                self.logger.debug(f"[{self.cam_id}] {name} access={metadata['access']}")
                metadata["type"] = self._guess_type(name)
                self.logger.debug(f"[{self.cam_id}] Guessed type for {name}: {metadata['type']}")
                for key in ["Min", "Max", "Inc"]:
                    try:
                        q_info = query.info(name, key)
                        metadata[key.lower()] = self.params.get(q_info, type(value) if key != "Inc" else float)
                        self.logger.debug(f"[{self.cam_id}] {name} {key}={metadata[key.lower()]}")
                    except Exception as e:
                        metadata[key.lower()] = None
                        self.logger.debug(f"[{self.cam_id}] No {key} for {name}: {e}")
                        if key == "Min" and metadata["type"] in ["Integer", "Float"]:
                            metadata[key.lower()] = 0
                        elif key == "Max" and metadata["type"] == "Integer":
                            metadata[key.lower()] = 2 ** 31 - 1
                        elif key == "Max" and metadata["type"] == "Float":
                            metadata[key.lower()] = 1.0e18
                        elif key == "Inc":
                            metadata[key.lower()] = 1 if metadata["type"] == "Integer" else 0.1
                try:
                    q_unit = query.info(name, "Unit")
                    metadata["unit"] = self.params.get(q_unit, str) if q_unit else ""
                    self.logger.debug(f"[{self.cam_id}] {name} Unit={metadata['unit']}")
                except Exception as e:
                    metadata["unit"] = ""
                    self.logger.debug(f"[{self.cam_id}] No Unit for {name}: {e}")
                    if name.lower() in ["width", "height", "offsetx", "offsety"]:
                        metadata["unit"] = "px"
                    elif "rate" in name.lower() or "frequency" in name.lower():
                        metadata["unit"] = "fps"
                    elif "time" in name.lower():
                        metadata["unit"] = "us"
                try:
                    q_desc = query.info(name, "Description") or query.info(name, "DisplayName") or query.info(name, "ToolTip")
                    metadata["description"] = self.params.get(q_desc, str) if q_desc else name
                    self.logger.debug(f"[{self.cam_id}] {name} Description={metadata['description']}")
                except Exception as e:
                    metadata["description"] = name
                    self.logger.debug(f"[{self.cam_id}] No Description for {name}: {e}")
                if metadata["type"] == "Enumeration":
                    try:
                        enum_entries = self.get_enumeration_entries(name)
                        metadata["enum_entries"] = enum_entries
                        self.logger.debug(f"[{self.cam_id}] {name} Enum Entries={enum_entries}")
                    except Exception as e:
                        metadata["enum_entries"] = []
                        metadata["enum_error"] = f"Failed to get enum entries: {e}"
                        self.logger.warning(f"[{self.cam_id}] Failed to get enum entries for {name}: {e}")
            except Exception as e:
                self.logger.error(f"[{self.cam_id}] Metadata retrieval failed for {name}: {e}", exc_info=True)
            return metadata
    def _guess_type(self, name: str) -> str:
        with self._lock:  # Thread-safe
            try:
                value = self.get_param(name)
                if value is None:
                    return 'String'
                if isinstance(value, bool):
                    return 'Boolean'
                if isinstance(value, int):
                    return 'Integer'
                if isinstance(value, float):
                    return 'Float'
                if isinstance(value, str):
                    try:
                        enum_entries = self.get_enumeration_entries(name)
                        if enum_entries and len(enum_entries) > 0:
                            self.logger.debug(f"[{self.cam_id}] {name} detected as Enumeration with entries: {enum_entries}")
                            return 'Enumeration'
                    except Exception as e:
                        self.logger.debug(f"[{self.cam_id}] No enum entries for {name}: {e}")
                    return 'String'
                return 'String'
            except Exception as e:
                self.logger.warning(f"[{self.cam_id}] Failed to guess type for {name}: {e}")
                return 'String'

    def set_exposure(self, exposure_us: float) -> None:
        with self._lock:
            self.set_param("ExposureTime", float(exposure_us))
            # 수정: f-string에서 'us'를 문자열로 포함
            self.logger.info(f"[{self.cam_id}] Exposure set to {exposure_us} us")

    def get_exposure(self) -> float:
        with self._lock:  # Thread-safe
            return float(self.get_param("ExposureTime"))

    def set_gain(self, gain: float) -> None:
        with self._lock:  # Thread-safe
            self.set_param("Gain", float(gain))
            self.logger.info(f"[{self.cam_id}] Gain set to {gain}")

    def get_gain(self) -> float:
        with self._lock:  # Thread-safe
            return float(self.get_param("Gain"))

    # src/core/camera_controller.py
    # ─────────────────────────────────────────────────────────
    def execute_software_trigger(self, *, cxp_id: int = 0) -> None:
        """
        ① 카메라/Grabber remote 의 *TriggerSoftware* 가 있으면 우선 사용
        ② 없으면 Device-Module 의 **CxpTriggerMessageSend** 로 폴백
           (필요 시 CxpTriggerMessageID 선행 세팅)
        """
        # ── ① 가장 일반적인 TriggerSoftware ──────────────────────
        try:
            self.execute_command("TriggerSoftware", timeout=0.5)
            return
        except CommandExecutionError:
            self.logger.debug("[%s] TriggerSoftware unavailable – trying CXP message",
                              self.cam_id)

        # ── ② CXP Trigger Message (grabber.device) ───────────────
        dev = getattr(self.grabber, "device", None)
        if dev is None:
            raise CommandExecutionError(f"[{self.cam_id}] No DeviceModule for CXP trigger")

        try:
            if "CxpTriggerMessageID" in dev.features():
                dev.set("CxpTriggerMessageID", cxp_id)
            self.execute_command("CxpTriggerMessageSend", timeout=0.5)
            self.logger.info("[%s] CxpTriggerMessageSend(ID=%d) OK", self.cam_id, cxp_id)
        except Exception as exc:
            raise CommandExecutionError(
                f"[{self.cam_id}] Failed to send CXP trigger: {exc}"
            ) from exc

