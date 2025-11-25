#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py – 최종 안정화 버전 (아이콘 설정 완벽 반영)

● 실행 순서
  0. (추가) 콘솔 없는 실행 대비: 표준 스트림 확보
  1. Crash 진단 기능 활성화
  2. 환경 변수 설정 (DLL 충돌 방지)
  3. PyQt5 플러그인 경로 설정
  4. 로깅 시스템 초기화
  5. [핵심] 시스템 및 하드웨어 초기화 (카메라 연결)
  6. QApplication 생성 + Windows AppUserModelID + 아이콘 설정
  7. MainWindow 생성 및 UI 상태 동기화 (아이콘 재설정)
  8. 애플리케이션 실행
"""
from __future__ import annotations

# ──────────────────────────────────────────────────────────────
# 0) 콘솔 없는 실행 환경 대비 – 표준 스트림 확보
# ──────────────────────────────────────────────────────────────
import sys, os, pathlib, io

if sys.stderr is None or sys.stdout is None:
    log_dir = pathlib.Path("logs")
    log_dir.mkdir(exist_ok=True)
    fallback = open(log_dir / "silent_output.log", "a", encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = fallback
    if sys.stderr is None:
        sys.stderr = fallback

# ──────────────────────────────────────────────────────────────
# 1) Crash 진단 기능 활성화
# ──────────────────────────────────────────────────────────────
import signal, faulthandler, logging, threading
from pathlib import Path

faulthandler.enable(all_threads=True, file=sys.stderr)
signal.signal(signal.SIGSEGV, lambda *a: faulthandler.dump_traceback(file=sys.stderr))

# ──────────────────────────────────────────────────────────────
# 2) 환경 변수 설정: eGrabber 폴더를 PATH에서 제거
# ──────────────────────────────────────────────────────────────
_EG_KEYWORDS = ("egrabber", "coaxlink", "euresys")
os.environ["PATH"] = os.pathsep.join(
    p for p in os.environ.get("PATH", "").split(os.pathsep)
    if not any(k in p.lower() for k in _EG_KEYWORDS)
)

# ──────────────────────────────────────────────────────────────
# 3) PyQt5 플러그인 경로 설정
# ──────────────────────────────────────────────────────────────
import PyQt5
qt_plugins = Path(PyQt5.__file__).with_name("Qt") / "plugins"
os.environ["QT_PLUGIN_PATH"] = str(qt_plugins)

# ──────────────────────────────────────────────────────────────
# 4) 로깅 초기화
# ──────────────────────────────────────────────────────────────
try:
    from src.utils.logging_setup import init_logging
    Path("logs").mkdir(exist_ok=True)
    init_logging(level=logging.DEBUG,
                 log_file="logs/camera_test_suite.log",
                 file_level=logging.INFO)
except Exception as exc:
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(levelname)s] %(asctime)s - %(message)s")
    logging.getLogger(__name__).warning("basicConfig fallback (%s)", exc)

logger = logging.getLogger(__name__)
logger.info("Launcher started.")

# ──────────────────────────────────────────────────────────────
# 5) 시스템 및 하드웨어 초기화
# ──────────────────────────────────────────────────────────────
try:
    from src.core import initializer
    initializer.initialize_system()
except Exception as exc:
    logger.critical("시스템 초기화 중 치명적인 에러 발생: %s", exc, exc_info=True)
    sys.exit(1)

# ──────────────────────────────────────────────────────────────
# 6) QApplication 생성 + AppUserModelID + 아이콘 설정
# ──────────────────────────────────────────────────────────────
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtGui import QIcon

threading.stack_size(2 * 1024 * 1024)
app = QApplication(sys.argv)

# (Windows) 작업표시줄 그룹/아이콘 결정을 위한 AppUserModelID 설정
if sys.platform.startswith("win"):
    try:
        import ctypes
        # 조직/제품명 규칙으로 고유값 지정 (필요 시 변경)
        app_id = "Vieworks.CameraTestSuite"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        logger.info("AppUserModelID set: %s", app_id)
    except Exception as e:
        logger.warning("Failed to set AppUserModelID: %s", e)

# PyInstaller onefile일 때 내부 임시 폴더(_MEIPASS) 포함 경로에서 아이콘 로드
# --add-data "vieworks.ico;." 로 포함시킨 파일 사용
ico_path = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "vieworks.ico"
icon = QIcon(str(ico_path)) if ico_path.exists() else QIcon()

if not icon.isNull():
    app.setWindowIcon(icon)
    logger.info("App icon applied from: %s", ico_path)
else:
    logger.warning("아이콘 파일을 찾을 수 없거나 로드 실패: %s", ico_path)

try:
    from src.ui.main_window import MainWindow
except Exception as exc:
    logger.critical("MainWindow 임포트 실패: %s", exc, exc_info=True)
    QMessageBox.critical(None, "Startup Error", f"Failed to import UI modules:\n{exc}")
    sys.exit(1)

QApplication.setOrganizationName(MainWindow.ORG_NAME)
QApplication.setApplicationName(MainWindow.APP_NAME)

# ──────────────────────────────────────────────────────────────
# 7) 메인 윈도우 생성, 동기화 및 실행(아이콘 재설정)
# ──────────────────────────────────────────────────────────────
logger.info("메인 윈도우를 생성하고 UI 상태를 동기화합니다.")
win = MainWindow()
win.post_init_setup()

# 일부 Windows/드라이버 환경에서 메인 윈도우에도 아이콘을 직접 지정해야
# 작업표시줄/Alt+Tab 아이콘이 확실히 적용됨
if not icon.isNull():
    try:
        win.setWindowIcon(icon)
    except Exception as e:
        logger.warning("Failed to set window icon on MainWindow: %s", e)

win.show()
logger.info("애플리케이션 실행 시작.")
sys.exit(app.exec_())
