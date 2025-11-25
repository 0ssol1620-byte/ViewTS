#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/utils/memento_recorder.py

Robust wrapper around Euresys Memento CLI "dump".
- Clean stop with Ctrl+C / SIGINT first
- Salvage dump if the follow-session produced no/empty file
- User-writable output directory fallback
"""

from __future__ import annotations

import atexit
import datetime as _dt
import glob
import logging
import os
import pathlib
import platform
import re
import signal
import subprocess
import sys
import time
from threading import Thread, Event
from typing import Optional, Callable

logging.getLogger(__name__).setLevel(logging.DEBUG)
_IS_WIN = platform.system() == "Windows"
__all__ = ["MementoRecorder", "MementoError"]

class MementoError(RuntimeError):
    pass

# ── Discover CLI ────────────────────────────────────────────────
def _find_exe() -> pathlib.Path:
    exe = "memento.exe" if _IS_WIN else "memento"
    env = os.getenv("EURESYS_MEMENTO_PATH")
    if env:
        p = pathlib.Path(env)
        if p.is_file():
            return p
    for root in os.getenv("PATH", "").split(os.pathsep):
        if not root:
            continue
        p = pathlib.Path(root) / exe
        if p.exists():
            return p
    for folder in [
        r"C:\Program Files\Euresys\Memento\bin\x86_64",
        r"C:\Program Files\Euresys\Memento\bin",
        r"C:\Program Files\Euresys\eGrabber\bin",
        "/usr/local/bin",
        "/usr/bin",
    ]:
        p = pathlib.Path(folder) / exe
        if p.exists():
            return p
    raise MementoError("memento.exe not found")

_VER_RE = re.compile(r"(\d+)\.(\d+)")
def _check_version(path: pathlib.Path) -> str:
    txt = subprocess.check_output([str(path), "--version"], text=True).strip()
    m = _VER_RE.search(txt) or (_VER_RE.search("0.0"))
    ver = m.group(0)
    if int(m.group(1)) < 24:
        raise MementoError(f"Memento {ver} < 24.x not supported")
    logging.debug("[Memento] CLI version %s", ver)
    return ver

def _user_memento_dir() -> pathlib.Path:
    home = pathlib.Path(os.getenv("USERPROFILE", "")) if _IS_WIN else pathlib.Path.home()
    if not str(home):
        home = pathlib.Path.home()
    return (home / "memento").expanduser()

def _ensure_writable_dir(dir_path: pathlib.Path) -> pathlib.Path:
    dir_path = dir_path.expanduser()
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        test = dir_path / ".probe_write"
        with open(test, "w", encoding="utf-8") as f:
            f.write("ok")
        test.unlink(missing_ok=True)
        return dir_path
    except Exception as e:
        logging.warning("[Memento] Output dir not writable (%s) → fallback to %s", e, _user_memento_dir())
        fb = _user_memento_dir()
        fb.mkdir(parents=True, exist_ok=True)
        return fb

# ── Main class ──────────────────────────────────────────────────
class MementoRecorder:
    def __init__(
        self,
        out_dir: str | pathlib.Path,
        *,
        basename: str | None = None,
        follow: bool = True,
        on_event: Optional[Callable[[str], None]] = None,
    ):
        self.exe = _find_exe()
        self.version = _check_version(self.exe)
        self.follow = follow
        self.on_event = on_event or (lambda *_: None)

        safe_dir = _ensure_writable_dir(pathlib.Path(out_dir))
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.basename = basename or f"memento_{ts}"
        self.out_dir = safe_dir
        # 확장자는 자유지만, .memento를 유지 (버전별 기본값은 memento.out일 수 있음)
        self.outfile = self.out_dir / f"{self.basename}.memento"

        self._proc: subprocess.Popen | None = None
        self._stdout_th: Thread | None = None
        self._stop_evt = Event()
        self._stopped_once = False

        self._launch()
        atexit.register(self.stop)

    # ── Public ────────────────────────────────────────────────
    def stop(self, timeout: int = 10):
        """
        Cleanly stop the running "memento dump --follow".
        1) Try Ctrl+C / SIGINT
        2) Wait up to `timeout`
        3) Escalate (TERM → KILL)
        4) If no/empty file → run a short one-shot "salvage dump"
        """
        if self._stopped_once:
            return
        self._stopped_once = True

        if not self._proc:
            try: self.on_event("stopped")
            finally: return

        try:
            if self._proc.poll() is None:
                # 1) Graceful signal
                try:
                    if _IS_WIN:
                        self._try_ctrl_c_windows()
                    else:
                        try:
                            os.killpg(self._proc.pid, signal.SIGINT)
                        except Exception:
                            os.kill(self._proc.pid, signal.SIGINT)
                except Exception as e:
                    logging.debug("[Memento] graceful-signal skipped (%r)", e)

                # 2) Wait
                try:
                    self._proc.wait(timeout)
                except Exception:
                    logging.warning("[Memento] graceful stop timed-out → escalate.")
                    try:
                        if not _IS_WIN:
                            try:
                                os.killpg(self._proc.pid, signal.SIGTERM)
                            except Exception:
                                self._proc.terminate()
                            try:
                                self._proc.wait(timeout=2)
                            except Exception:
                                os.killpg(self._proc.pid, signal.SIGKILL)
                        else:
                            self._proc.terminate()
                            try:
                                self._proc.wait(timeout=2)
                            except Exception:
                                self._proc.kill()
                    except Exception as e:
                        logging.debug("[Memento] escalation failed/ignored (%r)", e)
        finally:
            # Close I/O + join relay
            try:
                if self._proc and self._proc.stdout:
                    try: self._proc.stdout.close()
                    except Exception: pass
                if self._stdout_th and self._stdout_th.is_alive():
                    self._stop_evt.set()
                    self._stdout_th.join(timeout=0.5)
            except Exception:
                pass

            # Short settle time for finalize/rename
            self._post_stop_verify()

            # Salvage dump if empty or missing
            try:
                if (not self.outfile.exists()) or self.outfile.stat().st_size == 0:
                    logging.warning("[Memento] primary file missing/empty → running salvage dump.")
                    self._salvage_dump()
                    # Verify again
                    self._post_stop_verify()
            except Exception as e:
                logging.debug("[Memento] salvage check failed (%r)", e)

            self._stop_evt.set()
            try: self.on_event("stopped")
            except Exception: pass
            self._proc = None

    # ── Windows Ctrl+C delivery ───────────────────────────────
    def _try_ctrl_c_windows(self) -> None:
        if not self._proc:
            return
        # First try the simple route
        try:
            self._proc.send_signal(signal.CTRL_C_EVENT)
            time.sleep(0.2)
            return
        except Exception:
            pass
        # Fallback: Attach to child's console and generate event
        try:
            import ctypes
            from ctypes import wintypes as wt
            k32 = ctypes.windll.kernel32
            AttachConsole = k32.AttachConsole
            FreeConsole = k32.FreeConsole
            SetConsoleCtrlHandler = k32.SetConsoleCtrlHandler
            GenerateConsoleCtrlEvent = k32.GenerateConsoleCtrlEvent
            AttachConsole.argtypes = [wt.DWORD]; AttachConsole.restype = wt.BOOL
            FreeConsole.argtypes = []; FreeConsole.restype = wt.BOOL
            SetConsoleCtrlHandler.argtypes = [ctypes.c_void_p, wt.BOOL]; SetConsoleCtrlHandler.restype = wt.BOOL
            GenerateConsoleCtrlEvent.argtypes = [wt.DWORD, wt.DWORD]; GenerateConsoleCtrlEvent.restype = wt.BOOL

            AttachConsole(self._proc.pid)
            SetConsoleCtrlHandler(None, True)
            try:
                if not GenerateConsoleCtrlEvent(0, self._proc.pid):  # CTRL_C_EVENT=0
                    GenerateConsoleCtrlEvent(1, self._proc.pid)       # CTRL_BREAK_EVENT=1
            finally:
                time.sleep(0.2)
                SetConsoleCtrlHandler(None, False)
                FreeConsole()
        except Exception as e:
            logging.debug("[Memento] Windows Ctrl+C delivery failed (%r)", e)

    # ── Internals ─────────────────────────────────────────────
    def _relay(self, proc: subprocess.Popen):
        try:
            if proc.stdout is None:
                return
            for line in proc.stdout:
                if self._stop_evt.is_set():
                    break
                logging.debug("[Memento/CLI] %s", line.rstrip())
        except Exception as e:
            logging.debug("[Memento] relay terminated (%r)", e)

    def _launch(self) -> None:
        cmd = [str(self.exe), "dump", "--output", str(self.outfile)]
        if self.follow:
            cmd.append("--follow")

        creationflags = 0
        preexec_fn = None
        if _IS_WIN:
            creationflags = 0x00000200  # CREATE_NEW_PROCESS_GROUP
        else:
            preexec_fn = os.setsid

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            creationflags=creationflags,
            preexec_fn=preexec_fn,
        )

        t = Thread(target=self._relay, args=(self._proc,), daemon=True, name="MementoCLI-stdout")
        t.start()
        self._stdout_th = t

        try: self.on_event("started")
        except Exception: pass
        logging.info("[Memento] recording → %s", self.outfile)

    def _post_stop_verify(self):
        """Allow short time for finalize/rename, then log status."""
        try:
            deadline = time.time() + 1.5
            while time.time() < deadline:
                if self.outfile.exists() and self.outfile.stat().st_size > 0:
                    break
                time.sleep(0.1)
            if not self.outfile.exists() or self.outfile.stat().st_size == 0:
                pattern1 = str(self.out_dir / "*.memento")
                pattern2 = str(self.out_dir / "*.out")
                candidates = sorted(
                    glob.glob(pattern1) + glob.glob(pattern2),
                    key=lambda p: os.path.getmtime(p),
                    reverse=True,
                )
                if candidates:
                    logging.warning("[Memento] expected output not ready. Recent files: %s", candidates[:3])
                else:
                    logging.warning("[Memento] no output file created yet: %s", self.outfile)
            else:
                sz = self.outfile.stat().st_size / 1024.0
                logging.info("✅ [Memento] saved → %s (%.1f KB)", self.outfile, sz)
        except Exception as e:
            logging.debug("[Memento] post-stop verification skipped (%r)", e)

    def _salvage_dump(self):
        """Run a short non-follow dump to force materialization of traces."""
        try:
            cmd = [str(self.exe), "dump", "--output", str(self.outfile), "--goback"]
            # pastboot는 선택. 필요하면 주석 제거:
            # cmd.append("--pastboot")
            res = subprocess.run(cmd, text=True, capture_output=True, timeout=15)
            if res.returncode != 0:
                logging.warning("[Memento] salvage dump failed rc=%s, stderr=%s", res.returncode, res.stderr.strip())
            else:
                logging.info("[Memento] salvage dump done. stdout=%s", (res.stdout or "").strip())
        except Exception as e:
            logging.warning("[Memento] salvage dump error (%r)", e)

    # ── Context manager ───────────────────────────────────────
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.stop()
        except Exception as e:
            logging.debug("[Memento] stop() raised in __exit__: %r", e)
        return False

# ── CLI quick test ─────────────────────────────────────────────
def _main():
    import time as _t
    out = _user_memento_dir()
    with MementoRecorder(out_dir=out) as rec:
        logging.info("Recording 3 s …")
        _t.sleep(3)
    try:
        if rec.outfile.exists() and rec.outfile.stat().st_size:
            logging.info("✅ saved → %s (%.1f KB)", rec.outfile, rec.outfile.stat().st_size / 1024)
        else:
            logging.error("❌  Memento file was not created.")
    except Exception:
        logging.error("❌  Failed to validate output file.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname).1s %(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    _main()
