import sys, importlib, runpy
import builtins
import importlib.util as iu

# 모든 eGrabber / CameraController import 을 가짜 모듈로 치환
class _Dummy: pass
sys.modules['egrabber'] = _Dummy()
sys.modules['EGrabber'] = _Dummy()
sys.modules['src.core.camera_controller'] = _Dummy()
sys.modules['src.core.camera_exceptions'] = _Dummy()

print(">>> safe-stub injected, now running main")
runpy.run_module('src.ui.main_window', run_name='__main__')
