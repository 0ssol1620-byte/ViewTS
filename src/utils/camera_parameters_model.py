# (새/수정) utils/camera_parameters_model.py
from pathlib import Path
from xml.etree import ElementTree as ET
from typing import Union, Dict, List, Any   # ← Union 추가

class CameraParametersModel:
    _instance: "CameraParametersModel | None" = None

    def __new__(cls, *_, **__):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ─────────────────────────────────────────────────────────
    def __init__(self, xml_dir: Union[str, Path] = "camera_xml"):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.xml_dir = Path(xml_dir)
        self._param_map: Dict[str, List[str]] = {}
        self.reload()                      # XML 우선 로드

    # ─────────────────────────────────────────────────────────
    # 기존: XML 디렉터리만 파싱
    def reload(self) -> None:
        self._param_map.clear()
        if not self.xml_dir.exists():
            return
        for xf in self.xml_dir.glob("*.xml"):
            try:
                root = ET.parse(xf).getroot()
            except Exception:
                continue
            for p in root.iter("Parameter"):
                name = p.attrib.get("Name")
                if not name:
                    continue
                enums = [e.attrib.get("Name") for e in p.iter("EnumEntry")
                         if e.attrib.get("Name")]
                self._param_map.setdefault(name, [])
                self._param_map[name].extend([v for v in enums
                                              if v not in self._param_map[name]])

    # ● 새로 추가: **Euresys Camera API → GenICam Feature** ingest
    def ingest_from_controller(self, controller: Any) -> None:
        """
        active-controller 로부터 Feature 이름·열거형 값을 캐싱한다.
        실패 시 예외를 삼켜서 안전하게 무시.
        """
        if not controller:
            return
        try:
            fnames = controller.list_all_features(only_available=True)
        except Exception:
            return
        for fname in fnames:
            try:
                meta = controller.get_parameter_metadata(fname)
                if meta.get("type") != "Enumeration":
                    # 정수/실수/불리언 파라미터도 이름만이라도 등록
                    self._param_map.setdefault(fname, [])
                    continue
                enums = meta.get("enum_entries", [])
                self._param_map.setdefault(fname, [])
                self._param_map[fname].extend(v for v in enums
                                              if v not in self._param_map[fname])
            except Exception:
                continue    # 개별 feature 오류는 무시

    # ─────────────────────────────────────────────────────────
    # 조회 API
    def list_parameter_names(self) -> List[str]:
        return sorted(self._param_map.keys())

    def list_values_for(self, name: str) -> List[str]:
        return self._param_map.get(name, [])

    def is_enum(self, name: str) -> bool:
        return bool(self._param_map.get(name))
