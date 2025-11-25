# --- START OF FILE src/core/actions_base.py ---

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from src.utils.sandbox import safe_eval

# ────────────────────────────────────────────────────────────────────────────
#  Optional dependencies
# ────────────────────────────────────────────────────────────────────────────
try:
    from egrabber import memento_push as _memento_push          # type: ignore
except ImportError:                                             # pragma: no cover
    def _memento_push(msg: str) -> None:
        return                                                  # dummy

NUMPY_AVAILABLE = False
try:
    import numpy as np  # noqa: F401
    NUMPY_AVAILABLE = True
except ImportError:                                            # pragma: no cover
    np = None  # type: ignore

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
#  기본 타입
# ────────────────────────────────────────────────────────────────────────────
ContextKey   = str
ActionResult = Dict[str, Any]


class StepActionResult(dict):
    """
    Dict 서브클래스: 루프/플로우-제어 액션(endloop_control 등)에서
    상태·다음 스텝을 리턴할 때 사용.
    예시::
        return StepActionResult(status="loop_continue", next_step="MyLabel")
    """
    pass


# ────────────────────────────────────────────────────────────────────────────
#  파라미터-타입 식별용 상수
# ────────────────────────────────────────────────────────────────────────────
PARAM_TYPE_INT                = "int"
PARAM_TYPE_FLOAT              = "float"
PARAM_TYPE_STRING             = "string"
PARAM_TYPE_BOOL               = "bool"
PARAM_TYPE_CAMERA_PARAM       = "camera_parameter"
PARAM_TYPE_ENUM               = "enum"
PARAM_TYPE_FILE_SAVE          = "file_save"
PARAM_TYPE_FILE_LOAD          = "file_load"
PARAM_TYPE_CONTEXT_KEY        = "context_key"
PARAM_TYPE_CONTEXT_KEY_OUTPUT = "context_key_output"

# ────────────────────────────────────────────────────────────────────────────
#  카테고리 정규화   ★ NEW ★
# ────────────────────────────────────────────────────────────────────────────
# UI 에서 카테고리를 알파벳/대소문자/공백 오차 없이 묶기 위해 정의
_CATEGORY_ALIASES: Dict[str, str] = {
    "camera control": "Camera Control",
    "camera_control": "Camera Control",
    "grabber":        "Grabber",
    "diagnostic":     "Diagnostics",
    "diagnostics":    "Diagnostics",
    "log":            "Logging",
    "logging":        "Logging",
    "flow":           "Flow",
    "control":        "Control",
    "system":         "System",
    "image":          "Image",
    "context":        "Context",
    "misc":           "Misc",
}

_CATEGORY_PRIORITY: List[str] = [
    # 원하는 표시-순서 (없으면 알파벳순)
    "Flow", "Control", "Camera Control", "Grabber",
    "Image", "Context", "Diagnostics", "Logging",
    "System", "Misc",
]


def _normalize_category(cat: str | None) -> str:
    """대소문자/언더바/공백 무시 정규화."""
    if not cat:
        return "Misc"
    key = cat.strip().lower().replace("_", " ")
    return _CATEGORY_ALIASES.get(key, cat.strip())


# ────────────────────────────────────────────────────────────────────────────
#  액션 인자 & 정의
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class ActionArgument:
    name: str
    display_name: str
    type: str
    description: str
    default_value: Any = None
    options: List[str] = field(default_factory=list)
    required: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None

    def __post_init__(self) -> None:
        # ENUM 타입인데 options 없으면 빈 리스트 보장
        if self.type == PARAM_TYPE_ENUM and self.options is None:
            self.options = []


@dataclass
class ActionDefinition:
    id:           str
    display_name: str
    category:     str
    description:  str
    execute_func: Callable[..., "ActionResult"]
    arguments:    List["ActionArgument"] = field(default_factory=list)
    deprecated:   bool = False
    aliases:      List[str] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────────────
#  전역 레지스트리 & 데코레이터
# ────────────────────────────────────────────────────────────────────────────
ACTION_REGISTRY: Dict[str, ActionDefinition] = {}


def _wrap_with_memento(action_id: str,
                       func: Callable[..., ActionResult]
                       ) -> Callable[..., ActionResult]:
    """액션 실행 전/후 eGrabber memento 태그 삽입."""
    def _wrapped(controller, context, *args, **kwargs):
        tag = f"Action-{action_id}"
        _memento_push(f"{tag}-Start")
        try:
            return func(controller, context, *args, **kwargs)
        finally:
            _memento_push(f"{tag}-End")
    _wrapped.__name__ = func.__name__
    _wrapped.__doc__  = func.__doc__
    return _wrapped


def register_action(
    _func: Optional[Callable[..., ActionResult]] = None,
    *,
    id: Optional[str] = None,
    display_name: Optional[str] = None,
    category: str = "Misc",
    description: Optional[str] = None,
    arguments: Optional[List[ActionArgument]] = None,
    deprecated: bool = False,
    aliases: Optional[List[str]] = None
) -> Callable[..., ActionResult]:
    """데코레이터: 함수를 ACTION_REGISTRY에 등록."""
    aliases = aliases or []

    def decorator(func: Callable[..., ActionResult]) -> Callable[..., ActionResult]:
        action_id = id or func.__name__
        cat_norm  = _normalize_category(category)
        wrapped   = _wrap_with_memento(action_id, func)

        ACTION_REGISTRY[action_id] = ActionDefinition(
            id=action_id,
            display_name=display_name or action_id,
            category=cat_norm,
            description=description or (func.__doc__ or ""),
            execute_func=wrapped,
            arguments=arguments or [],
            deprecated=deprecated,
            aliases=aliases,
        )
        # alias 키도 같은 객체를 가리키게 등록
        for al in aliases:
            ACTION_REGISTRY[al] = ACTION_REGISTRY[action_id]
        return wrapped

    return decorator if _func is None else decorator(_func)


# ────────────────────────────────────────────────────────────────────────────
#  레지스트리 헬퍼
# ────────────────────────────────────────────────────────────────────────────
def get_action_definition(action_id: str) -> Optional[ActionDefinition]:
    return ACTION_REGISTRY.get(action_id)


def list_available_actions(category: Optional[str] = None) -> List[str]:
    """
    category 지정 시 → 해당 카테고리의 ID 목록
    미지정        → 모든 ID 목록(알파벳순)
    """
    if category:
        cat_norm = _normalize_category(category)
        return sorted(
            aid for aid, ad in ACTION_REGISTRY.items()
            if ad.category == cat_norm and aid == ad.id
        )
    return sorted(
        aid for aid, ad in ACTION_REGISTRY.items()
        if aid == ad.id
    )


# ★ NEW – UI 카테고리 목록을 얻는 전용 헬퍼 ★
def list_action_categories() -> List[str]:
    """현재 레지스트리에 존재하는 카테고리명을 정규화해 반환."""
    cats: Set[str] = {
        ad.category for aid, ad in ACTION_REGISTRY.items() if aid == ad.id
    }
    # 우선순위 리스트에 존재하는 카테고리는 해당 순서로, 나머지는 알파벳순
    ordered: List[str] = [c for c in _CATEGORY_PRIORITY if c in cats]
    ordered.extend(sorted(cats - set(ordered)))
    return ordered


# ────────────────────────────────────────────────────────────────────────────
#  컨텍스트 치환 & 형 변환
# ────────────────────────────────────────────────────────────────────────────
CTX_VAR_PATTERN = re.compile(r"\{([^}]+)\}")


def _resolve_context_vars(value: Any, context: Dict[str, Any]) -> Any:
    """문자열 내 {ctx_key} 치환 및 eval: 표현식 평가."""
    if not isinstance(value, str):
        return value
    v = value.strip()

    # ─── eval: ... ───────────────────────────────────────────────
    if v.startswith("eval:"):
        try:
            return safe_eval(v[5:], {k: v for k, v in context.items()
                                     if isinstance(k, str)})
        except Exception as e:                                          # noqa: BLE001
            logger.error("[eval] %s", e, exc_info=True)
            return value

    # ─── {ctx_key} 치환 ──────────────────────────────────────────
    def replace_match(match: re.Match[str]) -> str:
        full_key = match.group(1)

        # 필터 지원 예: {my_list|length}
        if "|" in full_key:
            key, filter_name = full_key.split("|", 1)
            if key in context:
                val = context[key]
                if filter_name == "length":
                    return str(len(val))
                # 확장 필터가 필요하면 여기에 추가
            return match.group(0)

        # {context.foo} → context["foo"]
        if full_key.startswith("context."):
            ck = full_key[8:]
            return str(context.get(ck, match.group(0)))

        return str(context.get(full_key, match.group(0)))

    return CTX_VAR_PATTERN.sub(replace_match, v)


def _convert_value(val: Any) -> Any:
    """문자열 → bool/int/float 자동 캐스팅(실패 시 원본 반환)."""
    if isinstance(val, (bool, int, float)):
        return val
    if not isinstance(val, str):
        return val
    s = val.strip()
    if not s:
        return s
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return val


# ────────────────────────────────────────────────────────────────────────────
#  ActionResult 생성기
# ────────────────────────────────────────────────────────────────────────────
def _ar_success(msg: str,
                details: Optional[Dict[str, Any]] = None,
                **extra) -> ActionResult:
    res: ActionResult = {"status": "success", "message": msg}
    if details:
        res.update(details)
    if extra:
        res.update(extra)
    return res


def _ar_fail(msg: str,
             details: Optional[Dict[str, Any]] = None,
             **extra) -> ActionResult:
    res: ActionResult = {"status": "failed", "message": msg}
    if details:
        res.update(details)
    if extra:
        res.update(extra)
    return res


__all__ = [
    # 타입
    "ContextKey", "ActionResult", "StepActionResult",
    # 파라미터 타입 상수
    "PARAM_TYPE_INT", "PARAM_TYPE_FLOAT", "PARAM_TYPE_STRING", "PARAM_TYPE_BOOL",
    "PARAM_TYPE_CAMERA_PARAM", "PARAM_TYPE_ENUM",
    "PARAM_TYPE_FILE_SAVE", "PARAM_TYPE_FILE_LOAD",
    "PARAM_TYPE_CONTEXT_KEY", "PARAM_TYPE_CONTEXT_KEY_OUTPUT",
    # 레지스트리 관련
    "ActionArgument", "ActionDefinition", "ACTION_REGISTRY",
    "register_action", "get_action_definition", "list_available_actions",
    "list_action_categories",
    # 기타 util
    "_resolve_context_vars", "_convert_value",
    "_ar_success", "_ar_fail",
    # NUMPY flag
    "NUMPY_AVAILABLE",
]

# --- END OF FILE src/core/actions_base.py ---
