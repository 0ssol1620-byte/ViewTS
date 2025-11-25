"""
경량 이벤트 버스 + 데이터 클래스 정의
------------------------------------

GUI(Qt) 신호와 별개로, 순수 Python 층에서 publish/subscribe 패턴을 씁니다.
복잡한 외부 라이브러리(FastAPI/Eventlet 등) 없이 최소 기능만 제공합니다.
"""

from __future__ import annotations

import logging
import sys
from collections import defaultdict
from dataclasses import dataclass as _dc
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, TypeVar

logger = logging.getLogger(__name__)
_T = TypeVar("_T")

# ──────────────────────────────────────────────
#  dataclass – 슬롯 옵션이 지원되는 버전만 활성화
# ──────────────────────────────────────────────
def _dataclass(*d_args, **d_kwargs):          # noqa: D401
    """
    dataclass decorator with optional `slots=True` support.

    • Py 3.10+  →  그대로 사용 (slots 유지)
    • 그 이하   →  slots 파라미터를 제거하고 적용
    """
    if sys.version_info < (3, 10):
        d_kwargs.pop("slots", None)
    return _dc(*d_args, **d_kwargs)
# ──────────────────────────────────────────────
#  Event base & concrete event dataclasses
# ──────────────────────────────────────────────
class Event:
    """모든 이벤트의 기반형 (마커 클래스)."""


@_dataclass(slots=True)
class ErrorImageCaptured(Event):
    """에러 이미지 저장 완료 → UI / 기록 시스템 알림용."""
    path: Optional[Path]
    meta: Dict[str, Any]


# ──────────────────────────────────────────────
#  초경량 publish / subscribe 구현
# ──────────────────────────────────────────────
_SubCallback = Callable[[Event], None]
_subscribers: dict[Type[Event], List[_SubCallback]] = defaultdict(list)

def subscribe(ev_type: Type[_T], cb: Callable[[_T], None]) -> None:
    if cb not in _subscribers[ev_type]:
        _subscribers[ev_type].append(cb)
        logger.debug("Subscribed %s to %s", cb, ev_type.__name__)

def unsubscribe(ev_type: Type[_T], cb: Callable[[_T], None]) -> None:
    _subscribers.get(ev_type, []).remove(cb)

def publish(event: Event) -> None:
    ev_type = type(event)
    for cb in list(_subscribers.get(ev_type, [])):
        try:
            cb(event)          # type: ignore[arg-type]
        except Exception as exc:
            logger.warning("Subscriber %s raised: %s", cb, exc, exc_info=exc)

def get_subscribers(ev_type: Type[Event]) -> Iterable[_SubCallback]:
    return tuple(_subscribers.get(ev_type, []))

__all__ = [
    "Event",
    "ErrorImageCaptured",
    "publish",
    "subscribe",
    "unsubscribe",
    "get_subscribers",
]