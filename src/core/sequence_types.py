#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/core/sequence_types.py

카메라 테스트 시퀀스와 관련된 데이터 구조를 정의합니다.

주요 특징:
- `dataclass`를 사용하여 명확하고 간결한 데이터 구조 정의.
- `SequenceStep`에 `continue_on_fail` 플래그를 포함하여 단계별 실패 정책 제어.
- JSON 직렬화/역직렬화를 위한 클래스 메서드(`to_json`, `from_json`) 제공.
- 타입 힌트를 사용하여 코드 안정성 및 가독성 향상.
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Literal, Tuple, Callable, Union

# 로거 설정
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
#  Helper Functions
# ────────────────────────────────────────────────────────────────────────────

def _dedup(seq: List[Any], sig: Callable[[Any], Tuple[Any, ...]]) -> List[Any]:
    """
    주어진 시그니처 함수를 기반으로 리스트에서 중복을 제거하면서 순서를 보존합니다.
    """
    seen: set[Tuple[Any, ...]] = set()
    out: List[Any] = []
    for obj in seq:
        try:
            key = sig(obj)
            if key in seen:
                continue
            seen.add(key)
            out.append(obj)
        except Exception:
            # 시그니처 생성 실패 시 원본 객체 포함
            out.append(obj)
    return out


# ────────────────────────────────────────────────────────────────────────────
#  Type Definitions for Readability
# ────────────────────────────────────────────────────────────────────────────

ValidationSource = Literal['action_result', 'context']
"""검증 데이터의 출처를 지정합니다 ('액션 결과' 또는 '컨텍스트')."""

ValidationOperator = Literal[
    '==', '!=', '>', '<', '>=', '<=',
    'exists', 'not_exists', 'contains', 'not_contains',
    'is_true', 'is_false', 'is_none', 'is_not_none',
    'matches_regex', 'in_list', 'not_in_list'
]
"""검증에 사용할 비교 연산자입니다."""


# ────────────────────────────────────────────────────────────────────────────
#  Core Data Structures
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationRule:
    """
    SequenceStep의 결과를 검증하기 위한 단일 규칙을 정의합니다.
    """
    name: str = ""
    source: ValidationSource = 'action_result'
    key: str = 'status'
    operator: ValidationOperator = '=='
    target_value: Any = 'success'
    fail_message: Optional[str] = None
    enabled: bool = True


# ─────────────────────────────────────────────────────────────
#  SequenceStep  (loop-control 필드 완전 복원본)
# ─────────────────────────────────────────────────────────────
@dataclass
class SequenceStep:
    # --- identity --------------------------------------------------------
    id: str
    name: str
    action_id: str

    # --- basic execution --------------------------------------------------
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    validations: List[ValidationRule] = field(default_factory=list)
    delay_before_ms: int = 0
    delay_after_ms: int = 0
    continue_on_fail: bool = False
    condition: Optional[str] = None   # “eval:” 표현식 등

    # --- LOOP CONTROL ★ (SequenceRunner 호환용) ---------------------------
    is_loop: bool = False
    loop_start_label: Optional[str] = None   # 사람이 읽는 라벨
    loop_start_index: Optional[int] = None   # 실행 전에 Runner 가 채움
    loop_end_label: Optional[str] = None     # (선택) 가독성용
    loop_condition_key: Optional[str] = None
    loop_operator: Optional[str] = None      # '==', '<', '>=', …
    loop_target_value: Optional[Any] = None
    index_key: Optional[str] = None          # foreach 전용 unique key
    max_loop_count: Optional[int] = None     # 안전장치

    # --- misc -------------------------------------------------------------
    repeat_count: int = 1
    notes: str = ""

    # ↓↓ 레거시 JSON 키를 허용하기 위해 init-time에서 정규화 ↓↓
    def __post_init__(self) -> None:
        # end_loop → loop_end_label 매핑
        if getattr(self, "loop_end_label", None) is None:
            # dataclass 가 대상 필드를 이미 갖고 있으므로 hasattr 체크 불필요
            legacy = getattr(self, "end_loop", None)          # type: ignore
            if legacy:
                self.loop_end_label = legacy

        # loop_label → loop_start_label 매핑
        if getattr(self, "loop_start_label", None) is None:
            legacy = getattr(self, "loop_label", None)        # type: ignore
            if legacy:
                self.loop_start_label = legacy

@dataclass
class Sequence:
    """
    하나의 완전한 테스트 시퀀스를 나타냅니다.
    """
    name: str = "Untitled Sequence"
    description: str = ""
    version: str = "1.0.0"
    steps: List[SequenceStep] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        시퀀스 객체를 직렬화를 위해 딕셔너리로 변환합니다.
        """
        return asdict(self)

    def to_json(self, indent: Optional[int] = 2) -> str:
        """
        시퀀스를 JSON 문자열로 직렬화합니다.
        """
        try:
            return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        except Exception as e:
            logger.error("Sequence to_json serialization error: %s", e, exc_info=True)
            raise TypeError(f"Serialization error for Sequence: {e}") from e


    # --- PATCH: Sequence.from_dict (완전 교체) -------------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sequence":
        """
        Robust deserializer that tolerates legacy-field names (“end_loop” 등) 및
        불필요한 키를 안전하게 무시한다.
        """
        import inspect

        if not isinstance(data, dict):
            raise TypeError("Sequence.from_dict() expects a mapping object")

        name = data.get("name", "Unnamed Sequence")
        description = data.get("description", "")
        version = data.get("version", "")
        steps_data = data.get("steps", [])

        # ── ① SequenceStep 파라미터 목록 확보 ──────────────────────────────
        allowed_keys: set[str] = set(
            inspect.signature(SequenceStep).parameters.keys()
        )

        parsed_steps: list[SequenceStep] = []
        for raw in steps_data:
            if not isinstance(raw, dict):
                raise ValueError("Each step must be a mapping object")

            step_dict = raw.copy()  # ↩︎ 변형 전용 복사본

            # ── ② 레거시 필드 → 신규 필드로 매핑 ───────────────────────────
            if "end_loop" in step_dict and "loop_end_label" not in step_dict:
                step_dict["loop_end_label"] = step_dict.pop("end_loop")

            if "loop_label" in step_dict and "loop_start_label" not in step_dict:
                step_dict["loop_start_label"] = step_dict.pop("loop_label")

            # ── ③ 허용되지 않는 키 제거(추가-메타/주석 등) ─────────────────
            filtered = {k: v for k, v in step_dict.items() if k in allowed_keys}

            try:
                parsed_steps.append(SequenceStep(**filtered))
            except TypeError as exc:
                raise TypeError(
                    f"Failed to deserialize SequenceStep: {exc}. "
                    f"Offending data: {step_dict}"
                ) from exc

        if not parsed_steps:
            raise ValueError("Sequence must contain at least one step")

        return cls(
            name=name,
            description=description,
            version=version,
            steps=parsed_steps,
        )

    # ------------------------------------------------------------------------

    @classmethod
    def from_json(cls, json_str: str) -> 'Sequence':
        """
        JSON 문자열로부터 Sequence 객체를 역직렬화합니다.
        """
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON format for Sequence: %s", e, exc_info=True)
            raise ValueError(f"Invalid JSON format: {e}") from e


# ────────────────────────────────────────────────────────────────────────────
#  Legacy Deserialization Functions (for backward compatibility)
# ────────────────────────────────────────────────────────────────────────────
# 새로운 코드에서는 클래스 메서드 (Sequence.from_dict) 사용을 권장합니다.

def sequence_from_dict(d: Dict[str, Any]) -> Sequence:
    """[Legacy] 딕셔너리로부터 Sequence 객체를 로드합니다."""
    logger.warning("Using deprecated function 'sequence_from_dict'. Please use 'Sequence.from_dict' instead.")
    return Sequence.from_dict(d)