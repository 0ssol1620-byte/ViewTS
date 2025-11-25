#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LinkCount Reconnect Test (v1.0)
───────────────────────────────
목표: 아래 시퀀스를 N회 반복 수행해 Link Count 보존 문제를 재현/검증한다.

    1) Link Count = 1 인 상태에서 Link Count = 2 로 변경
    2) Userset 에 해당 모드 변경 정보 저장 (UserSet1)
    3) Device Reset 수행
    4) Device Discovery 후 프로그램에서 카메라와 재접속
    5) 재접속 후 Link Count 값이 2 와 일치하는지 확인
       - 재접속 후 2 이면 OK
       - 재접속 후 1 이면 Error (문제 재현)

사용 예:
  $ python linkcount_reconnect_test_v1_0.py
  → sequences/LinkCount_Reconnect_Test_v1_0_<timestamp>.json 생성

주의:
- Link Count 피처명은 장비/펌웨어마다 다를 수 있다. 우선순위로
  "DeviceLinkCount" → 실패 시 "LinkCount" 를 시도한다.
- Userset 슬롯은 기본 "UserSet1" 을 사용한다.
- 재부팅/재연결 사이에 장치가 올라오는 시간을 고려해 적절한 대기(Wait)를 둔다.
"""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
from typing import List

from src.core.sequence_types import Sequence, SequenceStep

# ─────────────────────────────── Globals ───────────────────────────────
_TS = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
_OUT_DIR = pathlib.Path("sequences")
_OUT_DIR.mkdir(exist_ok=True)

# 반복 횟수 (필요 시 조정)
REPEAT_COUNT = 20
# 재부팅/재연결 사이 안정화 대기(ms)
REBOOT_EXTRA_WAIT_MS = 8000
RECONNECT_WAIT_MS = 1500

# Link Count 피처 후보 (우선순위대로 시도)
LINKCOUNT_FEATURES = [
    "DeviceLinkCount",   # 우선
    "LinkCount",         # 대안
]

# Userset 슬롯
USERSET_ID = "UserSet1"

# ─────────────────────────────── Helpers ───────────────────────────────

def S(lbl: str, act: str, **p) -> SequenceStep:
    """간단 스텝 헬퍼"""
    return SequenceStep(id=lbl, name=lbl, action_id=act, parameters=p or {})


def _set_linkcount_to(value: int) -> List[SequenceStep]:
    steps: List[SequenceStep] = []
    for i, feat in enumerate(LINKCOUNT_FEATURES):
        tag = f"LC_Set_{feat}_{value}"
        # 존재하지 않으면 조용히 다음 후보로 이동
        steps += [S(tag, "set_parameter", parameter_name=feat, value=value, _on_error="continue")]
    return steps


def _assert_linkcount_equals(expected: int) -> List[SequenceStep]:
    steps: List[SequenceStep] = []
    # 후보 각각에 대해 Assert (어느 하나라도 통과하면 시퀀스는 계속)
    for feat in LINKCOUNT_FEATURES:
        steps += [S(f"LC_Assert_{feat}_{expected}", "assert_feature",
                    feature=feat, expected=str(expected), _on_error="continue")]
    # 모든 Assert가 통과하지 못한 경우를 포착하기 위해 다시 읽어서 로그 출력
    # (테스트 결과 가독성 향상)
    for feat in LINKCOUNT_FEATURES:
        steps += [
            S(f"LC_Read_{feat}", "read_parameter", feature=feat, output_context_key=f"cur_{feat}", _on_error="continue"),
        ]
    # 요약 로그
    steps += [
        S("LC_Log_Summary", "log_message",
          level="INFO",
          message=(
              "[LC] After reconnect: "
              + " ".join([f"{f}={{cur_{f}}}" for f in LINKCOUNT_FEATURES])
              + f" | expected={expected}"
          ),
        )
    ]
    return steps


def _repeat_block(block_id: str, steps_body: List[SequenceStep]) -> List[SequenceStep]:
    """count 기반 Repeat 블록 래퍼"""
    return [
        S(f"{block_id}_LoopStart", "repeat_block_start", count=REPEAT_COUNT, block_id=block_id),
        *steps_body,
        S(f"{block_id}_LoopEnd", "repeat_block_end", block_id=block_id),
    ]


# ──────────────────────────── Sequence builder ────────────────────────────

def gen_linkcount_reconnect_test_v1_0() -> Sequence:
    steps: List[SequenceStep] = []

    # 공통 준비
    steps += [
        S("LC_LogStart", "log_message", message="[LC] LinkCount Reconnect Test v1.0"),
        S("LC_Connect", "connect_camera"),
        S("LC_Stop", "execute_command", command_name="AcquisitionStop", _on_error="continue"),
        S("LC_ResetCounters", "reset_all_counters"),
    ]

    # 반복 블록 본문
    body: List[SequenceStep] = []

    # 1) LinkCount = 1 로 강제 (안정적 시작을 위해)
    body += [S("LC_Log_Set1", "log_message", message="[LC] Set LinkCount → 1")]
    body += _set_linkcount_to(1)
    body += [S("LC_Wait_After1", "wait", duration_ms=200)]

    # 1→2) LinkCount = 2 로 변경
    body += [S("LC_Log_Set2", "log_message", message="[LC] Set LinkCount → 2")]
    body += _set_linkcount_to(2)

    # 2) Userset 저장 (UserSet1)
    body += [
        S("LC_UserSet_Select", "set_parameter", parameter_name="UserSetSelector", value=USERSET_ID, _on_error="continue"),
        S("LC_UserSet_Save", "user_set_save", set_id=USERSET_ID),
    ]

    # 3) Device Reset 수행
    body += [
        S("LC_Log_Reboot", "log_message", message="[LC] Reboot device"),
        S("LC_Reboot", "camera_reboot"),
        S("LC_Wait_Reboot", "wait", duration_ms=REBOOT_EXTRA_WAIT_MS),  # 장치 재등장 대기
    ]

    # 4) Device Discovery + 재접속
    body += [
        S("LC_Reconnect", "connect_camera"),
        S("LC_Wait_Reconnect", "wait", duration_ms=RECONNECT_WAIT_MS),
    ]

    # 5) 재접속 후 LinkCount == 2 확인 (미일치 시 실패 → 문제 재현)
    body += _assert_linkcount_equals(2)

    # 사람이 보기 쉬운 판정 로그 (간단 OK 라인)
    body += [
        S("LC_Read_Primary", "read_parameter", feature=LINKCOUNT_FEATURES[0], output_context_key="cur_primary", _on_error="continue"),
        S("LC_OK_If2", "log_message", level="INFO",
          message="[LC] OK if any feature equals 2 → cur_primary={cur_primary}")
    ]

    # 반복 래핑
    steps += _repeat_block("LC_Repeat", body)

    # 종료 정리
    steps += [
        S("LC_FinalLog", "log_message", message="[LC] LinkCount Reconnect Test finished."),
        S("LC_Disconnect", "disconnect_camera"),
    ]

    return Sequence(
        name="LinkCount_Reconnect_Test_v1_0",
        description=(
            "LinkCount 를 1→2 로 변경 후 Userset 저장, 장치 재부팅·재연결을 수행하고 "
            "재접속 시 LinkCount 가 2 로 유지되는지 반복 검증한다."
        ),
        version="1.0",
        steps=steps,
    )


# ─────────────────────────────── Save helper ───────────────────────────────

def _save(seq: Sequence) -> pathlib.Path:
    p = _OUT_DIR / f"{seq.name}_{_TS}.json"
    p.write_text(json.dumps(seq.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ {seq.name} → {p.name}")
    return p


# ─────────────────────────────── Entrypoint ───────────────────────────────
if __name__ == "__main__":
    _save(gen_linkcount_reconnect_test_v1_0())
    print("🚀 LinkCount reconnect test sequence generated.")
