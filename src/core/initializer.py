#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/core/initializer.py

애플리케이션의 하드웨어 및 시스템 초기화를 전담하는 모듈.
이 모듈의 함수는 애플리케이션 시작 시 반드시 호출되어야 합니다.
"""
import logging
from typing import TYPE_CHECKING

# 순환 참조 방지를 위해 타입 힌트만 사용
if TYPE_CHECKING:
    from src.core.camera_controller import CameraController

logger = logging.getLogger(__name__)


def initialize_system():
    """
    애플리케이션의 모든 하드웨어 초기화를 책임지는 유일한 함수.
    프로그램 시작 시 단 한번 호출됩니다.
    이 함수는 controller_pool을 채우는 역할을 합니다.
    """
    # 임포트를 함수 내에서 수행하여 순환 참조 문제를 원천적으로 방지합니다.
    from src.core import controller_pool
    from src.core.camera_controller import CameraController

    logger.info("System Initializer: 모든 카메라 연결 및 등록을 시작합니다...")
    try:
        # 이 함수가 프로그램 전체에서 유일하게 connect_all을 호출하는 지점입니다.
        # 기존 연결이 있다면 모두 정리하고 새로 시작합니다.
        CameraController.connect_all(enable_param_cache=True)

        connected_ids = controller_pool.list_ids()
        if connected_ids:
            logger.info(f"초기화 완료. {len(connected_ids)}개의 카메라가 성공적으로 연결되었습니다.")
        else:
            logger.warning("초기화 완료. 연결된 카메라를 찾지 못했습니다.")

    except Exception as e:
        logger.critical(f"치명적 오류: 시작 시 카메라를 초기화할 수 없습니다. {e}", exc_info=True)
        # 실제 애플리케이션에서는 여기서 사용자에게 오류를 알리고 종료하는 것이 좋습니다.
        # 예를 들어, PyQt 애플리케이션이라면 이 단계 이후에 메시지 박스를 띄울 수 있습니다.
        # import sys
        # from PyQt5.QtWidgets import QMessageBox, QApplication
        # app = QApplication.instance() or QApplication(sys.argv)
        # QMessageBox.critical(None, "Fatal Error", f"Could not initialize cameras: {e}")
        # sys.exit(1)