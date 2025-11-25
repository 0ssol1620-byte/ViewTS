"""
ErrorImageManager
=================
• 카메라 Grab 버퍼 또는 NumPy 배열을 TIFF/PNG 파일로 저장
• 보존 주기 / 최대 개수 등의 정책을 담당

Dependencies
------------
* numpy
* opencv-python (cv2)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import cv2  # ★★★ [수정] imageio -> cv2로 변경
from src.core import events

logger = logging.getLogger(__name__)


class ErrorImageManager:
    """
    버퍼 또는 NumPy 배열을 디스크에 저장하고, 보존 정책을 적용한다.
    """
    def __init__(self, *, base_dir: Path | str | None = None, max_keep: int = 500):
        self._root: Path = Path(base_dir or Path(__file__).resolve().parents[2] / "logs" / "error_images")
        self._root.mkdir(parents=True, exist_ok=True)
        self._max_keep = max_keep
        self.save_enabled = True

    def save_error_image_from_numpy(self, frame: np.ndarray, cam_id: str | int, custom_path: Path | str | None = None) -> Optional[Path]:
        """
        NumPy 배열을 직접 받아 이미지 파일로 저장하고 파일 경로를 반환합니다.
        """
        if not self.save_enabled or not isinstance(frame, np.ndarray):
            return None
        try:
            return self._save_and_manage(frame, str(cam_id), custom_path)
        except Exception as e:
            # ★★★ [수정] 예외 발생 시 더 상세한 로그 기록 ★★★
            logger.error(f"Failed to write numpy error image for cam '{cam_id}' using path '{custom_path}'. Reason: {e}", exc_info=True)
            raise e # 예외를 다시 발생시켜 호출한 쪽에서 인지하도록 함

    def _save_and_manage(self, array_to_save: np.ndarray, cam_id: str, custom_path: Path | str | None) -> Path:
        """
        실제 파일 저장 및 보존 정책 적용을 담당하는 내부 메서드.
        """
        cam_dir = self._root / cam_id
        cam_dir.mkdir(parents=True, exist_ok=True)

        if custom_path:
            fpath = Path(custom_path)
            fpath.parent.mkdir(parents=True, exist_ok=True)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            fname = f"ERROR_{cam_id}_{ts}_{uuid.uuid4().hex[:6]}.tiff"
            fpath = cam_dir / fname

        # ★★★ [수정] iio.imwrite -> cv2.imwrite로 변경 ★★★
        # cv2.imwrite는 pathlib.Path 객체를 받을 수 없으므로 str()으로 변환합니다.
        success = cv2.imwrite(str(fpath), array_to_save)
        if not success:
            raise IOError(f"cv2.imwrite failed to save image to {fpath}")

        logger.info("Saved error image: %s  shape=%s", fpath.name, array_to_save.shape)
        self._apply_retention_policy(cam_dir)
        return fpath

    def _apply_retention_policy(self, cam_dir: Path) -> None:
        """`max_keep` 초과 시 가장 오래된 파일부터 삭제."""
        try:
            image_files = sorted(p for p in cam_dir.iterdir() if p.is_file() and p.suffix.lower() in ['.tiff', '.png', '.jpeg', '.jpg', '.bmp'])
            if len(image_files) > self._max_keep:
                files_to_delete = image_files[: len(image_files) - self._max_keep]
                for fp in files_to_delete:
                    fp.unlink(missing_ok=True)
                    logger.info(f"Retention policy: Removed old error image {fp.name}")
        except Exception as exc:
            logger.warning("Retention policy failed in %s: %s", cam_dir, exc)

    def save_and_publish(self, buffer: Any, cam_id: str | int, *, meta: dict[str, Any] | None = None) -> Path:
        """(이 함수는 현재 시퀀스에서 직접 사용되지 않음)"""
        try:
            arr: np.ndarray = buffer.get_numpy_array()
        except Exception as exc:
            raise RuntimeError(f"Cannot convert buffer to ndarray: {exc}") from exc

        path = self.save_error_image_from_numpy(arr, str(cam_id))
        events.publish(events.ErrorImageCaptured(path=path, meta=meta or {}))
        return path