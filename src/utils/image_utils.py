#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/utils/image_utils.py
───────────────────────────────────────────────────────────────────────────────
이미지 처리 유틸리티

• fast_resize() / make_thumbnail()  ← NEW (GrabWorker 부하 감소용)
• save_frame() / archive_error_image()  ← 확장자 자동(.bmp) + 다단 폴백 + dtype 보정
• compare_frames_advanced()  ← SSIM 채널축 안전
• calculate_stats()
• is_scrambled_image()       ← SSIM 채널축 안전
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Any, Optional, Tuple

# ─────────────────────────── dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:          # pragma: no cover
    np = None  # type: ignore
    NUMPY_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:          # pragma: no cover
    cv2 = None  # type: ignore
    CV2_AVAILABLE = False

try:
    from skimage.metrics import structural_similarity as ssim          # type: ignore
    SKIMAGE_AVAILABLE = True
except ImportError:         # pragma: no cover
    SKIMAGE_AVAILABLE = False

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════
# 공개 기능 플래그 / 상수 (호출부에서 import 하여 사용)
# ═════════════════════════════════════════════════════════════════════
IMAGE_UTILS_AVAILABLE: bool = bool(NUMPY_AVAILABLE)                    # 기본 유틸(리사이즈·통계 등)
IMAGE_SAVE_SUPPORTED: bool = bool(NUMPY_AVAILABLE and CV2_AVAILABLE)   # 파일 저장 가능 여부
IMAGE_COMPARE_SUPPORTED: bool = bool(NUMPY_AVAILABLE)                  # 기본 비교(MSE/AbsDiff/PSNR)
IMAGE_SSIM_SUPPORTED: bool = bool(NUMPY_AVAILABLE and SKIMAGE_AVAILABLE)

# 저장 가능 확장자 / 기본 확장자 / 파일 대화상자 필터
SAVE_SUPPORTED_EXTS: Tuple[str, ...] = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")
DEFAULT_IMAGE_EXT: str = ".bmp"
FILE_DIALOG_FILTERS = [
    ("Bitmap (*.bmp)", "*.bmp"),
    ("PNG (*.png)", "*.png"),
    ("JPEG (*.jpg;*.jpeg)", "*.jpg;*.jpeg"),
    ("TIFF (*.tif;*.tiff)", "*.tif;*.tiff"),
    ("WebP (*.webp)", "*.webp"),
    ("All Files (*.*)", "*.*"),
]

__all__ = [
    # functions
    "fast_resize", "make_thumbnail", "save_frame", "archive_error_image",
    "compare_frames_advanced", "calculate_stats", "is_scrambled_image",
    # flags / constants
    "IMAGE_UTILS_AVAILABLE", "IMAGE_SAVE_SUPPORTED",
    "IMAGE_COMPARE_SUPPORTED", "IMAGE_SSIM_SUPPORTED",
    "SAVE_SUPPORTED_EXTS", "DEFAULT_IMAGE_EXT", "FILE_DIALOG_FILTERS",
]

# ═════════════════════════════════════════════════════════════════════
# 내부 헬퍼 (모듈 외부로 노출 X)
# ═════════════════════════════════════════════════════════════════════
def _normalize_filepath(filepath: str) -> Tuple[str, str]:
    """확장자 누락/이상치 처리해 저장 가능한 경로로 정규화한다."""
    root, ext = os.path.splitext(filepath)
    ext_l = ext.lower()
    if not ext_l:
        # 확장자 없으면 .bmp 부여
        fixed = root + DEFAULT_IMAGE_EXT
        return fixed, DEFAULT_IMAGE_EXT
    if ext_l not in SAVE_SUPPORTED_EXTS:
        logger.warning("Unsupported image extension '%s'. Using default '%s'.", ext_l, DEFAULT_IMAGE_EXT)
        fixed = root + DEFAULT_IMAGE_EXT
        return fixed, DEFAULT_IMAGE_EXT
    return filepath, ext_l


def _strip_alpha_if_needed(img: "np.ndarray", target_ext: str) -> "np.ndarray":
    """BMP/JPEG 저장 시 4채널이면 알파 제거. PNG/TIFF/WebP는 4채널 지원."""
    if img.ndim == 3 and img.shape[2] == 4 and target_ext in (".bmp", ".jpg", ".jpeg"):
        # BGRA → BGR
        return img[:, :, :3]
    return img


def _to_8bit(img: "np.ndarray") -> "np.ndarray":
    """이미지를 0..255 uint8로 변환."""
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.bool_:
        return (img.astype(np.uint8) * 255)
    if np.issubdtype(img.dtype, np.floating):
        # 값 범위 추정: [0,1] 또는 [0,255] 케이스 커버
        maxv = float(np.nanmax(img))
        minv = float(np.nanmin(img))
        if np.isfinite(maxv) and maxv <= 1.0 and minv >= 0.0:
            scaled = img * 255.0
        else:
            # 일반적 0..255 범위를 벗어나면 클리핑
            scaled = np.clip(img, 0.0, 255.0)
        return scaled.astype(np.uint8)
    if np.issubdtype(img.dtype, np.integer):
        # 16비트 등 → 8비트 downscale (상위 비트 버림)
        info = np.iinfo(img.dtype)
        if info.max > 255:
            scaled = (img.astype(np.float32) * (255.0 / float(info.max)))
            return np.clip(scaled, 0, 255).astype(np.uint8)
        return img.astype(np.uint8)
    # 그 외 dtype은 안전하게 8비트로
    return np.clip(img.astype(np.float32), 0, 255).astype(np.uint8)


def _coerce_for_ext(img: "np.ndarray", ext: str) -> "np.ndarray":
    """
    확장자별 허용 포맷으로 dtype/채널 보정.
    - PNG/TIFF/WebP: uint8/uint16/float 지원(실제 저장은 OpenCV 구현에 따름). 여기선 16비트 유지 시 PNG/TIFF 우선.
    - BMP/JPEG: 8비트만 안정적 → 8비트 강제.
    - 1채널(HxWx1) → HxW로 압축.
    """
    if img is None:
        return img
    out = img
    # HxWx1 → HxW
    if out.ndim == 3 and out.shape[2] == 1:
        out = out[:, :, 0]
    # 채널/알파 정리
    out = _strip_alpha_if_needed(out, ext)

    # dtype 보정
    if ext in (".png", ".tif", ".tiff"):
        # PNG/TIFF는 uint16도 잘 저장됨. float은 OpenCV가 8비트로 양자화할 수 있으므로 명시 변환 권장 X.
        # 다만 float NaN/Inf 방지는 필요.
        if np.issubdtype(out.dtype, np.floating):
            # NaN/Inf → 0 처리 후 8비트로
            out = np.nan_to_num(out, nan=0.0, posinf=255.0, neginf=0.0)
            out = _to_8bit(out)
        elif out.dtype == np.bool_:
            out = _to_8bit(out)
        # uint16은 그대로 두되, 값 범위 체크
        if out.dtype == np.uint16:
            return out
        # 그 외 정수는 그대로 또는 8비트로
        if out.dtype != np.uint8:
            out = _to_8bit(out)
        return out
    else:
        # BMP/JPEG/WebP는 8비트가 가장 안전
        return _to_8bit(out)


def _ensure_dir(path: str) -> None:
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def _try_imwrite(path: str, img: "np.ndarray") -> bool:
    try:
        ok = cv2.imwrite(path, img)
        return bool(ok)
    except Exception as e:  # pragma: no cover
        logger.error("cv2.imwrite raised: %s", e, exc_info=True)
        return False


def _save_frame_with_fallback(frame: "np.ndarray", filepath: str) -> Tuple[bool, Optional[str]]:
    """
    실제 저장을 수행하고, 필요 시 확장자 폴백을 수행한다.
    반환: (성공여부, 실제 저장경로[성공 시])
    """
    if not IMAGE_SAVE_SUPPORTED:
        raise ValueError("OpenCV (cv2) + NumPy required for saving frames.")
    if not isinstance(frame, np.ndarray):
        raise ValueError("NumPy array required for frame.")
    if not filepath:
        raise ValueError("Filepath must be provided.")

    # 1) 경로 정규화 및 1차 시도
    path, ext = _normalize_filepath(filepath)
    _ensure_dir(path)
    img = _coerce_for_ext(frame, ext)
    if _try_imwrite(path, img):
        return True, path

    logger.warning("imwrite failed for '%s'. Trying fallbacks…", path)

    # 2) 확장자 폴백 순서
    fallbacks = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
    for fb_ext in fallbacks:
        if fb_ext == ext:
            continue
        alt_path = os.path.splitext(path)[0] + fb_ext
        alt_img = _coerce_for_ext(frame, fb_ext)
        if _try_imwrite(alt_path, alt_img):
            logger.info("Saved with fallback extension → %s", alt_path)
            return True, alt_path

    logger.error("All imwrite fallbacks failed for base '%s'.", os.path.splitext(path)[0])
    return False, None


# ═════════════════════════════════════════════════════════════════════
# Fast thumbnail / resize helpers
# ═════════════════════════════════════════════════════════════════════
def fast_resize(
    frame: "np.ndarray",
    target_w: int,
    *,
    interp: str = "linear",
) -> "np.ndarray":
    """
    초고속 리사이즈 유틸.

    OpenCV가 있으면 cv2.resize() → INTER_LINEAR / AREA / NEAREST 등
    없으면 NumPy 슬라이싱 기반 2^n 다운샘플링.

    Args:
        frame (np.ndarray): 입력 (RGB/BGR/Gray).
        target_w (int): 목표 가로 픽셀.
        interp (str): 'nearest' | 'linear' | 'area'.

    Returns:
        np.ndarray: 리사이즈 결과 (copy).
    """
    if not NUMPY_AVAILABLE or not isinstance(frame, np.ndarray):
        raise ValueError("NumPy ndarray required.")

    h, w = frame.shape[:2]
    if target_w <= 0 or target_w >= w:
        return frame.copy()

    scale = target_w / w
    target_h = int(round(h * scale))

    if CV2_AVAILABLE:
        it_map = {
            "nearest": cv2.INTER_NEAREST,
            "area": cv2.INTER_AREA,
            "linear": cv2.INTER_LINEAR,
        }
        it = it_map.get(interp.lower(), cv2.INTER_LINEAR)
        return cv2.resize(frame, (target_w, target_h), interpolation=it)

    # ── Fallback: nearest-neighbor downsampling by slice
    step = max(int(round(1 / scale)), 1)
    return frame[::step, ::step].copy()


def make_thumbnail(frame: "np.ndarray", max_px: int = 320) -> "np.ndarray":
    """
    긴 변이 max_px 이하가 되도록 축소한 썸네일 반환.

    Args:
        frame (np.ndarray): 입력 프레임.
        max_px (int): 가로나 세로 중 큰 쪽 최대 픽셀.

    Returns:
        np.ndarray: 썸네일 (copy).
    """
    if not NUMPY_AVAILABLE or not isinstance(frame, np.ndarray):
        raise ValueError("NumPy ndarray required.")

    h, w = frame.shape[:2]
    if max(h, w) <= max_px:
        return frame.copy()

    if h >= w:
        return fast_resize(frame, int(w * max_px / h), interp="area")
    return fast_resize(frame, max_px, interp="area")


# ═════════════════════════════════════════════════════════════════════
# Frame save / archive helpers
# ═════════════════════════════════════════════════════════════════════
def save_frame(frame: "np.ndarray", filepath: str) -> bool:
    """
    프레임을 파일로 저장한다.
    - 확장자 누락/미지원 → '.bmp' 기본 적용
    - 실패 시 PNG → JPEG → TIFF → BMP → WebP 순으로 폴백

    Returns:
        bool: 저장 성공 여부
    """
    ok, _ = _save_frame_with_fallback(frame, filepath)
    if ok:
        logger.info("Frame saved (requested path: %s)", filepath)
    else:
        logger.error("cv2.imwrite() failed for %s", filepath)
    return ok


def archive_error_image(
    frame: "np.ndarray",
    test_case_id: Optional[str] = None,
    step_name: Optional[str] = None,
    base_dir: str = "error_images",
) -> Optional[str]:
    """문제 이미지 보존용 아카이브. 실제 저장된 경로(폴백 반영)를 반환한다."""
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_part = f"{test_case_id}_" if test_case_id else ""
    step_part = f"{step_name}_" if step_name else ""
    filename = f"{test_part}{step_part}{ts}.png"  # 기본 제안은 PNG로, 이후 내부에서 폴백 가능
    path = os.path.join(base_dir, filename)

    ok, saved_path = _save_frame_with_fallback(frame, path)
    return saved_path if ok else None


# ═════════════════════════════════════════════════════════════════════
# Comparison / stats / scrambled check
# ═════════════════════════════════════════════════════════════════════
def compare_frames_advanced(
    frame_a: "np.ndarray",
    frame_b: "np.ndarray",
    metric: str = "AbsDiffMean",
) -> Dict[str, Any]:
    """다양한 메트릭으로 프레임 비교."""
    if not NUMPY_AVAILABLE or not isinstance(frame_a, np.ndarray) or not isinstance(frame_b, np.ndarray):
        return {"status": "error", "message": "NumPy ndarray required.", "value": None}
    if frame_a.shape != frame_b.shape:
        return {"status": "error", "message": "Shape mismatch.", "value": None}

    m = metric.lower()
    try:
        a = frame_a.astype(np.float32, copy=False)
        b = frame_b.astype(np.float32, copy=False)

        if m in ("absdiffmean", "absdiff", "l1"):
            val = float(np.mean(np.abs(a - b)))
            return {"status": "success", "message": f"AbsDiffMean={val:.4f}", "value": val}

        if m in ("mse", "l2"):
            val = float(np.mean((a - b) ** 2))
            return {"status": "success", "message": f"MSE={val:.4f}", "value": val}

        if m == "ssim":
            if not SKIMAGE_AVAILABLE:
                raise ImportError("scikit-image required for SSIM.")
            dr = float(frame_a.max() - frame_a.min()) if frame_a.size else 0.0
            if frame_a.ndim == 3 and frame_a.shape[2] in (3, 4):
                # 4채널이면 알파 제거 후 채널축 명시
                fa = frame_a[:, :, :3] if frame_a.shape[2] == 4 else frame_a
                fb = frame_b[:, :, :3] if frame_b.shape[2] == 4 else frame_b
                val = float(ssim(fa, fb, channel_axis=2, data_range=dr if dr > 0 else 255))
            else:
                val = float(ssim(frame_a, frame_b, data_range=dr if dr > 0 else 255))
            return {"status": "success", "message": f"SSIM={val:.4f}", "value": val}

        if m == "psnr":
            mse_val = float(np.mean((a - b) ** 2))
            if mse_val == 0:
                return {"status": "success", "message": "PSNR=inf", "value": float("inf")}
            max_pix = float(np.max(frame_a))
            # max_pix가 0이면 PSNR 정의상 -inf가 되므로 방어적 처리
            max_pix = max(max_pix, 1.0)
            val = float(20 * np.log10(max_pix / np.sqrt(mse_val)))
            return {"status": "success", "message": f"PSNR={val:.4f}", "value": val}

        raise ValueError(f"Unsupported metric: {metric}")
    except Exception as e:  # pragma: no cover
        logger.error("compare_frames_advanced error: %s", e, exc_info=True)
        return {"status": "error", "message": str(e), "value": None}


def calculate_stats(
    frame: "np.ndarray",
    stats_to_calculate: Dict[str, bool] | None = None,
) -> Dict[str, Any]:
    """프레임 통계."""
    if not NUMPY_AVAILABLE or not isinstance(frame, np.ndarray):
        return {"status": "error", "message": "NumPy ndarray required.", "values": {}}
    if frame.size == 0:
        return {"status": "error", "message": "Empty frame.", "values": {}}

    stats_to_calculate = stats_to_calculate or {"mean": True, "stddev": True, "minmax": True}
    values: Dict[str, float] = {}
    try:
        if stats_to_calculate.get("mean"):
            values["mean"] = float(np.mean(frame))
        if stats_to_calculate.get("stddev"):
            values["stddev"] = float(np.std(frame))
        if stats_to_calculate.get("minmax"):
            values["min"] = float(np.min(frame))
            values["max"] = float(np.max(frame))
        return {"status": "success", "message": "ok", "values": values}
    except Exception as e:  # pragma: no cover
        logger.error("calculate_stats error: %s", e, exc_info=True)
        return {"status": "error", "message": str(e), "values": {}}


def is_scrambled_image(
    current: "np.ndarray",
    reference: "np.ndarray",
    ssim_threshold: float = 0.85,
    mean_diff_threshold: float = 15.0,
    archive_on_error: bool = False,
    test_case_id: Optional[str] = None,
    step_name: Optional[str] = None,
) -> bool:
    """평균 차이+SSIM 기반 스크램블 이미지 판정."""
    if current is None or reference is None:
        return True
    if not NUMPY_AVAILABLE or not isinstance(current, np.ndarray) or not isinstance(reference, np.ndarray):
        return True
    if current.shape != reference.shape:
        return True

    # 평균 절대차
    mean_diff = float(np.mean(np.abs(current.astype(np.float32) - reference.astype(np.float32))))

    # SSIM (가능하면 채널축 명시)
    ssim_score = 1.0
    if SKIMAGE_AVAILABLE:
        try:
            ref = reference
            cur = current
            # 알파 제거
            if ref.ndim == 3 and ref.shape[2] == 4 and cur.ndim == 3 and cur.shape[2] == 4:
                ref = ref[:, :, :3]
                cur = cur[:, :, :3]
            rng = float(ref.max() - ref.min()) if ref.size else 0.0
            data_range = rng if rng > 0 else 255.0
            if ref.ndim == 3 and ref.shape[2] in (3, 4):
                ssim_score = float(ssim(cur, ref, channel_axis=2, data_range=data_range))
            else:
                ssim_score = float(ssim(cur, ref, data_range=data_range))
        except Exception as e:
            logger.error("SSIM error: %s", e, exc_info=True)
            ssim_score = 0.0
    else:
        logger.debug("skimage not available; SSIM skipped.")

    scrambled = (mean_diff > mean_diff_threshold) or (ssim_score < ssim_threshold)
    if scrambled and archive_on_error:
        saved = archive_error_image(current, test_case_id, step_name)
        if saved:
            logger.info("Scrambled image archived at %s", saved)
    return scrambled
