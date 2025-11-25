#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/core/action_registry.py

Central registry that maps every action-ID to its ActionDefinition.

▶  HOW TO USE
    from src.core.action_registry import (
        get_action_definition,
        list_available_actions
    )

    # fetch meta
    ad = get_action_definition("connect_camera")
    print(ad.display_name)

    # enumerate all actions (id, display_name, category)
    for aid, name, cat in list_available_actions():
        print(aid, name, cat)
"""

from __future__ import annotations
import logging
from typing import Dict, Callable, List, Tuple, Optional, cast

from src.core.actions_base import (
    ActionDefinition,
    ActionArgument,
    ActionResult,
    ContextKey,
    PARAM_TYPE_INT,
    PARAM_TYPE_FLOAT,
    PARAM_TYPE_STRING,
    PARAM_TYPE_BOOL,
    PARAM_TYPE_CAMERA_PARAM,
    PARAM_TYPE_ENUM,
    PARAM_TYPE_FILE_SAVE,
    PARAM_TYPE_FILE_LOAD,
    PARAM_TYPE_CONTEXT_KEY,
    PARAM_TYPE_CONTEXT_KEY_OUTPUT,

)

# concrete implementations
from src.core.actions_impl import (
    execute_nop,
    execute_wait,
    execute_log_message,
    execute_connect_camera,
    execute_disconnect_camera,
    execute_set_parameter,
    execute_set_trigger_mode,
    execute_camera_reboot,
    execute_get_parameter_metadata,
    execute_set_exposure_auto,
    execute_set_width_with_increment,
    execute_set_multi_roi_region,
    execute_user_set_save,  # ★ NEW
    execute_user_set_load,  # ★ NEW
    execute_set_binning,  # ★ NEW
    execute_start_grab,
    execute_stop_grab,
    execute_grab_frames,
    execute_execute_command,
    execute_save_image,
    execute_compare_brightness,
    execute_compare_images_advanced,
    execute_calculate_image_stats,
    execute_wait_until,
    execute_set_context_variable,
    execute_loop_control,
    execute_detect_scrambled_image,
    execute_copy_context_value,
    reset_frame_loss_monitor,
    execute_check_frame_loss,
    execute_detect_random_defect_pixel,
    execute_read_parameter,
    execute_assert_feature,
    execute_flush_grabber,
    execute_connect_all_cameras,
    execute_get_controller_ids,
    execute_foreach_controller,
    execute_endforeach_controller,
    execute_broadcast_software_trigger,
    execute_broadcast_camera_command,
    execute_device_send_linktrigger0,
    execute_configure_internal_trigger,
    execute_set_trigger_source_auto,
    execute_setup_camera_for_hw_trigger,
    execute_setup_grabber_for_hw_trigger,
    execute_grabber_hw_trigger,
    execute_setup_grabber_cic_for_host_trigger,
    execute_execute_grabber_cycle_start,
    execute_endloop_control,
    execute_get_next_frame,
    execute_feature_command_flexible,
    execute_get_last_cached_frame,
    execute_wait_for_stable_frame,
    execute_repeat_block_start,
    execute_repeat_block_end,
    execute_configure_trigger,
    execute_reset_hardware_counters,
    execute_reset_all_counters,
    execute_configure_linescan_dimensions
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
#  REGISTRY – every new ActionDefinition goes here
# ═══════════════════════════════════════════════════════════════════════
ACTION_REGISTRY: Dict[str, ActionDefinition] = {
    "user_set_save": ActionDefinition(
        id="user_set_save",
        display_name="Save User Set",
        category="Camera",
        description="Saves the camera's current settings to a non-volatile memory slot (User Set).",
        execute_func=cast(Callable[..., ActionResult], execute_user_set_save),
        arguments=[
            ActionArgument("set_id", "User Set ID", PARAM_TYPE_ENUM,
                           "The memory slot to save the settings to.",
                           default_value="UserSet1",
                           options=["Default", "UserSet1", "UserSet2", "UserSet3"])
        ]
    ),
    "configure_linescan_dimensions": ActionDefinition(
        id="configure_linescan_dimensions",
        display_name="Configure Linescan Dimensions",
        category="Grabber",
        description="Sets the grabber's ScanLength and BufferHeight. The change takes effect on the next 'Start Grab'.",
        execute_func=cast(Callable[..., ActionResult], execute_configure_linescan_dimensions),
        arguments=[
            ActionArgument("scan_length", "Scan Length", PARAM_TYPE_INT,
                           "The desired logical height of the image (in lines).",
                           required=True, default_value=1024, min_value=1, max_value=65536),
            ActionArgument("buffer_height", "Buffer Height", PARAM_TYPE_INT,
                           "Optional: Physical buffer height. If empty, it will be set equal to Scan Length.",
                           required=False, default_value=1024, min_value=1, max_value=65536)
        ]
    ),
    "user_set_load": ActionDefinition(
        id="user_set_load",
        display_name="Load User Set",
        category="Camera",
        description="Loads settings from a non-volatile memory slot (User Set) into the camera.",
        execute_func=cast(Callable[..., ActionResult], execute_user_set_load),
        arguments=[
            ActionArgument("set_id", "User Set ID", PARAM_TYPE_ENUM,
                           "The memory slot to load settings from.",
                           default_value="UserSet1",
                           options=["Default", "UserSet1", "UserSet2", "UserSet3"])
        ]
    ),

    "set_binning": ActionDefinition(
        id="set_binning",
        display_name="Set Binning",
        category="Camera",
        description="Sets binning with independent control over horizontal/vertical modes (Sum/Average) and factors.",
        execute_func=cast(Callable[..., ActionResult], execute_set_binning),
        arguments=[
            ActionArgument("selector", "Selector", PARAM_TYPE_ENUM,
                           "Top-level selector (e.g., Sensor, Logic). Leave empty to ignore.",
                           default_value="", required=False,
                           options=["", "Sensor", "Logic"]),
            ActionArgument("horizontal_mode", "Horizontal Mode", PARAM_TYPE_ENUM,
                           "Algorithm for horizontal binning. Leave empty to ignore.",
                           default_value="", required=False,
                           options=["", "Sum", "Average"]),
            ActionArgument("vertical_mode", "Vertical Mode", PARAM_TYPE_ENUM,
                           "Algorithm for vertical binning. Leave empty to ignore.",
                           default_value="", required=False,
                           options=["", "Sum", "Average"]),
            ActionArgument("horizontal_factor", "Horizontal Factor", PARAM_TYPE_INT,
                           "Desired horizontal binning factor (e.g., 1, 2, 4).", default_value=1, min_value=1),
            ActionArgument("vertical_factor", "Vertical Factor", PARAM_TYPE_INT,
                           "Desired vertical binning factor (e.g., 1, 2, 4).", default_value=1, min_value=1),
        ]
    ),
    "reset_all_counters": ActionDefinition(
        id="reset_all_counters",
        display_name="Reset All Counters",
        category="Diagnostics",
        description="Resets all software (FrameLossMonitor) and hardware (Trigger/Stream loss) counters to prepare for a new test.",
        execute_func=cast(Callable[..., ActionResult], execute_reset_all_counters),
        arguments=[],
    ),
    "check_frame_loss": ActionDefinition(
        id="check_frame_loss",
        display_name="Verify All Losses",
        category="Diagnostics",
        description="The ultimate validation tool for all software and hardware losses, including internal pipeline consistency checks.",
        execute_func=cast(Callable[..., ActionResult], execute_check_frame_loss),
        arguments=[
            ActionArgument("max_lost_frames", "Max Lost Frames (App)", PARAM_TYPE_INT,
                           "Allowed software-level frame loss (from FrameLossMonitor).", default_value=0),
            ActionArgument("max_lost_triggers", "Max Lost Triggers (HW)", PARAM_TYPE_INT,
                           "Allowed hardware-level trigger loss (CycleLostTriggerCount).", default_value=0),
            ActionArgument("max_stream_errors", "Max Stream Errors (HW)", PARAM_TYPE_INT,
                           "Allowed total hardware-level stream data error count.", default_value=0),
        ]
    ),

    "wait": ActionDefinition(
        id="wait", display_name="Wait (ms)", category="Control",
        description="지정된 시간(밀리초)만큼 실행을 일시 중지합니다.",
        execute_func=cast(Callable[..., ActionResult], execute_wait),
        arguments=[
            ActionArgument("duration_ms", "Duration (ms)", PARAM_TYPE_INT,
                         description="일시 중지할 시간(ms)입니다.", default_value=1000)
        ]
    ),

    "log_message": ActionDefinition(
        id="log_message", display_name="Log Message", category="Control",
        description="사용자 정의 메시지를 로그에 기록합니다.",
        execute_func=cast(Callable[..., ActionResult], execute_log_message),
        arguments=[
            ActionArgument("message", "Message", PARAM_TYPE_STRING, description="기록할 메시지 내용입니다."),
            ActionArgument("level", "Level", PARAM_TYPE_ENUM, description="로그의 심각도 수준입니다.", options=["DEBUG", "INFO", "WARNING", "ERROR"], default_value="INFO"),
        ]
    ),
    "wait_until": ActionDefinition(
        id="wait_until",
        display_name="Wait Until Cond.",
        category="Control",
        description="Waits until a context variable meets a condition.",
        execute_func=cast(Callable[..., ActionResult], execute_wait_until),
        arguments=[
            ActionArgument("condition_context_key", "Context key", PARAM_TYPE_CONTEXT_KEY, "Key to monitor."),
            ActionArgument(
                "operator",
                "Operator",
                PARAM_TYPE_ENUM,
                "Comparison operator.",
                default_value="==",
                options=['==', '!=', '>', '<', '>=', '<=', 'exists', 'not_exists',
                         'contains', 'not_contains'],
            ),
            ActionArgument("target_value", "Target", PARAM_TYPE_STRING, "Target value (string)."),
            ActionArgument(
                "poll_interval_ms",
                "Poll interval (ms)",
                PARAM_TYPE_INT,
                "Check interval.",
                default_value=100,
                required=False,
                min_value=10,
                max_value=5000,
                step=10,
            ),
            ActionArgument(
                "timeout_ms",
                "Timeout (ms)",
                PARAM_TYPE_INT,
                "Maximum wait time.",
                default_value=5000,
                min_value=100,
                max_value=600000,
                step=100,
            ),
        ],
    ),
    "get_next_frame": ActionDefinition(
        id="get_next_frame",
        display_name="Get Next Frame",
        category="Image Acquisition",
        description="Dequeues a single frame without stopping acquisition.",
        execute_func=cast(Callable[..., ActionResult], execute_get_next_frame),
        arguments=[
            ActionArgument(
                "output_context_key",
                "Output Key",
                PARAM_TYPE_CONTEXT_KEY_OUTPUT,
                "Context key that will receive the frame.",
                default_value="frame",
            ),
            ActionArgument(
                "timeout_ms",
                "Timeout (ms)",
                PARAM_TYPE_INT,
                "Per-frame timeout.",
                default_value=1500,
                min_value=100,
                max_value=10000,
                step=100,
            ),
        ],
    ),
    "wait_for_stable_frame": ActionDefinition(
        id="wait_for_stable_frame",
        display_name="Wait For Stable Frame",
        category="Image Acquisition",
        description="Waits for the frame dimension to stabilize after an acquisition change.",
        execute_func=cast(Callable[..., ActionResult], execute_wait_for_stable_frame),
        arguments=[
            ActionArgument("output_context_key", "Output Key", PARAM_TYPE_CONTEXT_KEY_OUTPUT,
                           "Context key to store the stable frame.", default_value="frame", required=True),
            ActionArgument("stability_count", "Stability Count", PARAM_TYPE_INT,
                           "Number of consecutive frames that must have the same shape.", default_value=3, min_value=2),
            ActionArgument("poll_interval_ms", "Poll Interval (ms)", PARAM_TYPE_INT,
                           "How often to check for a new frame.", default_value=50, min_value=10),
            ActionArgument("timeout_ms", "Timeout (ms)", PARAM_TYPE_INT, "Maximum time to wait for a stable frame.",
                           default_value=2000, min_value=500),
        ]
    ),
    "repeat_block_start": ActionDefinition(
        id="repeat_block_start",
        display_name="Loop Start",
        category="Flow",
        description="지정한 횟수만큼 코드를 반복하는 루프를 시작합니다.",
        execute_func=cast(Callable[..., ActionResult],
                          execute_repeat_block_start),
        arguments=[
            ActionArgument("count", "Repeat Count", PARAM_TYPE_INT,
                           "루프를 반복할 횟수입니다.", default_value=3, min_value=1),
            ActionArgument("block_id", "Block ID", PARAM_TYPE_STRING,
                           "루프의 고유 ID입니다. 비워두면 자동으로 생성됩니다.", required=False),
        ],
    ),

    "repeat_block_end": ActionDefinition(
        id="repeat_block_end",
        display_name="Loop End",
        category="Flow",
        description="Repeat Block 루프의 끝을 표시하고, 다음 반복으로 점프합니다.",
        execute_func=cast(Callable[..., ActionResult],
                          execute_repeat_block_end),
        arguments=[
            ActionArgument("block_id", "Block ID", PARAM_TYPE_STRING,
                           "쌍을 이루는 'Repeat Block'의 ID와 동일해야 합니다.", required=True),
        ],
    ),

    # ─────────────────────────────────── Camera control ─────────────────────────────────────
    "connect_camera": ActionDefinition(
        id="connect_camera",
        display_name="Connect Camera",
        category="Camera",
        description="Establishes camera connection.",
        execute_func=cast(Callable[..., ActionResult], execute_connect_camera),
    ),

    "disconnect_camera": ActionDefinition(
        id="disconnect_camera",
        display_name="Disconnect Camera",
        category="Camera",
        description="Closes camera connection.",
        execute_func=cast(Callable[..., ActionResult], execute_disconnect_camera),
    ),

    "set_parameter": ActionDefinition(
        id="set_parameter", display_name="Set Parameter", category="Camera",
        description="카메라의 GenICam 파라미터를 설정합니다.",
        execute_func=cast(Callable[..., ActionResult], execute_set_parameter),
        arguments=[
            ActionArgument("parameter_name", "Parameter", PARAM_TYPE_CAMERA_PARAM, "설정할 기능의 이름"),
            ActionArgument("value", "Value", PARAM_TYPE_STRING, "설정할 값"),
        ],
    ),
    "setup_grabber_for_hw_trigger": ActionDefinition(
        id="setup_grabber_for_hw_trigger", display_name="Setup Grabber for HW Trigger", category="Grabber",
        description="그래버를 CIC 모드로 설정하여 하드웨어 트리거를 생성할 준비를 합니다.",
        execute_func=cast(Callable[..., ActionResult], execute_setup_grabber_for_hw_trigger),
    ),

    "execute_grabber_hw_trigger": ActionDefinition(
        id="execute_grabber_hw_trigger", display_name="Execute Grabber HW Trigger", category="Grabber",
        description="그래버에서 단일 하드웨어 트리거 펄스(StartCycle)를 보냅니다. (사전 설정 필요)",
        execute_func=cast(Callable[..., ActionResult], execute_grabber_hw_trigger),
    ),

    "set_trigger_mode": ActionDefinition(
        id="set_trigger_mode",
        display_name="Set Trigger Mode",
        category="Camera",
        description="Sets TriggerMode to 'On' or 'Off'.",
        execute_func=cast(Callable[..., ActionResult], execute_set_trigger_mode),
        arguments=[
            ActionArgument(
                "mode",
                "Mode",
                PARAM_TYPE_ENUM,
                "Trigger mode.",
                default_value="Off",
                options=["On", "Off"],
            )
        ],
    ),

    "camera_reboot": ActionDefinition(
        id="camera_reboot",
        display_name="Camera Reboot",
        category="Camera",
        description="Issues a reboot command (if supported).",
        execute_func=cast(Callable[..., ActionResult], execute_camera_reboot),
    ),

    "set_width_with_increment": ActionDefinition(
        id="set_width_with_increment",
        display_name="Width ±Inc",
        category="Camera",
        description="Changes Width by Inc*multiplier.",
        execute_func=cast(Callable[..., ActionResult], execute_set_width_with_increment),
        arguments=[
            ActionArgument(
                "direction",
                "Direction",
                PARAM_TYPE_ENUM,
                "increase / decrease",
                default_value="decrease",
                options=["increase", "decrease"],
            ),
            ActionArgument(
                "step_multiplier",
                "Step×",
                PARAM_TYPE_INT,
                "Multiplier.",
                default_value=1,
                min_value=1,
                max_value=100,
                step=1,
            ),
            ActionArgument(
                "context_width_key",
                "Context key",
                PARAM_TYPE_CONTEXT_KEY,
                "Key to store current Width.",
                default_value="Width",
                required=False,
            ),
        ],
    ),

    "start_grab": ActionDefinition(
        id="start_grab", display_name="Start Grab", category="Image Acquisition",
        description="이미지 수집(Acquisition)을 시작합니다.",
        execute_func=cast(Callable[..., ActionResult], execute_start_grab),
        arguments=[
            ActionArgument("buffer_count", "Buffers", PARAM_TYPE_INT,
                         description="할당할 내부 버퍼의 개수입니다.", default_value=64)
        ]
    ),
    "stop_grab": ActionDefinition(
        id="stop_grab",
        display_name="Stop Grab",
        category="Image Acquisition",
        description="Stops acquisition.",
        execute_func=cast(Callable[..., ActionResult], execute_stop_grab),
    ),

    "grab_frames": ActionDefinition(
        id="grab_frames",
        display_name="Grab Frames",
        category="Image Acquisition",
        description=(
            "지정한 수의 프레임을 획득합니다. "
            "메모리 최적화가 적용되어 대량의 프레임을 안정적으로 처리할 수 있습니다."
        ),
        execute_func=cast(Callable[..., ActionResult], execute_grab_frames),
        arguments=[
            ActionArgument("frame_count", "Frame Count", PARAM_TYPE_INT,
                           "획득할 프레임의 개수입니다.",
                           default_value=100, min_value=1, max_value=100_000, step=1),
            ActionArgument("timeout_ms", "Timeout (ms)", PARAM_TYPE_INT,
                           "각 프레임을 기다리는 최대 대기 시간(ms)입니다.",
                           default_value=1000, min_value=10, max_value=60_000, step=10),
        ]
    ),
    "execute_command": ActionDefinition(
        id="execute_command",
        display_name="Execute Cmd",
        category="Camera",
        description="Executes a camera command node.",
        execute_func=cast(Callable[..., ActionResult], execute_execute_command),
        arguments=[ActionArgument("command_name", "Command", PARAM_TYPE_CAMERA_PARAM, "Command name.")],
    ),
    "save_image": ActionDefinition(
        id="save_image", display_name="Save Image", category="Image Processing",
        description="컨텍스트의 이미지를 파일로 저장합니다.",
        execute_func=cast(Callable[..., ActionResult], execute_save_image),
        arguments=[
            ActionArgument("frame_context_key", "Frame key", PARAM_TYPE_CONTEXT_KEY,
                         description="저장할 이미지가 있는 컨텍스트 키입니다.", default_value="frame"),
            ActionArgument("filepath", "Path", PARAM_TYPE_FILE_SAVE,
                         description="이미지를 저장할 파일 경로입니다."),
        ],
    ),

    "compare_brightness": ActionDefinition(
        id="compare_brightness",
        display_name="Compare Bright.",
        category="Image Processing",
        description="Compares mean brightness of two frames.",
        execute_func=cast(Callable[..., ActionResult], execute_compare_brightness),
        arguments=[
            ActionArgument(
                "current_brightness_key", "Current key", PARAM_TYPE_CONTEXT_KEY,
                "Key for current value.", default_value="brightness_sd_mean"
            ),
            ActionArgument(
                "previous_brightness_key", "Prev. key", PARAM_TYPE_CONTEXT_KEY,
                "Key for previous value.", default_value="prev_brightness_sd"
            ),
            # ⬇⬇⬇ FLOAT → STRING (min/max 제거)
            ActionArgument(
                "threshold", "Threshold", PARAM_TYPE_STRING,
                "Numeric literal or {ctx_key}.", default_value="10.0"
            ),
        ],
    ),

    "compare_images_advanced": ActionDefinition(
        id="compare_images_advanced",
        display_name="Compare Images (Advanced)",
        category="Image Processing",
        description="Compares two images using a selectable metric (AbsDiffMean, MSE, PSNR, SSIM).",
        execute_func=cast(Callable[..., ActionResult], execute_compare_images_advanced),
        arguments=[
            ActionArgument("frame_context_key_a", "Frame A (Reference)", PARAM_TYPE_CONTEXT_KEY,
                           "The context key for the first image (reference).",
                           default_value="ref_frame"),
            ActionArgument("frame_context_key_b", "Frame B (Current)", PARAM_TYPE_CONTEXT_KEY,
                           "The context key for the second image (this one will be saved on error).",
                           default_value="frame"),
            ActionArgument("metric", "Comparison Metric", PARAM_TYPE_ENUM,
                           "The metric to use for comparison.",
                           default_value="AbsDiffMean", options=["AbsDiffMean", "MSE", "PSNR", "SSIM"]),
            ActionArgument("threshold", "Threshold", PARAM_TYPE_FLOAT,
                           "The failure threshold. For AbsDiffMean/MSE, fail if > threshold. For PSNR/SSIM, fail if < threshold.",
                           default_value=10.0),
            ActionArgument("fail_on_mismatch", "Fail if mismatch?", PARAM_TYPE_BOOL,
                           "If True, the step will fail if images do not match.",
                           default_value=True),
            ActionArgument("save_on_error", "Save on Error?", PARAM_TYPE_BOOL,
                           "If True, saves Frame B if it doesn't match Frame A.",
                           default_value=True),
            ActionArgument("error_image_path", "Error Image Path", PARAM_TYPE_FILE_SAVE,
                           "Path template for the faulty image (Frame B).",
                           default_value="logs/error_images/{cam_id}/mismatch_{{timestamp}}.tiff"),
        ],
    ),
    "calculate_image_stats": ActionDefinition(
        id="calculate_image_stats",
        display_name="Calc Img Stats",
        category="Image Processing",
        description="Calculates mean/std/min/max of an image.",
        execute_func=cast(Callable[..., ActionResult], execute_calculate_image_stats),
        arguments=[
            ActionArgument("frame_context_key", "Frame key", PARAM_TYPE_CONTEXT_KEY, "Key of image.",
                           default_value="frame"),
            ActionArgument("output_context_key_prefix", "Output prefix", PARAM_TYPE_STRING, "Prefix for stats.",
                           default_value="{context.frame_context_key}_stats_", required=False),
            ActionArgument("calculate_mean", "Mean?", PARAM_TYPE_BOOL, "", default_value=True, required=False),
            ActionArgument("calculate_stddev", "StdDev?", PARAM_TYPE_BOOL, "", default_value=True, required=False),
            ActionArgument("calculate_minmax", "Min/Max?", PARAM_TYPE_BOOL, "", default_value=True, required=False),
        ],
    ),
    "detect_scrambled_image": ActionDefinition(
        id="detect_scrambled_image",
        display_name="Detect Scrambled Image (SSIM)",
        category="Image Processing",
        description="Detects structural image corruption (scrambling) using SSIM. Saves the failing image on mismatch.",
        execute_func=cast(Callable[..., ActionResult], execute_detect_scrambled_image),
        arguments=[
            ActionArgument("current_frame_key", "Current Frame Key", PARAM_TYPE_CONTEXT_KEY,
                           "The context key for the current frame to be checked."),
            ActionArgument("reference_frame_key", "Reference Frame Key", PARAM_TYPE_CONTEXT_KEY,
                           "The context key for the known-good reference frame."),
            ActionArgument("ssim_threshold", "SSIM Threshold", PARAM_TYPE_FLOAT,
                           "Lower bound for Structural Similarity Index (0.0 to 1.0).",
                           default_value=0.85, min_value=0.0, max_value=1.0),
            ActionArgument("save_on_error", "Save on Error?", PARAM_TYPE_BOOL,
                           "If True, saves the current frame if it is detected as scrambled.",
                           default_value=True),
            ActionArgument("error_image_path", "Error Image Path", PARAM_TYPE_FILE_SAVE,
                           "Path template for the faulty image.",
                           default_value="logs/error_images/{cam_id}/scrambled_{{timestamp}}.tiff"),
        ]
    ),
    "detect_random_defect_pixel": ActionDefinition(
        id="detect_random_defect_pixel",
        display_name="Detect Random Defect Pixel",
        category="Image Processing",
        description="Finds random defect pixels by comparing with the previous frame. Saves the frame if defects are found.",
        execute_func=cast(Callable[..., ActionResult], execute_detect_random_defect_pixel),
        arguments=[
            ActionArgument("current_frame_key", "Current frame key", PARAM_TYPE_CONTEXT_KEY, "현재 프레임 키.",
                           default_value="frame"),
            ActionArgument("previous_frame_key", "Previous frame key", PARAM_TYPE_CONTEXT_KEY, "이전 프레임 키.",
                           default_value="previous_frame"),
            ActionArgument("threshold_dn", "Threshold (|Δ| DN)", PARAM_TYPE_INT, "절대 Δ DN 임계값.", default_value=5),
            ActionArgument("threshold_sigma", "Threshold (σ×)", PARAM_TYPE_FLOAT, "ROI noise σ 배수 (0=사용 안 함).",
                           default_value=0.0),
            ActionArgument("fail_on_defect", "Fail if count>0?", PARAM_TYPE_BOOL, "결함 픽셀 발견 시 액션 실패 여부.",
                           default_value=True),
            ActionArgument("save_on_error", "Save on Error?", PARAM_TYPE_BOOL,
                           "If True, saves the current frame if any defects are found.",
                           default_value=True),
            ActionArgument("error_image_path", "Error Image Path", PARAM_TYPE_FILE_SAVE,
                           "Path template for the faulty (current) frame.",
                           default_value="logs/error_images/{cam_id}/defect_{timestamp}.tiff"),
            ActionArgument("log_details", "Log details?", PARAM_TYPE_BOOL, "True → (y,x,Δ) CSV/context 저장",
                           default_value=False),
            ActionArgument("max_details_per_frame", "Cap (#/frame)", PARAM_TYPE_INT, "CSV에 기록할 최대 행 수 (0=무제한)",
                           default_value=0, required=False),
            ActionArgument("save_mask", "Save mask?", PARAM_TYPE_BOOL, "True → boolean mask 저장.", default_value=False),
            ActionArgument("roi", "ROI x,y,w,h", PARAM_TYPE_STRING, "옵션 ROI (빈칸=전체).", required=False),
            ActionArgument("count_key", "Count ctx key", PARAM_TYPE_CONTEXT_KEY_OUTPUT, "결함 픽셀 수 저장 키.",
                           default_value="random_defect_pixel_count", required=False),
            ActionArgument("coords_key", "Coords ctx key", PARAM_TYPE_CONTEXT_KEY_OUTPUT, "좌표 리스트 저장 키.",
                           default_value="random_defect_pixel_coords", required=False),
            ActionArgument("mask_key", "Mask ctx key", PARAM_TYPE_CONTEXT_KEY_OUTPUT, "마스크 저장 키.",
                           default_value="random_defect_pixel_mask", required=False),
        ],
    ),


    # ──────────────────────────────── Context utilities ────────────────────────────────────
    "copy_context_value": ActionDefinition(
        id="copy_context_value",
        display_name="Copy Context",
        category="Context Management",
        description="Copies value from one context key to another.",
        execute_func=cast(Callable[..., ActionResult], execute_copy_context_value),
        arguments=[
            ActionArgument("source_key", "Source key", PARAM_TYPE_CONTEXT_KEY, "Existing context key."),
            ActionArgument("target_key", "Target key", PARAM_TYPE_CONTEXT_KEY_OUTPUT, "Destination key."),
        ],
    ),

    "set_context_variable": ActionDefinition(
        id="set_context_variable",
        display_name="Set Context Var",
        category="Context Management",
        description="Stores / updates a context variable.",
        execute_func=cast(Callable[..., ActionResult], execute_set_context_variable),
        arguments=[
            ActionArgument("key", "Key", PARAM_TYPE_CONTEXT_KEY_OUTPUT, "Variable name."),
            ActionArgument("value", "Value", PARAM_TYPE_STRING, "Value (supports {ctx} & eval:)."),
        ],
    ),
    "read_parameter": ActionDefinition(
        id="read_parameter", display_name="Get Parameter", category="Camera",
        description="Reads a feature and stores it in context.",
        execute_func=cast(Callable[..., ActionResult], execute_read_parameter),
        arguments=[
            ActionArgument("feature", "Feature", PARAM_TYPE_CAMERA_PARAM, "Name", required=True),
            ActionArgument("output_context_key", "Ctx key", PARAM_TYPE_CONTEXT_KEY_OUTPUT,
                           "Key to store.", default_value="", required=False),
        ],
    ),

    "assert_feature": ActionDefinition(
        id="assert_feature", display_name="Assert Feature", category="Camera",
        description="Compares feature value with expectation.",
        execute_func=cast(Callable[..., ActionResult], execute_assert_feature),
        arguments=[
            ActionArgument("feature", "Feature", PARAM_TYPE_CAMERA_PARAM, "Name", required=True),
            ActionArgument("expected", "Expected", PARAM_TYPE_STRING, "Expected value", required=True),
        ],
    ),
    "configure_internal_trigger": ActionDefinition(
        id="configure_internal_trigger",
        display_name="Configure Internal Trigger",
        category="Grabber",
        description="Sets CycleGenerator + DLT mapping on the Coaxlink board.",
        execute_func=cast(Callable[..., ActionResult],
                          execute_configure_internal_trigger),
        arguments=[
            ActionArgument("cycle_period_us", "Cycle (µs)",
                           PARAM_TYPE_FLOAT, "Cycle period in µs.",
                           required=True, min_value=0.1),
            ActionArgument("dlt_index", "DLT index",
                           PARAM_TYPE_INT, "DeviceLinkTriggerToolSelector value.",
                           default_value=1, required=False, min_value=0),
        ],
    ),
}

ACTION_REGISTRY["execute_set_parameter"] = ACTION_REGISTRY["set_parameter"]
# ═══════════════════════════════════════════════════════════════════════
#  helper API
# ═══════════════════════════════════════════════════════════════════════
def get_action_definition(action_id: str) -> Optional[ActionDefinition]:
    """Return the ActionDefinition for **action_id** or *None*."""
    return ACTION_REGISTRY.get(action_id)


# ─────────────────────────────────────────────────────────────
# list_available_actions  –  UI / API 호출용
# ─────────────────────────────────────────────────────────────
def list_available_actions(ui_visible: bool = True) -> List[Tuple[str, str, str]]:
    """
    Returns a list of (id, display_name, category) tuples.
    ui_visible=True → deprecated 이나 alias 는 숨김.
    """
    items: List[Tuple[str, str, str]] = []

    for aid, ad in ACTION_REGISTRY.items():
        # 1) UI 모드 → deprecated 숨김
        if ui_visible and ad.deprecated:
            continue
        # 2) alias 는 숨김 (aid ≠ canonical id)
        if aid != ad.id:
            continue
        items.append((aid, ad.display_name, ad.category))

    items.sort(key=lambda t: (t[2].lower(), t[1].lower()))
    return items
