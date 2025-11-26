# ViewTS – Vieworks Camera Test Suite

**ViewTS** (Vieworks Test Suite) is a PyQt5-based **camera test and validation suite** for Vieworks cameras using
Euresys Coaxlink + eGrabber. It provides a reproducible sequence engine, rich GUI, and
various utilities for camera bring-up, regression testing, and reliability verification.

The goal of ViewTS is:

> To let **non-programmers** (test engineers, FA/AE, production operators)  
> run **repeatable, automated camera tests** through a friendly UI –  
> while still giving developers a clean, extensible Python codebase.

<img width="940" height="509" alt="image" src="https://github.com/user-attachments/assets/f256a6b1-8891-4d5c-837a-07a883a0c50c" />

---

## Table of Contents

1. [Key Concepts & Benefits](#key-concepts--benefits)  
2. [Typical Use Cases](#typical-use-cases)  
3. [High-level Architecture](#high-level-architecture)  
4. [UI Overview – How to Use ViewTS Without Coding](#ui-overview--how-to-use-viewts-without-coding)  
   - [Main Window](#main-window)  
   - [Device Manager Dock](#device-manager-dock)  
   - [Camera Status Dock](#camera-status-dock)  
   - [Live View & Mosaic View](#live-view--mosaic-view)  
   - [Sequence Editor](#sequence-editor)  
   - [Error Images Tab](#error-images-tab)  
5. [Core Engine & Internals](#core-engine--internals)  
6. [Requirements & Installation](#requirements--installation)  
7. [Running ViewTS](#running-viewts)  
8. [Extending ViewTS (for Developers)](#extending-viewts-for-developers)  
9. [Limitations & Notes](#limitations--notes)

---

## Key Concepts & Benefits

### What ViewTS does

- **Connects and manages multiple cameras** via Coaxlink + eGrabber.
- Provides a **real-time tiled live view** with automatic down-scaling in a worker thread.
- Executes **JSON-based test sequences** (connect → configure → grab → assert → log).
- Captures and archives **error images** for failed steps or abnormal frames.
- Logs test runs into structured logs (CSV + text logs).

### Who it is for

- **Camera R&D / Validation engineers**  
  - Create repeatable test scenarios (link test, exposure sweep, thermal test…)
- **Field application & support engineers (FAE/AE)**  
  - Quickly reproduce issues on customer setups  
  - Provide portable test projects for customers
- **Production line / reliability engineers**  
  - Long-run stress tests, link stability checks, frame loss monitoring
- **Non-programmer operators**  
  - Run pre-defined test sequences via the GUI without writing code

### Key benefits

- ✅ **No scripting required for users** – sequences are data-driven (JSON); users operate via UI  
- ✅ **Repeatable** – the same sequence can be run by different people & sites  
- ✅ **Traceable** – logs + error images show exactly what happened and when  
- ✅ **Extendable** – developers can add new actions in Python without touching UI code  
- ✅ **Safe** – sequence engine supports error policies, loops, and guard rails

---

## Typical Use Cases

1. **Smoke test for a new camera**
   - Connect camera, open live view, verify FPS, exposure, link status.
   - Run a “basic_camera_test” sequence from the backtest examples.

2. **Reliability / burn-in test**
   - Run a long sequence that:
     - Changes exposure / gain / trigger modes
     - Performs continuous grabbing
     - Monitors frame loss and error counters
     - Captures error images if something goes wrong

3. **Regression test for firmware / driver updates**
   - Reuse the same sequences across firmware or SDK versions.
   - Compare logs and error images before/after changes.

4. **Customer reproduction**
   - Encode customer’s scenario as a JSON sequence.
   - Share a `ViewTS` bundle; the customer runs the same test without coding.

---

## High-level Architecture

```text
ViewTS/
└─ ViewTS-main/
   ├─ main.py                 # Application launcher
   ├─ vieworks.ico            # Windows icon
   └─ src/
      ├─ backtest/            # Example/test sequences (JSON + helper scripts)
      ├─ core/                # Core engine: actions, sequences, controllers
      │  ├─ camera_controller.py
      │  ├─ controller_pool.py
      │  ├─ action_registry.py
      │  ├─ actions_base.py
      │  ├─ actions_impl.py
      │  ├─ sequence_types.py
      │  ├─ sequence_runner.py
      │  ├─ sequence_coordinator.py
      │  ├─ multi_runner_manager.py
      │  ├─ error_image_manager.py
      │  ├─ events.py
      │  ├─ initializer.py
      │  └─ logging_config.py
      ├─ ui/                  # PyQt5 widgets & windows
      │  ├─ main_window.py
      │  ├─ grab_worker.py
      │  ├─ live_view_widget.py
      │  ├─ mosaic_live_widget.py
      │  ├─ camera_settings_dialog.py
      │  ├─ camera_status_dock.py
      │  ├─ device_manager_dock.py
      │  ├─ sequence_editor_widget.py
      │  ├─ error_images_tab.py
      │  └─ run_safe.py
      └─ utils/               # Helper modules
         ├─ image_utils.py
         ├─ logging_setup.py
         ├─ camera_parameters_model.py
         ├─ csv_logger.py
         ├─ memento_recorder.py
         └─ sandbox.py
```

At a glance:

- **`core/`** – “brain” of the system (cameras, actions, sequences, error images).
- **`ui/`** – visual layer only; talks to `core` via signals/slots and public APIs.
- **`utils/`** – logging, image helpers, camera parameter model, CSV logging.
- **`backtest/`** – real-world test scenarios as JSON + scripts.

---

## UI Overview – How to Use ViewTS Without Coding

This section explains the UI from a **non-programmer user** perspective.

### Main Window

The main window (from `src/ui/main_window.py`) acts as a dashboard:

- Top menu / toolbar (depending on configuration)
- Left/right **docks**:
  - Device Manager
  - Camera Status
  - Sequence Editor
- Central area:
  - Live View or Mosaic Live view
- Bottom:
  - Status bar (current sequence, camera status, hints)

You can **rearrange docks** (drag, float, dock) just like in typical Qt applications.

---

### Device Manager Dock

Implemented in `device_manager_dock.py`.

Typical capabilities:

- **List of available cameras**
  - Shows device IDs, model names, serials, etc. (depending on SDK)
- **Connect / Disconnect buttons**
  - Select a camera → click *Connect* to open it
  - Disconnect when you want to free the resource or switch cables
- **Link information**
  - Link status (connected / disconnected)
  - Optional link speed / lane configuration (depending on how it is wired to `CameraController`)

**Why it’s useful (for non-programmers):**

> You don’t need to know SDK APIs or run sample code.  
> Just pick your camera from a list and click **Connect**.

---

### Camera Status Dock

Implemented in `camera_status_dock.py`.

Shows live, read-only status information per camera, such as:

- Current **FPS** and frame count
- Exposure / gain (sometimes also ROI, binning, etc.)
- Link count / link state
- Event counters (errors, frame drops, resends…)

**Typical usage:**

- Verify that the camera is **running at expected FPS**.  
- Check that exposure / gain are in the expected range.  
- During long tests, keep an eye on **frame loss counters**.

---

### Live View & Mosaic View

Implemented in:

- `live_view_widget.py` – per-camera live widget  
- `mosaic_live_widget.py` – grid/tiled view  
- `grab_worker.py` – background thread feeding frames to the UI  

Features:

- **Real-time display** of camera images  
- Automatic **down-scaling in a worker thread** (using NumPy + optional `cv2`)  
- Each tile can:
  - Show last frame
  - Resize smoothly as you resize the main window
  - Update target width back to `GrabWorker` so it knows how big to scale

**Why it’s nice:**

- UI remains **responsive** even with multiple cameras, because all heavy image
  processing happens in `GrabWorker` (a `QThread`) instead of the main thread.
- You can **visually confirm**:
  - Exposure changes
  - Motion blur
  - Trigger timing
  - Line/area sensor behavior

---

### Sequence Editor

Implemented in `sequence_editor_widget.py`, backed by:

- `core/action_registry.py`  
- `core/sequence_types.py`  
- `core/sequence_runner.py`  
- `core/sequence_coordinator.py`  

Concept:

> A **sequence** is a list of **steps**.  
> Each step calls an **action** with some parameters (e.g. “connect camera”, “set exposure”, “grab N frames”, “assert feature value”).

From the UI, you typically see:

- A **list/table of steps** (ID, action name, parameters, notes).  
- Buttons:
  - Load sequence from JSON
  - Save sequence to JSON
  - Run / Stop
- A log pane for sequence messages (depending on configuration).

Examples of actions:

- `connect_camera`  
- `disconnect_camera`  
- `set_parameter` (e.g. ExposureTime, Gain)  
- `assert_feature` (e.g. check LinkCount == 2)  
- `start_grab`, `stop_grab`, `flush_and_grab`  
- Custom test actions defined in `actions_impl.py`  

**For non-programmers:**

- You **edit steps in a structured table**, not in code.  
- JSON is used as storage, but you don’t have to hand-edit JSON –  
  you can do most work via UI and only export/import files as needed.  
- Loops, error policies (`continue_on_fail`) and context variables are handled
  internally – the UI just exposes the important knobs.

---

### Error Images Tab

Implemented in `error_images_tab.py`, using logic from `error_image_manager.py`.

Behavior:

- Whenever a sequence detects an anomaly or assertion failure, the system can:
  - **Save the offending frame** to disk  
  - Record metadata about which step/test it belongs to
- The Error Images tab then:
  - Lists saved error images  
  - Shows basic metadata (timestamp, camera ID, test name, etc.)  
  - Allows the user to open or inspect error frames  

Why this matters:

- For reliability tests, it’s not enough to know “something failed” –  
  you want to **see the actual image** that caused the failure.  
- You can later reuse error frames in:
  - Bug reports  
  - Internal analysis  
  - Comparative tests across firmware/hardware versions  

---

## Core Engine & Internals

This section is mainly for developers and power users.

### Camera Controller & Pool

- `camera_controller.py`  
  - Wraps eGrabber (real or dummy) to provide:
    - Discover / connect / disconnect
    - Start/stop grabbing
    - Frame acquisition (single or continuous)
    - Parameter access (`get_param`, `set_param`)
  - Holds logic for:
    - Handling DMA/driver queues
    - Integrating with Euresys Memento (via `_find_memento_cli()` etc.)

- `controller_pool.py`  
  - Global registry of `CameraController` instances  
  - Provides `list_ids()`, `get(id)`, `broadcast(...)`  
  - Keeps multi-camera setups manageable

### Action System

- `actions_base.py`  
  - Defines `ActionDefinition`, `ActionArgument`, `ActionResult`, etc.

- `action_registry.py`  
  - Central registry mapping action IDs → definitions  
  - Used by both the **sequence engine** and the **UI** (for validation and labels)

- `actions_impl.py`  
  - Concrete `execute_*` functions  
  - Uses `egrabber.query` and `CameraController` to:
    - Change GenICam features
    - Perform grabs
    - Monitor frame loss, etc.

### Sequence Engine

- `sequence_types.py`  
  - Dataclasses representing:
    - `SequenceStep`  
    - `Sequence`  
    - Loop expressions and exit conditions  
    - Error policy flags (`continue_on_fail`, etc.)  
  - JSON (de-)serialization helpers so sequences can be stored in files.

- `sequence_runner.py`  
  - Executes sequences step-by-step.

- `sequence_coordinator.py`  
  - Orchestrates multiple sequences and manages shared context.

- `multi_runner_manager.py`  
  - Manages multiple runners (e.g., for multi-camera or multi-sequence scenarios).

### Utilities

- `image_utils.py`  
  - `fast_resize`, `make_thumbnail` for light-weight scaling  
  - `save_frame`, `archive_error_image` (robust BMP saves with dtype handling)  
  - SSIM-based comparison & scrambled-image detection (via `skimage.metrics.ssim`)

- `camera_parameters_model.py`  
  - Parses camera XML files to build a parameter map  
  - Provides high-level helpers for UI controls (drop-downs, etc.)

- `csv_logger.py`  
  - Structured CSV logging for test results

- `logging_setup.py`  
  - Unified logging configuration (console + rotating files)

---

## Requirements & Installation

### Python & OS

- **Python**: 3.8+ recommended  
- **OS**: Windows is the primary target (Coaxlink + eGrabber),  
  though parts of the code can run on other platforms for development.

### Python dependencies

From the source code under `src/`, the project relies on the following external libraries:

- `PyQt5` – GUI (Qt Widgets, Core, Gui)  
- `numpy` – frame buffers and numeric work  
- `opencv-python` (`cv2`) – fast resizing and image conversions for thumbnails/live view  
- `scikit-image` – SSIM-based comparison (`skimage.metrics.structural_similarity`)  
- `egrabber` – Euresys Coaxlink / eGrabber camera SDK (Python bindings)  

A concrete `requirements.txt` aligned with your environment might look like:

```text
PyQt5==5.15.11
numpy==1.24.4
opencv-python==4.11.0.86
scikit-image==0.21.0
egrabber==25.3.2.80
```

> **Note**  
> - `GenApi` and related components are usually installed as part of the camera SDK  
>   and are not normally installed via `pip`.  
> - `PyQt5-Qt5` and `PyQt5_sip` are pulled in automatically with `PyQt5`.

### SDK & Drivers

- Install **Euresys Coaxlink** drivers and **eGrabber SDK** (including Python bindings).  
- Ensure GenICam/GenApi components are correctly installed and visible to the SDK.

---

## Running ViewTS

From the `ViewTS-main` directory:

```bash
pip install -r requirements.txt
python main.py
```

If the SDK and drivers are correctly installed and a camera is connected:

1. The launcher will:
   - Initialize logging  
   - Initialize the hardware (`src/core/initializer.initialize_system()`)  
   - Open the main window  

2. The **Device Manager Dock** should list available cameras.  
3. Connect a camera and check the **Live View** and **Camera Status**.

You can start by:

- Running a **basic test sequence** from `src/backtest/` (e.g., link trigger test).  
- Watching **frame loss counters** and **error images** for anomalies.

---

## Extending ViewTS (for Developers)

If you want to customize ViewTS:

### Add a new action

1. Define the action in `actions_base.py` or register it via `action_registry.py`.  
2. Implement `execute_<your_action>(...)` in `actions_impl.py`.  
3. Register it with an `ActionDefinition` so that:
   - The **sequence engine** can call it.  
   - The **UI** (sequence editor) knows its parameters and labels.

### Add a custom sequence

1. Create a JSON file under `src/backtest/` (or anywhere else).  
2. Load it via the **Sequence Editor** UI.  
3. Save/modify using the UI, or edit JSON manually for advanced flows.

### Modify the UI

- `main_window.py` orchestrates which docks and tabs are visible.  
- You can:
  - Add a new dock  
  - Add new menu/toolbar actions  
  - Connect signals/slots to your custom logic  

---

## Limitations & Notes

- ViewTS targets **Euresys Coaxlink + eGrabber** ecosystems.  
  Other frame grabbers or camera SDKs would require porting `camera_controller.py`
  and related parts.
- Without physical cameras and drivers:
  - Some parts of the UI can run in “dummy” mode.  
  - However, most real tests and sequences require actual hardware.
- The sequence engine is powerful (loops, conditionals, error policies), so:
  - For non-programmers, it is recommended to start from **example sequences**
    in `src/backtest/` rather than building complex flows from scratch.

---

If you have specific usage scenarios (e.g., “test link recovery after cable pull”,
“exposure sweep across 1000 frames”), you can encode them as sequences and reuse
them across teams and sites – that’s the core strength of ViewTS.
