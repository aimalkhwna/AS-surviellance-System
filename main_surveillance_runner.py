"""
IMCITS — 3-Camera Sequential Trajectory Engine (Iriun-01 -> Laptop-01 -> Iriun-02)
==================================================================================
Camera 1: Shahzeb Mobile (cam-iriun-01)
Camera 2: Laptop Webcam  (cam-laptop-01)
Camera 3: Aimal Mobile   (cam-iriun-02)
"""
from __future__ import annotations

import os
import sys
import time
import socket
import logging
import cv2
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime

# Local module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from live_camera_streamer import MultiCameraManager
from poi_detector_matcher import POIDetectorMatcher
from camera_graph_tracker import CameraGraphTracker

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("IMCITS-Runner")

# ─── Exact 3-Camera Configuration (Iriun-01 -> Laptop-01 -> Iriun-02) ────────
CAMERA_CANDIDATES = [
    {
        "id": "cam-iriun-01",
        "name": "Shahzeb Mobile",
        # Camera 1: Shahzeb Mobile (Index 3, 5)
        "sources": [3, 5],
    },
    {
        "id": "cam-laptop-01",
        "name": "Laptop Webcam",
        # Camera 2: Laptop Webcam (Index 0)
        "sources": [0],
    },
    {
        "id": "cam-iriun-02",
        "name": "Aimal Mobile",
        # Camera 3: Aimal Mobile (Index 1, 2, 4, 6)
        "sources": [1, 2, 4, 6],
    },
]

_CAMERA_WARMUP_SECONDS = 3.0


# ─── Fast Camera Probe with Instant Socket URL Check ─────────────────────────
def probe_camera(source: int | str, warmup_reads: int = 4) -> bool:
    """Probe if camera source yields valid non-black frames without hanging."""
    try:
        # Fast socket check if source is an HTTP/RTSP URL
        if isinstance(source, str) and (source.startswith("http://") or source.startswith("rtsp://")):
            try:
                url_clean = source.split("//")[1].split("/")[0]
                if ":" in url_clean:
                    host, port_str = url_clean.split(":")
                    port = int(port_str)
                else:
                    host = url_clean
                    port = 80 if source.startswith("http://") else 554

                s = socket.create_connection((host, port), timeout=0.08)
                s.close()
            except Exception:
                return False  # URL host unreachable, skip instantly

        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            src_idx = int(source)
            cap = cv2.VideoCapture(src_idx, cv2.CAP_DSHOW)
            if not cap or not cap.isOpened():
                cap = cv2.VideoCapture(src_idx)
        else:
            cap = cv2.VideoCapture(source)

        if not cap or not cap.isOpened():
            return False

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        got_frame = False

        for _ in range(warmup_reads):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                if float(frame.mean()) > 5.0:
                    got_frame = True
                    break
            time.sleep(0.04)

        cap.release()
        return got_frame
    except Exception:
        return False


def discover_cameras(candidates: list) -> list:
    """Assigns each camera role a unique live device index."""
    used_sources: set = set()
    live_cameras = []

    print("\n[Step 1/2] Discovering live camera streams (Shahzeb Mobile -> Laptop Webcam -> Aimal Mobile)...")

    for cam in candidates:
        cam_id = cam["id"]
        cam_name = cam["name"]
        found = None

        for src in cam["sources"]:
            if src in used_sources:
                continue
            print(f"  [{cam_name}] Probing source {src}...", end=" ", flush=True)
            if probe_camera(src):
                print("✅ LIVE")
                found = src
                used_sources.add(src)
                break
            else:
                print("❌ no signal")

        if found is not None:
            live_cameras.append((cam_id, cam_name, found))
        else:
            logger.warning(f"[{cam_name}] No signal found. Camera set to OFFLINE.")

    return live_cameras


# ─── Main Execution ────────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print(" 🎥 IMCITS — Trajectory Predictor (Shahzeb Mobile ➔ Laptop Webcam ➔ Aimal Mobile)")
    print("=" * 78)

    # 1. Initialize Re-ID Identity Engine
    detector = POIDetectorMatcher(
        yolo_model_name="yolov8n.pt",
        targets_dir="targets",
        match_threshold=0.50
    )
    detector.load_targets()

    # 2. Initialize Camera Topology Graph (cam-iriun-01 ➔ cam-laptop-01 ➔ cam-iriun-02)
    tracker = CameraGraphTracker()
    tracker.setup_default_topology()

    # 3. Discover Cameras
    live_cameras = discover_cameras(CAMERA_CANDIDATES)

    if not live_cameras:
        print("\n [ERROR] No live cameras detected!")
        sys.exit(1)

    # 4. Start Camera Workers
    camera_manager = MultiCameraManager()
    for cam_id, cam_name, src in live_cameras:
        camera_manager.add_camera(cam_id, cam_name, src)

    print(f"\n[Step 2/2] Starting {len(live_cameras)} camera worker thread(s)...")
    camera_manager.start_all()

    print(f"  Warming up streams ({_CAMERA_WARMUP_SECONDS:.0f}s)...", end=" ", flush=True)
    time.sleep(_CAMERA_WARMUP_SECONDS)
    print("done.\n")

    print("=" * 78)
    print(" 🚀 REAL-TIME 3-CAMERA PERSON TRACKING & TRAJECTORY PREDICTION ACTIVE")
    print("    • Unique persistent Global IDs assigned to each person.")
    print("    • Real-time trajectory prediction forecasts next expected camera node.")
    print("    • Strict 1-to-1 mutual exclusivity enforced per frame.")
    print("    Press 'q' or 'ESC' on any window to exit.")
    print("=" * 78 + "\n")

    last_alert_time: Dict[str, float] = {}

    try:
        while True:
            latest_frames = camera_manager.get_latest_frames()

            for cam_id, (cam_name, frame, fps, is_connected) in latest_frames.items():
                if frame is None or not is_connected:
                    continue

                display_frame = frame.copy()
                h, w = display_frame.shape[:2]
                now = time.time()

                # Step A: Detect person bounding boxes using YOLO
                raw_detections = detector.detect_persons(frame, conf_threshold=0.45)

                # Step B: Assign Persistent Global Person IDs (Hungarian 1-to-1 matching)
                assigned_detections = detector.assign_identities_for_frame(
                    raw_detections, camera_id=cam_id, timestamp=now
                )

                # Step C: Compute Trajectory Predictions & Render Overlays
                for det in assigned_detections:
                    bbox = det["bbox"]
                    conf = det["conf"]
                    person_id = det["person_id"]
                    display_name = det["display_name"]
                    color = det["color"]
                    x1, y1, x2, y2 = bbox

                    # Update Trajectory Tracking & Predict Next Camera
                    track = tracker.record_sighting(
                        person_id=person_id,
                        camera_id=cam_id,
                        camera_name=cam_name,
                        confidence=det["match_score"]
                    )

                    next_cam_name = track.predicted_next_name or "End of Network"
                    eta_sec = track.predicted_transit_time
                    prob_percent = int(track.predicted_probability * 100)

                    # Draw Bounding Box with persistent color
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                    # Label Line 1: ID & Confidence
                    label1 = f"ID: {display_name} ({conf*100:.0f}%)"
                    # Label Line 2: Trajectory Prediction
                    label2 = f"Next -> {next_cam_name} (~{eta_sec:.0f}s | {prob_percent}%)"

                    # Render Text Pill 1 (Top of Box)
                    (tw1, th1), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(display_frame, (x1, max(0, y1 - 24)), (x1 + tw1 + 8, y1), color, -1)
                    cv2.putText(
                        display_frame, label1, (x1 + 4, max(16, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA
                    )

                    # Render Trajectory Prediction Pill 2 (Bottom of Box)
                    (tw2, th2), _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(display_frame, (x1, y2), (x1 + tw2 + 8, y2 + 20), (30, 30, 30), -1)
                    cv2.putText(
                        display_frame, label2, (x1 + 4, y2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA
                    )

                    # Log Trajectory Prediction Console Alert (every 3.5 seconds per person)
                    if now - last_alert_time.get(f"{person_id}_{cam_id}", 0) >= 3.5:
                        last_alert_time[f"{person_id}_{cam_id}"] = now

                        path_str = " ➔ ".join(track.predicted_path_chain)
                        ts = datetime.fromtimestamp(now).strftime("%H:%M:%S")

                        print("=" * 78)
                        print(f" 🔮 [TRAJECTORY PREDICTION ALERT] [{ts}]")
                        print(f"  Person ID            : {display_name}")
                        print(f"  Current Location     : {cam_name} [{cam_id}]")
                        print(f"  PREDICTED NEXT CAM   : '{next_cam_name}' (ETA: ~{eta_sec:.0f}s | Prob: {prob_percent}%)")
                        print(f"  Forecasted Movement  : {path_str}")
                        print("=" * 78 + "\n")

                # Step D: Status Header Banner
                status_dot = (0, 255, 0) if is_connected else (0, 0, 255)
                banner = f"{cam_name} | FPS: {fps:.1f} | Active Persons: {len(assigned_detections)}"
                cv2.rectangle(display_frame, (0, 0), (w, 35), (25, 25, 25), -1)
                cv2.putText(
                    display_frame, banner, (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
                )
                cv2.circle(display_frame, (w - 20, 18), 8, status_dot, -1)

                # Step E: OpenCV Live Video Display
                try:
                    win_title = f"IMCITS - {cam_name} (Src:{camera_manager.workers[cam_id].source})"
                    cv2.imshow(win_title, display_frame)
                except Exception:
                    pass

            # Keyboard Exit Check
            try:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    print("\nStopping multi-camera tracking engine...")
                    break
            except Exception:
                time.sleep(0.04)

    except KeyboardInterrupt:
        print("\nSurveillance engine stopped by user.")
    finally:
        camera_manager.stop_all()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print(" [SHUTDOWN] All camera streams closed.")


if __name__ == "__main__":
    main()
