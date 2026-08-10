"""
IMCITS — Live Multi-Camera Stream Ingestion Worker
=================================================
Captures live frames concurrently from multiple video sources:
- DirectShow devices (e.g. "Iriun Webcam", "HP HD Camera")
- Direct Device IDs (e.g. index 0, 1, 2)
- RTSP / HTTP video stream URLs

Uses dedicated background threads per camera with queue management
to ensure zero latency and non-blocking frame retrieval.
"""
from __future__ import annotations

import cv2
import time
import threading
import logging
import numpy as np
from queue import Queue, Empty
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Minimum mean pixel brightness to consider a frame "real" (not a black frame)
_MIN_FRAME_BRIGHTNESS = 5.0
# Number of warmup reads to flush the camera buffer before marking it live
_WARMUP_READS = 10


class CameraStreamWorker:
    """
    Background worker thread for a single live camera stream.
    Continuously reads frames and stores only the freshest frame.
    """

    def __init__(self, camera_id: str, camera_name: str, source: Union[int, str]) -> None:
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.source = source
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.frame_queue: Queue[np.ndarray] = Queue(maxsize=2)
        self.last_frame: Optional[np.ndarray] = None
        self.fps: float = 0.0
        self.is_connected: bool = False
        self._blank_streak: int = 0  # consecutive blank-frame counter

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(
            target=self._capture_loop, daemon=True, name=f"CamThread-{self.camera_id}"
        )
        self.thread.start()
        logger.info(
            f"Started camera worker [{self.camera_id}] ({self.camera_name}) -> Source: {self.source}"
        )

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        logger.info(f"Stopped camera worker [{self.camera_id}] ({self.camera_name})")

    def _open_cap(self) -> Optional[cv2.VideoCapture]:
        """
        Open VideoCapture for numeric index sources.
        Tries CAP_DSHOW first (best on Windows for webcams & Iriun),
        then CAP_MSMF, then default backend.
        Does NOT force resolution so virtual cameras (Iriun) keep their native format.
        """
        cap = None
        src = self.source

        if isinstance(src, str) and src.isdigit():
            src = int(src)

        if isinstance(src, int):
            # Try DirectShow — fastest and most compatible with Iriun on Windows
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not cap or not cap.isOpened():
                cap = cv2.VideoCapture(src, cv2.CAP_MSMF)
            if not cap or not cap.isOpened():
                cap = cv2.VideoCapture(src)
        else:
            # RTSP / HTTP URL
            cap = cv2.VideoCapture(src)

        if cap and cap.isOpened():
            # Keep hardware buffer small so we always get the freshest frame
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # NOTE: Do NOT set resolution here — Iriun virtual driver may ignore it
            # or return black frames when forced to an unsupported size.
            return cap

        return None

    def _flush_warmup(self, cap: cv2.VideoCapture) -> None:
        """
        Read and discard warmup frames so the camera pipeline is fully open
        before we start consuming real frames. Iriun needs ~10-15 reads.
        """
        for _ in range(_WARMUP_READS):
            cap.grab()  # grab without decode — fastest flush
            time.sleep(0.03)

    def _is_blank(self, frame: np.ndarray) -> bool:
        """Return True if the frame is essentially black (camera not streaming yet)."""
        return float(frame.mean()) < _MIN_FRAME_BRIGHTNESS

    def _capture_loop(self) -> None:
        while self.running:
            cap = self._open_cap()
            if not cap:
                self.is_connected = False
                logger.warning(
                    f"[{self.camera_id}] Could not open source {self.source}. Retrying in 3s..."
                )
                time.sleep(3.0)
                continue

            # Flush warmup frames (critical for Iriun & laptop webcam)
            logger.debug(f"[{self.camera_id}] Flushing {_WARMUP_READS} warmup frames...")
            self._flush_warmup(cap)

            self.is_connected = True
            self._blank_streak = 0
            frame_count = 0
            start_time = time.time()

            while self.running and cap.isOpened():
                ret, frame = cap.read()

                if not ret or frame is None:
                    logger.warning(
                        f"Empty frame / read failure on [{self.camera_id}] ({self.camera_name}). Reconnecting..."
                    )
                    break

                # Skip blank / black frames (camera buffer not ready yet)
                if self._is_blank(frame):
                    self._blank_streak += 1
                    if self._blank_streak % 30 == 1:  # log every 30 blank frames
                        logger.debug(
                            f"[{self.camera_id}] Blank frame #{self._blank_streak} — waiting for signal..."
                        )
                    time.sleep(0.05)
                    continue

                self._blank_streak = 0
                self.last_frame = frame

                # Keep queue non-blocking — always drop old frame in favour of newest
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                else:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame)
                    except Empty:
                        pass

                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed >= 1.0:
                    self.fps = frame_count / elapsed
                    frame_count = 0
                    start_time = time.time()

            cap.release()
            self.is_connected = False
            if self.running:
                logger.info(f"[{self.camera_id}] Reconnecting in 2s...")
                time.sleep(2.0)

    def read_frame(self) -> Optional[np.ndarray]:
        """Returns the latest non-blank captured frame or None."""
        if self.last_frame is not None:
            return self.last_frame.copy()
        return None


class MultiCameraManager:
    """
    Manager for orchestrating multiple live camera stream workers.
    """

    def __init__(self) -> None:
        self.workers: Dict[str, CameraStreamWorker] = {}

    def add_camera(self, camera_id: str, camera_name: str, source: Union[int, str]) -> None:
        if camera_id in self.workers:
            self.workers[camera_id].stop()
        worker = CameraStreamWorker(camera_id, camera_name, source)
        self.workers[camera_id] = worker

    def start_all(self) -> None:
        for w in self.workers.values():
            w.start()

    def stop_all(self) -> None:
        for w in self.workers.values():
            w.stop()

    def get_latest_frames(self) -> Dict[str, Tuple[str, Optional[np.ndarray], float, bool]]:
        """
        Returns {camera_id: (camera_name, frame, fps, is_connected)} for all cameras.
        """
        results = {}
        for cid, w in self.workers.items():
            results[cid] = (w.camera_name, w.read_frame(), w.fps, w.is_connected)
        return results
