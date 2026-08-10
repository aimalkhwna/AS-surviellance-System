"""
IMCITS — Ultra-Reliable Global Person Re-Identification & Identity Bank
========================================================================

Pipeline (per camera, per frame):

  Camera → YOLOv8 Person Detection → ByteTrack (local track id)
         → Person Crop → InsightFace Face Detection → ArcFace 512-D Embedding
         → Global Gallery (cosine similarity, Hungarian 1-to-1 matching)
         → Global ID (Person #101, #102, ... / TARGET_<name>)

1. YOLOv8 Person Detection: high-precision person bounding boxes.
2. Per-Camera ByteTrack: keeps a stable *local* track id for each person
   within one camera's frames, so the expensive face-embedding step below
   only has to run once per person per camera visit instead of every frame.
3. InsightFace + ArcFace Face Recognition:
   - Detects the face inside each person crop (SCRFD detector, bundled
     with the `buffalo_l` InsightFace model pack).
   - Extracts a 512-d L2-normalized ArcFace embedding from that face.
   - Unlike an ImageNet-pretrained body embedder, ArcFace is a model
     purpose-trained for face verification — cosine similarity between two
     embeddings of the same person's face is meaningfully high (typically
     0.4-0.7+) and low for different people, out of the box, with no
     fine-tuning required.
   - If no face is found in a crop (person facing away, too small,
     occluded), that detection simply isn't matched/embedded this frame —
     it does not crash and does not force a bad match.
4. Thread-Safe Lifetime Global ID Reservation:
   - Monotonic ID incrementer (Person #101, Person #102, Person #103...).
   - Thread-locked execution ensures IDs are minted once and strictly locked.
   - Hungarian matching (scipy linear_sum_assignment) enforces 1-to-1 mutual
     exclusivity per frame, so two people in the same frame can never be
     assigned to the same identity.
"""

from __future__ import annotations

import os
import cv2
import glob
import logging
import random
import threading
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


from scipy.optimize import linear_sum_assignment

# Suppress TensorFlow / PyTorch logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logger = logging.getLogger(__name__)

# Lazy imports for fast startup / graceful degradation if a dependency is missing
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import torch
except Exception:
    torch = None

try:
    # pip install insightface onnxruntime  (or onnxruntime-gpu for CUDA)
    from insightface.app import FaceAnalysis
except Exception as e:
    FaceAnalysis = None
    _insightface_import_error = e
else:
    _insightface_import_error = None

try:
    import supervision as sv
except ImportError:
    sv = None


class _FaceEmbedder:
    """InsightFace face detector + ArcFace recognition wrapper.

    Person crops (BGR, any size) in, 512-d L2-normalized ArcFace face
    embeddings out — or None for a given crop if no face was found in it.
    Unlike a generic body-appearance embedder, this is a model actually
    trained for face verification, so it doesn't need a fine-tuned
    checkpoint to produce meaningful cosine similarities.
    """

    def __init__(self, model_name: str = "buffalo_l", det_size: Tuple[int, int] = (320, 320), use_gpu: Optional[bool] = None):
        if FaceAnalysis is None:
            raise RuntimeError(
                "insightface is not installed or failed to import "
                f"({_insightface_import_error}) - pip install insightface onnxruntime"
            )

        if use_gpu is None:
            use_gpu = bool(torch is not None and torch.cuda.is_available())

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        ctx_id = 0 if use_gpu else -1

        logger.info(f"Loading InsightFace model pack '{model_name}' (gpu={use_gpu})...")
        self._app = FaceAnalysis(name=model_name, providers=providers)
        self._app.prepare(ctx_id=ctx_id, det_size=det_size)
        logger.info("InsightFace ready.")

    def embed_one(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        try:
            faces = self._app.get(crop_bgr)
        except Exception as e:
            logger.debug(f"InsightFace inference error on a crop: {e}")
            return None
        if not faces:
            return None
        # If more than one face landed in a single person crop (rare, but
        # possible with a loose YOLO box), keep the largest — almost always
        # the actual person the box was drawn around.
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        vec = getattr(best, "normed_embedding", None)
        if vec is None:
            return None
        return np.asarray(vec, dtype=np.float32)

    def embed(self, crops: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """Batch entry point kept for interface symmetry with the rest of
        the pipeline — InsightFace's ONNX runtime session isn't easily
        batched across arbitrary crop sizes, so this just loops embed_one().
        Returns a list the same length as `crops`; entries are None where
        no face was found in that particular crop."""
        return [self.embed_one(c) for c in crops]


class GlobalPersonIdentity:
    """Represents a unique physical person locked to a persistent Global ID."""

    def __init__(self, person_id: str, is_target: bool = False, display_name: Optional[str] = None) -> None:
        self.person_id = person_id  # e.g., 'TARGET_JohnDoe' or 'Person #101'
        self.display_name = display_name if display_name else person_id
        self.is_target = is_target
        self.embeddings: List[np.ndarray] = []  # gallery of ArcFace embeddings (512-d, L2-normalized)
        self.mean_embedding: Optional[np.ndarray] = None
        self.max_gallery_size = 15
        self.last_seen_camera: str = "Unknown"
        self.last_seen_time: float = 0.0
        self.last_bbox: Optional[Tuple[int, int, int, int]] = None

        # Unique deterministic bounding box color (B, G, R)
        if is_target:
            self.color = (0, 0, 255)  # Bright RED for target POIs
        else:
            # Deterministic color palette per ID for maximum visual clarity
            palette = [
                (255, 191, 0),   # Deep Cyan / Amber
                (200, 50, 200),  # Bright Purple / Magenta
                (0, 230, 255),   # Bright Yellow
                (255, 105, 180), # Neon Pink
                (50, 205, 50),   # Lime Green
                (0, 140, 255),   # Orange
                (238, 130, 238), # Violet
                (255, 215, 0),   # Gold
            ]
            try:
                num = int(person_id.split("#")[-1])
                self.color = palette[(num - 101) % len(palette)]
            except Exception:
                rng = random.Random(hash(person_id))
                self.color = (rng.randint(60, 240), rng.randint(60, 240), rng.randint(60, 240))

    def add_embedding(self, vec: np.ndarray) -> None:
        if vec is None or vec.size == 0:
            return
        self.embeddings.append(vec)
        if len(self.embeddings) > self.max_gallery_size:
            self.embeddings.pop(0)

        mean_vec = np.mean(np.stack(self.embeddings, axis=0), axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm <= 0.0:
            norm = 1e-6
        self.mean_embedding = mean_vec / norm

    def match_score(self, query_vec: np.ndarray) -> float:
        """Returns the strongest cosine similarity against gallery embeddings."""
        if not self.embeddings or query_vec is None:
            return 0.0

        max_sim = 0.0
        for ref_vec in self.embeddings:
            if query_vec.shape == ref_vec.shape:
                sim = float(np.dot(query_vec, ref_vec))
                if sim > max_sim:
                    max_sim = sim

        if self.mean_embedding is not None and query_vec.shape == self.mean_embedding.shape:
            centroid_sim = float(np.dot(query_vec, self.mean_embedding))
            if centroid_sim > max_sim:
                max_sim = centroid_sim

        return max_sim


class POIDetectorMatcher:
    """
    YOLO Person Detector + Per-Camera ByteTrack + InsightFace/ArcFace Face
    Re-ID Identity Manager.
    """

    def __init__(
        self,
        yolo_model_name: str = "yolov8n.pt",
        targets_dir: str = "targets",
        match_threshold: float = 0.45,
        face_model_name: str = "buffalo_l",
        face_det_size: Tuple[int, int] = (320, 320),
        same_camera_stale_sec: float = 6.0,
        cross_camera_stale_sec: float = 45.0,
    ) -> None:
        self.targets_dir = targets_dir
        self.match_threshold = match_threshold
        self.acceptance_cost = 1.0 - self.match_threshold
        self.same_camera_stale_sec = same_camera_stale_sec
        self.cross_camera_stale_sec = cross_camera_stale_sec
        self.yolo_model = None
        self.identities: Dict[str, GlobalPersonIdentity] = {}
        self.auto_id_counter: int = 101
        self.lock = threading.Lock()

        # Load YOLO model
        if YOLO is not None:
            try:
                logger.info(f"Loading YOLO model '{yolo_model_name}'...")
                self.yolo_model = YOLO(yolo_model_name)
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
        else:
            logger.warning("ultralytics package missing. OpenCV HOG fallback active.")

        # Load InsightFace / ArcFace embedder
        self.embedder: Optional[_FaceEmbedder] = None
        if FaceAnalysis is not None:
            try:
                self.embedder = _FaceEmbedder(model_name=face_model_name, det_size=face_det_size)
            except Exception as e:
                logger.error(f"Failed to load InsightFace embedder: {e}")
        else:
            logger.warning(
                f"insightface package missing ({_insightface_import_error}). "
                "Every detection will mint a new identity — install with: "
                "pip install insightface onnxruntime"
            )

        # Per-camera ByteTrack instances + the local(camera) -> global(person_id) map they feed.
        self._trackers: Dict[str, object] = {}
        self._local_to_global: Dict[str, Dict[int, str]] = {}
        if sv is None:
            logger.warning("supervision package missing. Falling back to per-frame Re-ID matching (no ByteTrack).")

    # ---------- ByteTrack ----------

    def _get_tracker(self, camera_id: str):
        if sv is None:
            return None
        tracker = self._trackers.get(camera_id)
        if tracker is None:
            tracker = sv.ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=30,
                minimum_matching_threshold=0.8,
                frame_rate=15,
            )
            self._trackers[camera_id] = tracker
        return tracker

    def _track_locally(self, camera_id: str, valid_detections: List[Dict[str, Any]]) -> List[Optional[int]]:
        """Returns a per-detection local ByteTrack id (or None if unavailable/unassociated)."""
        tracker = self._get_tracker(camera_id)
        if tracker is None:
            return [None] * len(valid_detections)

        boxes = np.array([d["bbox"] for d in valid_detections], dtype=np.float32)
        confs = np.array([d["conf"] for d in valid_detections], dtype=np.float32)
        tracked = tracker.update_with_detections(
            sv.Detections(xyxy=boxes, confidence=confs, class_id=np.zeros(len(boxes), dtype=int))
        )
        if len(tracked) != len(valid_detections):
            return [None] * len(valid_detections)
        return [int(tid) for tid in tracked.tracker_id]

    def load_targets(self) -> int:
        """Loads pre-enrolled reference photos from targets_dir."""
        if not os.path.exists(self.targets_dir):
            os.makedirs(self.targets_dir, exist_ok=True)
            return 0

        image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        target_files = []
        for ext in image_extensions:
            target_files.extend(glob.glob(os.path.join(self.targets_dir, ext)))
            target_files.extend(glob.glob(os.path.join(self.targets_dir, "*", ext)))

        loaded_count = 0
        for file_path in target_files:
            poi_name = os.path.basename(os.path.dirname(file_path))
            if not poi_name or poi_name == os.path.basename(self.targets_dir):
                poi_name = os.path.splitext(os.path.basename(file_path))[0]

            img = cv2.imread(file_path)
            if img is None:
                continue

            vec = self.extract_invariant_feature_vector(img)
            if vec is None:
                logger.warning(f"No face found in reference photo '{os.path.basename(file_path)}' for '{poi_name}' — skipped.")
                continue

            pid = f"TARGET_{poi_name}"
            if pid not in self.identities:
                self.identities[pid] = GlobalPersonIdentity(person_id=pid, is_target=True, display_name=poi_name)
            self.identities[pid].add_embedding(vec)
            loaded_count += 1
            logger.info(f"Enrolled target POI '{poi_name}' from {os.path.basename(file_path)}")

        logger.info(f"Total POIs enrolled: {len(self.identities)} target identities")
        return loaded_count

    def extract_invariant_feature_vector(self, crop_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts a 512-d L2-normalized ArcFace face embedding from a person
        crop (or a reference photo). Returns None if no face is found —
        same method name/signature as before so load_targets() and any
        other caller keep working unchanged.
        """
        if crop_img is None or crop_img.size == 0 or crop_img.shape[0] < 20 or crop_img.shape[1] < 20:
            return None
        if self.embedder is None:
            return None
        return self.embedder.embed_one(crop_img)

    def detect_persons(self, frame: np.ndarray, conf_threshold: float = 0.40) -> List[Dict[str, Any]]:
        """
        Runs YOLO person detection at optimal resolution.
        Returns list of detected person bounding boxes and crops sorted left-to-right.
        """
        detections = []
        if frame is None or frame.size == 0:
            return detections

        h, w = frame.shape[:2]

        if self.yolo_model is not None:
            try:
                infer_frame = frame
                scale = 1.0
                if max(h, w) > 960:
                    scale = 640.0 / max(h, w)
                    infer_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

                results = self.yolo_model.predict(
                    source=infer_frame, conf=conf_threshold, classes=[0], verbose=False
                )
                for res in results:
                    boxes = res.boxes
                    if boxes is None:
                        continue
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0].item())
                        if scale != 1.0:
                            x1, y1, x2, y2 = int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        if (x2 - x1) > 20 and (y2 - y1) > 20:
                            crop = frame[y1:y2, x1:x2]
                            detections.append({
                                "bbox": (x1, y1, x2, y2),
                                "conf": conf,
                                "crop": crop,
                            })
                detections.sort(key=lambda d: d["bbox"][0])
                return detections
            except Exception as e:
                logger.error(f"YOLO inference error: {e}")

        # Fallback OpenCV HOG Detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(4, 4), scale=1.05)
        for (x, y, bw, bh), weight in zip(boxes, weights):
            if weight >= 0.2:
                x1, y1, x2, y2 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
                crop = frame[y1:y2, x1:x2]
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "conf": float(weight),
                    "crop": crop,
                })
        detections.sort(key=lambda d: d["bbox"][0])
        return detections

    def _score_identity_match(self, identity: GlobalPersonIdentity, camera_id: str, bbox: Tuple[int, int, int, int], frame_width: int, frame_height: int, similarity: float) -> float:
        same_camera = identity.last_seen_camera == camera_id

        max_gap = self.same_camera_stale_sec if same_camera else self.cross_camera_stale_sec
        if identity.last_seen_time and (time.time() - identity.last_seen_time) > max_gap:
            return -1.0

        if not same_camera or identity.last_bbox is None:
            return similarity

        prev_x1, prev_y1, prev_x2, prev_y2 = identity.last_bbox
        curr_x1, curr_y1, curr_x2, curr_y2 = bbox
        prev_cx = (prev_x1 + prev_x2) / 2.0
        prev_cy = (prev_y1 + prev_y2) / 2.0
        curr_cx = (curr_x1 + curr_x2) / 2.0
        curr_cy = (curr_y1 + curr_y2) / 2.0
        movement = ((curr_cx - prev_cx) ** 2 + (curr_cy - prev_cy) ** 2) ** 0.5
        max_allowed_movement = max(90.0, max(frame_width, frame_height) * 0.25)
        if movement > max_allowed_movement:
            return -1.0
        spatial_score = max(0.0, 1.0 - (movement / max_allowed_movement))
        return (similarity * 0.7) + (spatial_score * 0.3)

    def assign_identities_for_frame(
        self, detections: List[Dict[str, Any]], camera_id: str, timestamp: float, frame_width: Optional[int] = None, frame_height: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Thread-Safe ByteTrack + Hungarian Mutual Exclusivity Assignment & Persistent ID Reservation.
        """
        if not detections:
            return []

        with self.lock:
            valid_detections = [det for det in detections if float(det.get("conf", 0.0)) >= 0.55]
            if not valid_detections:
                return []

            frame_width = frame_width or 640
            frame_height = frame_height or 480

            local_ids = self._track_locally(camera_id, valid_detections)
            local_map = self._local_to_global.setdefault(camera_id, {})

            assigned_results: Dict[int, dict] = {}
            unresolved_idx: List[int] = []

            for i, local_id in enumerate(local_ids):
                if local_id is not None and local_id in local_map:
                    pid = local_map[local_id]
                    identity = self.identities.get(pid)
                    if identity is not None:
                        identity.last_seen_camera = camera_id
                        identity.last_seen_time = timestamp
                        identity.last_bbox = valid_detections[i]["bbox"]
                        assigned_results[i] = {"identity": identity, "match_score": 1.0, "is_new": False}
                        continue
                unresolved_idx.append(i)

            # Slow path: run the face embedder only on detections that actually need a Re-ID lookup.
            embeddings: Dict[int, np.ndarray] = {}
            if unresolved_idx and self.embedder is not None:
                crops = [valid_detections[i]["crop"] for i in unresolved_idx]
                vecs = self.embedder.embed(crops)  # list, entries may be None (no face found)
                for k, i in enumerate(unresolved_idx):
                    embeddings[i] = vecs[k]

            known_id_list = list(self.identities.keys())
            if unresolved_idx and known_id_list:
                cost = np.ones((len(unresolved_idx), len(known_id_list)), dtype=np.float32)
                debug_sims: Dict[Tuple[int, int], float] = {}
                for r, i in enumerate(unresolved_idx):
                    vec = embeddings.get(i)
                    if vec is None:
                        continue  # no face found in this crop — can't be matched or minted with an embedding
                    for c, pid in enumerate(known_id_list):
                        identity = self.identities[pid]
                        sim = identity.match_score(vec)
                        if sim < 0.20:
                            continue
                        score = self._score_identity_match(identity, camera_id, valid_detections[i]["bbox"], frame_width, frame_height, sim)
                        cost[r, c] = 1.0 - score
                        debug_sims[(r, c)] = sim

                row_idx, col_idx = linear_sum_assignment(cost)
                for r, c in zip(row_idx, col_idx):
                    i = unresolved_idx[r]
                    pid = known_id_list[c]
                    sim_dbg = debug_sims.get((r, c))
                    if cost[r, c] > self.acceptance_cost:
                        if sim_dbg is not None:
                            logger.debug(
                                f"[cam {camera_id}] det#{i} REJECTED best candidate {pid}: "
                                f"sim={sim_dbg:.3f} cost={cost[r, c]:.3f} "
                                f"(need cost<={self.acceptance_cost:.3f}, i.e. sim>={self.match_threshold:.2f})"
                            )
                        continue
                    identity = self.identities[pid]
                    vec = embeddings.get(i)
                    sim = identity.match_score(vec) if vec is not None else 0.0
                    logger.debug(f"[cam {camera_id}] det#{i} MATCHED to {pid}: sim={sim:.3f} cost={cost[r, c]:.3f}")
                    if vec is not None:
                        identity.add_embedding(vec)
                    identity.last_seen_camera = camera_id
                    identity.last_seen_time = timestamp
                    identity.last_bbox = valid_detections[i]["bbox"]
                    assigned_results[i] = {"identity": identity, "match_score": sim, "is_new": False}
                    if local_ids[i] is not None:
                        local_map[local_ids[i]] = pid

            # Anything still unresolved mints the NEXT MONOTONIC RESERVED ID (101, 102, 103...)
            for i in unresolved_idx:
                if i in assigned_results:
                    continue
                vec = embeddings.get(i)
                new_pid = f"Person #{self.auto_id_counter}"
                self.auto_id_counter += 1
                logger.debug(f"[cam {camera_id}] det#{i} -> minting NEW identity {new_pid} (no acceptable match found)")
                new_identity = GlobalPersonIdentity(person_id=new_pid, is_target=False)
                if vec is not None:
                    new_identity.add_embedding(vec)
                new_identity.last_seen_camera = camera_id
                new_identity.last_seen_time = timestamp
                new_identity.last_bbox = tuple(valid_detections[i]["bbox"])
                self.identities[new_pid] = new_identity
                assigned_results[i] = {"identity": new_identity, "match_score": 1.0, "is_new": True}
                if local_ids[i] is not None:
                    local_map[local_ids[i]] = new_pid

            output_detections = []
            for i, det in enumerate(valid_detections):
                res = assigned_results.get(i)
                if res is None:
                    continue
                det_copy = dict(det)
                det_copy["person_id"] = res["identity"].person_id
                det_copy["display_name"] = res["identity"].display_name
                det_copy["is_target"] = res["identity"].is_target
                det_copy["color"] = res["identity"].color
                det_copy["match_score"] = res["match_score"]
                det_copy["is_new"] = res["is_new"]
                output_detections.append(det_copy)

            return output_detections