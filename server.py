"""
IMCITS — Web Dashboard Bridge Server
=====================================
Runs the existing 3-camera YOLO + OSNet tracking engine
(poi_detector_matcher / live_camera_streamer / camera_graph_tracker)
in a background thread, headlessly (no cv2.imshow windows), and streams
results to the SENTRY web dashboard:

  - Socket.IO 'detection' events   -> real-time bounding boxes + logs
  - GET  /api/cameras              -> camera list + online status
  - POST /api/cameras/discover     -> probe sources & start capture workers
  - DELETE /api/cameras/<cam_id>   -> stop and remove a camera
  - GET  /api/stream/<cam_id>      -> live MJPEG feed for a camera tile
  - GET  /api/targets              -> list enrolled Persons of Interest
  - POST /api/targets              -> register a reference photo (POI)
  - DELETE /api/targets/<id>       -> un-enroll a POI and delete its reference photos
  - POST /api/tracking/start|stop  -> toggle the detection loop
  - GET  /api/topology              -> user-defined camera graph (nodes+edges)
  - POST /api/topology/nodes        -> add a non-camera node (e.g. 'Exit Gate')
  - DELETE /api/topology/nodes/<id> -> remove a node
  - POST /api/topology/edges        -> connect two nodes (eta seconds, probability %)
  - DELETE /api/topology/edges      -> disconnect two nodes

Only sightings of an ENROLLED target (is_target=True) are pushed to the
dashboard — the engine still assigns Person #101, #102... to bystanders
internally (needed for correct Re-ID / mutual exclusivity), but the
dashboard's job is tracking a specific person of interest, not logging
everyone in frame.

Install:
  pip install flask flask-socketio eventlet opencv-python ultralytics torch torchreid supervision scipy

Run (from project root, so `targets/` resolves correctly):
  python src/server.py
Then open http://localhost:5000
"""
from __future__ import annotations

import os
import sys
import time
import json
import shutil
import threading
import logging
from datetime import datetime
from typing import Dict, List, Optional

import cv2
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_socketio import SocketIO

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from live_camera_streamer import MultiCameraManager
from poi_detector_matcher import POIDetectorMatcher
from register_poi import register_target
# NOTE: camera_graph_tracker.CameraGraphTracker (fixed, hardcoded topology)
# is no longer used for predictions — the camera-to-camera graph is now
# defined by the user at runtime in the dashboard and stored in
# topology.json (see TopologyStore below).

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("IMCITS-Server")

# ─── Same 3-camera candidate config as imcits_runner.py ──────────────────────
CAMERA_CANDIDATES = [
    {"id": "cam-iriun-01", "name": "Shahzeb Mobile", "sources": [3, 5]},
    {"id": "cam-laptop-01", "name": "Laptop Webcam", "sources": [0]},
    {"id": "cam-iriun-02", "name": "Aimal Mobile", "sources": [1, 2, 4, 6]},
]
_CAMERA_WARMUP_SECONDS = 3.0
_ALERT_COOLDOWN_SEC = 2.5          # min gap between repeated events for the same person+camera
_STREAM_JPEG_QUALITY = 70
_STREAM_FRAME_INTERVAL = 0.08      # ~12 fps per browser tile, plenty for a dashboard

def _find_dir(dirname: str) -> str:
    """Looks for `dirname` next to this script first (flat layout: server.py
    at project root), then one level up (split layout: server.py in src/,
    dirname in the project root). Falls back to the flat-layout path (and
    lets the caller error loudly) if neither exists yet."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, dirname),
        os.path.join(here, "..", dirname),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)
    return os.path.abspath(candidates[0])


STATIC_DIR = _find_dir("web")
TARGETS_DIR = _find_dir("targets")
logger.info(f"Serving dashboard from: {STATIC_DIR}")
logger.info(f"Reading/writing targets in: {TARGETS_DIR}")
if not os.path.isfile(os.path.join(STATIC_DIR, "sentry-dashboard.html")):
    logger.warning(
        f"sentry-dashboard.html not found in {STATIC_DIR} — "
        f"place it in a 'web' folder either next to server.py or in the project root."
    )

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
app.config["SECRET_KEY"] = "imcits-dev"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ─── Engine components (same classes as imcits_runner.py) ────────────────────
# match_threshold: ArcFace face embeddings are purpose-trained for face
# verification, so genuine same-person cosine similarity is meaningfully
# higher than the old ImageNet-only body embedder ever produced — 0.45 is
# a solid starting point. Raise it (e.g. 0.55) if you start seeing
# different people matched to the same target; lower it if a real match
# from an off-angle/distant face isn't being picked up.
_REID_MATCH_THRESHOLD = 0.45

detector = POIDetectorMatcher(
    yolo_model_name="yolov8n.pt",
    targets_dir=TARGETS_DIR,
    match_threshold=_REID_MATCH_THRESHOLD,
    face_model_name="buffalo_l",  # auto-downloaded by insightface on first run
)
detector.load_targets()

camera_manager = MultiCameraManager()

engine_state = {
    "cameras": [],      # [{id, name, source, online}]
    "tracking": False,  # detection loop on/off
    "started": False,   # discover_cameras() already ran
}
_last_alert_time: Dict[str, float] = {}
_engine_lock = threading.Lock()

# ─── "Deleted" camera ids (current process only) ──────────────────────────────
# In-memory only, on purpose: a deletion here means "stop showing/using this
# camera for this run of the server" — restart the process and discovery
# starts fresh from CAMERA_CANDIDATES again.
removed_camera_ids: set = set()


# ─── User-defined camera topology (nodes + directed edges) ───────────────────
# Replaces the old hardcoded CameraGraphTracker.setup_default_topology().
# Cameras are auto-added as nodes on discovery; the user draws the edges
# (and can add non-camera nodes, e.g. an "Exit Gate") from the dashboard.
_TOPOLOGY_PATH = os.path.join(os.path.dirname(TARGETS_DIR), "topology.json")


class TopologyStore:
    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()
        self.nodes: Dict[str, dict] = {}   # id -> {id, name, is_camera}
        self.edges: List[dict] = []        # [{from, to, eta, probability}]
        self._load()

    def _load(self):
        if os.path.isfile(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.nodes = {n["id"]: n for n in data.get("nodes", [])}
                self.edges = data.get("edges", [])
            except Exception as e:
                logger.warning(f"Could not load {self._path}: {e}")

    def _save(self):
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump({"nodes": list(self.nodes.values()), "edges": self.edges}, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save {self._path}: {e}")

    def sync_cameras(self, cameras: List[dict]):
        """Adds/updates a node for every discovered camera without touching
        user-added virtual nodes or existing edges."""
        with self._lock:
            for cam in cameras:
                self.nodes[cam["id"]] = {"id": cam["id"], "name": cam["name"], "is_camera": True}
            self._save()

    def add_node(self, node_id: str, name: str, is_camera: bool = False) -> dict:
        with self._lock:
            node = {"id": node_id, "name": name, "is_camera": is_camera}
            self.nodes[node_id] = node
            self._save()
            return node

    def remove_node(self, node_id: str):
        with self._lock:
            self.nodes.pop(node_id, None)
            self.edges = [e for e in self.edges if e["from"] != node_id and e["to"] != node_id]
            self._save()

    def upsert_edge(self, from_id: str, to_id: str, eta: float, probability: float) -> dict:
        with self._lock:
            edge = {"from": from_id, "to": to_id, "eta": eta, "probability": probability}
            self.edges = [e for e in self.edges if not (e["from"] == from_id and e["to"] == to_id)]
            self.edges.append(edge)
            self._save()
            return edge

    def remove_edge(self, from_id: str, to_id: str):
        with self._lock:
            self.edges = [e for e in self.edges if not (e["from"] == from_id and e["to"] == to_id)]
            self._save()

    def predict_next(self, camera_id: str) -> Optional[dict]:
        """Highest-probability outgoing edge from camera_id, or None if the
        user hasn't connected this camera to anything yet."""
        candidates = [e for e in self.edges if e["from"] == camera_id]
        if not candidates:
            return None
        best = max(candidates, key=lambda e: e["probability"])
        to_node = self.nodes.get(best["to"])
        return {
            "name": to_node["name"] if to_node else best["to"],
            "eta": best["eta"],
            "probability": best["probability"],
        }

    def as_dict(self) -> dict:
        return {"nodes": list(self.nodes.values()), "edges": self.edges}


topology = TopologyStore(_TOPOLOGY_PATH)


# ─── Camera discovery (ported from imcits_runner.probe_camera / discover_cameras) ─
def probe_camera(source, warmup_reads: int = 4) -> bool:
    try:
        if isinstance(source, int) or (isinstance(source, str) and str(source).isdigit()):
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
            if ret and frame is not None and frame.size > 0 and float(frame.mean()) > 5.0:
                got_frame = True
                break
            time.sleep(0.04)
        cap.release()
        return got_frame
    except Exception:
        return False


def discover_cameras() -> List[dict]:
    used_sources: set = set()
    found = []
    logger.info("Discovering live camera streams...")
    for cam in CAMERA_CANDIDATES:
        if cam["id"] in removed_camera_ids:
            logger.info(f"  [{cam['name']}] -> skipped (previously removed from dashboard)")
            continue
        assigned_source = None
        for src in cam["sources"]:
            if src in used_sources:
                continue
            if probe_camera(src):
                assigned_source = src
                used_sources.add(src)
                break
        found.append({
            "id": cam["id"],
            "name": cam["name"],
            "source": assigned_source,
            "online": assigned_source is not None,
        })
        logger.info(f"  [{cam['name']}] -> {'LIVE on src ' + str(assigned_source) if assigned_source is not None else 'OFFLINE'}")
    return found


# ─── Background detection loop (headless version of imcits_runner.main()'s inner loop) ─
def detection_loop():
    while True:
        if not engine_state["tracking"]:
            time.sleep(0.2)
            continue

        latest_frames = camera_manager.get_latest_frames()
        now = time.time()

        for cam_id, (cam_name, frame, fps, is_connected) in latest_frames.items():
            if frame is None or not is_connected:
                continue

            h, w = frame.shape[:2]
            # NOTE: assign_identities_for_frame() hard-drops any detection
            # with conf < 0.55 (see poi_detector_matcher.py). Detecting at a
            # lower threshold here just means YOLO boxes get silently
            # discarded downstream, so match it exactly.
            raw_detections = detector.detect_persons(frame, conf_threshold=0.55)
            assigned = detector.assign_identities_for_frame(
                raw_detections, camera_id=cam_id, timestamp=now, frame_width=w, frame_height=h
            )

            for det in assigned:
                if not det.get("is_target"):
                    continue  # only enrolled Persons of Interest are surfaced to the dashboard

                key = f"{det['person_id']}_{cam_id}"
                if now - _last_alert_time.get(key, 0) < _ALERT_COOLDOWN_SEC:
                    continue
                _last_alert_time[key] = now

                x1, y1, x2, y2 = det["bbox"]
                prediction = topology.predict_next(cam_id)

                payload = {
                    "person_id": det["person_id"],
                    "display_name": det["display_name"],
                    "camera_id": cam_id,
                    "camera_name": cam_name,
                    "confidence": round(det["match_score"] * 100, 1),
                    "bbox_pct": {
                        "x": round(x1 / w * 100, 2),
                        "y": round(y1 / h * 100, 2),
                        "w": round((x2 - x1) / w * 100, 2),
                        "h": round((y2 - y1) / h * 100, 2),
                    },
                    "predicted_next": prediction["name"] if prediction else None,
                    "predicted_eta_sec": prediction["eta"] if prediction else None,
                    "predicted_probability": prediction["probability"] if prediction else None,
                    "timestamp": datetime.fromtimestamp(now).strftime("%H:%M:%S"),
                }
                socketio.emit("detection", payload)
                logger.info(f"[{cam_name}] {det['display_name']} matched ({payload['confidence']}%)")

        time.sleep(0.05)


# ─────────────────────────── REST API ───────────────────────────
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "sentry-dashboard.html")


@app.route("/api/cameras", methods=["GET"])
def api_cameras():
    return jsonify(engine_state["cameras"])


@app.route("/api/cameras/discover", methods=["POST"])
def api_discover():
    with _engine_lock:
        if engine_state["started"]:
            return jsonify(engine_state["cameras"])

        found = discover_cameras()
        for cam in found:
            if cam["online"]:
                camera_manager.add_camera(cam["id"], cam["name"], cam["source"])

        camera_manager.start_all()
        time.sleep(_CAMERA_WARMUP_SECONDS)

        engine_state["cameras"] = found
        engine_state["started"] = True
        topology.sync_cameras(found)
    return jsonify(found)


@app.route("/api/cameras/<cam_id>", methods=["DELETE"])
def api_remove_camera(cam_id):
    with _engine_lock:
        if not any(c["id"] == cam_id for c in engine_state["cameras"]):
            return jsonify({"error": "unknown camera id"}), 404

        engine_state["cameras"] = [c for c in engine_state["cameras"] if c["id"] != cam_id]

        # MultiCameraManager doesn't expose a remove method, but each worker
        # is a real CameraStreamWorker with .stop() — use that directly and
        # drop it from the manager's dict so get_latest_frames()/streaming
        # stop referencing it.
        worker = camera_manager.workers.pop(cam_id, None)
        if worker is not None:
            try:
                worker.stop()
            except Exception as e:
                logger.warning(f"Error stopping camera worker for {cam_id}: {e}")

        topology.remove_node(cam_id)

        removed_camera_ids.add(cam_id)  # excluded from discover_cameras() for the rest of this run
    return jsonify({"status": "removed", "id": cam_id})


# ─── Topology API (user-defined camera graph) ─────────────────────────────
@app.route("/api/topology", methods=["GET"])
def api_topology_get():
    return jsonify(topology.as_dict())


@app.route("/api/topology/nodes", methods=["POST"])
def api_topology_add_node():
    """JSON body: {name}. Adds a non-camera node (e.g. 'Exit Gate')."""
    name = (request.json or {}).get("name", "").strip()
    if not name:
        return jsonify({"error": "name is required"}), 400
    node_id = "node-" + "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-")
    if node_id in topology.nodes:
        return jsonify({"error": "a node with that name already exists"}), 400
    node = topology.add_node(node_id, name, is_camera=False)
    return jsonify(node)


@app.route("/api/topology/nodes/<node_id>", methods=["DELETE"])
def api_topology_remove_node(node_id):
    topology.remove_node(node_id)
    return jsonify({"status": "removed"})


@app.route("/api/topology/edges", methods=["POST"])
def api_topology_upsert_edge():
    """JSON body: {from, to, eta, probability}."""
    body = request.json or {}
    from_id, to_id = body.get("from"), body.get("to")
    if not from_id or not to_id or from_id == to_id:
        return jsonify({"error": "from and to are required and must differ"}), 400
    if from_id not in topology.nodes or to_id not in topology.nodes:
        return jsonify({"error": "unknown node id"}), 400
    try:
        eta = float(body.get("eta", 0))
        probability = max(0.0, min(100.0, float(body.get("probability", 0))))
    except (TypeError, ValueError):
        return jsonify({"error": "eta and probability must be numbers"}), 400
    edge = topology.upsert_edge(from_id, to_id, eta, probability)
    return jsonify(edge)


@app.route("/api/topology/edges", methods=["DELETE"])
def api_topology_remove_edge():
    body = request.json or {}
    from_id, to_id = body.get("from"), body.get("to")
    if not from_id or not to_id:
        return jsonify({"error": "from and to are required"}), 400
    topology.remove_edge(from_id, to_id)
    return jsonify({"status": "removed"})


@app.route("/api/stream/<cam_id>")
def api_stream(cam_id):
    """MJPEG feed for one camera tile."""
    def gen():
        while True:
            frames = camera_manager.get_latest_frames()
            data = frames.get(cam_id)
            if data:
                _, frame, _, connected = data
                if connected and frame is not None:
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _STREAM_JPEG_QUALITY])
                    if ok:
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(_STREAM_FRAME_INTERVAL)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/targets", methods=["GET"])
def api_list_targets():
    with _engine_lock:
        targets = [
            {"id": pid, "name": identity.display_name}
            for pid, identity in detector.identities.items() if identity.is_target
        ]
    return jsonify(targets)


@app.route("/api/targets", methods=["POST"])
def api_register_target():
    """multipart/form-data: name=<str>, image=<file>.

    Runs YOLO on the upload and enrolls a crop of the highest-confidence
    detected person — NOT the raw uploaded image (embedding a whole scene
    instead of a person crop is what let a photo of a piece of paper match
    100% earlier). Then verifies InsightFace can actually find a face in
    that crop, since the ArcFace embedder needs one — catching this at
    upload time instead of silently failing to enroll on the next
    load_targets() call.
    """
    name = request.form.get("name", "").strip()
    file = request.files.get("image")
    if not name or not file:
        return jsonify({"error": "name and image are required"}), 400

    os.makedirs(TARGETS_DIR, exist_ok=True)
    tmp_path = os.path.join(TARGETS_DIR, f"_upload_{int(time.time())}.jpg")
    file.save(tmp_path)

    img = cv2.imread(tmp_path)
    if img is None:
        os.remove(tmp_path)
        return jsonify({"error": "could not read the uploaded image"}), 400

    with _engine_lock:
        detections = detector.detect_persons(img, conf_threshold=0.5)

    if not detections:
        os.remove(tmp_path)
        return jsonify({
            "error": "no person detected in this photo — upload a clear photo where the "
                     "person is visible (not a document, screenshot, or empty scene)"
        }), 400

    best = max(detections, key=lambda d: d["conf"])
    x1, y1, x2, y2 = best["bbox"]
    crop = img[y1:y2, x1:x2]

    with _engine_lock:
        face_vec = detector.extract_invariant_feature_vector(crop)
    if face_vec is None:
        os.remove(tmp_path)
        return jsonify({
            "error": "a person was detected but no clear face was found — use a photo "
                     "where the face is visible and reasonably well-lit"
        }), 400

    crop_path = tmp_path.replace(".jpg", "_crop.jpg")
    cv2.imwrite(crop_path, crop)
    os.remove(tmp_path)

    ok = register_target(name, crop_path, targets_dir=TARGETS_DIR)
    if os.path.exists(crop_path):
        os.remove(crop_path)
    if not ok:
        return jsonify({"error": "failed to register target"}), 500

    with _engine_lock:
        detector.load_targets()  # reload embedding gallery so the new POI is matchable immediately

    return jsonify({"status": "registered", "name": name})


@app.route("/api/targets/<path:target_id>", methods=["DELETE"])
def api_remove_target(target_id):
    """Un-enrolls a Person of Interest: drops it from the live matching
    gallery immediately AND deletes its reference photos on disk, so a
    later restart / load_targets() call doesn't silently re-enroll it.
    This is what fixes 'I registered a new photo but it still matches the
    old person' — the old identity's embeddings stay in the gallery
    (and can out-score a freshly registered, thinly-sampled new one)
    until it's explicitly removed."""
    with _engine_lock:
        identity = detector.identities.pop(target_id, None)
        if identity is None:
            return jsonify({"error": "target not found"}), 404

        folder_name = identity.display_name.strip().replace(" ", "_")
        folder_path = os.path.join(TARGETS_DIR, folder_name)
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path, ignore_errors=True)

    return jsonify({"status": "removed", "name": identity.display_name})


@app.route("/api/tracking/start", methods=["POST"])
def api_start_tracking():
    if not engine_state["started"]:
        return jsonify({"error": "call /api/cameras/discover first"}), 400
    if not any(identity.is_target for identity in detector.identities.values()):
        return jsonify({"error": "no target registered yet — upload a reference photo first"}), 400
    engine_state["tracking"] = True
    return jsonify({"status": "tracking"})


@app.route("/api/tracking/stop", methods=["POST"])
def api_stop_tracking():
    engine_state["tracking"] = False
    return jsonify({"status": "idle"})


if __name__ == "__main__":
    threading.Thread(target=detection_loop, daemon=True).start()
    logger.info("SENTRY bridge server starting on http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000)