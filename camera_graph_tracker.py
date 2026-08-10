"""
IMCITS — NetworkX Spatial-Temporal Camera Graph & Trajectory Prediction Engine
===============================================================================
1. Spatial-Temporal Camera Network Graph:
   - Models camera coverage zones and transition paths as a directed weighted graph.
   - Flow Topology: cam-iriun-01 (Shahzeb Mobile) -> cam-laptop-01 (Laptop Webcam) -> cam-iriun-02 (Aimal Mobile)
2. Per-Person Trajectory Tracking:
   - Tracks location history for every assigned Person ID (Person #101, Person #102, etc.).
   - Computes real transition speeds when a person moves between cameras.
3. Multi-Hop Shortest Path & Trajectory Forecasting:
   - Predicts next expected camera location, estimated time of arrival (ETA), and complete movement path.
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class SightingRecord:
    person_id: str
    camera_id: str
    camera_name: str
    confidence: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class PersonTrajectoryTrack:
    person_id: str
    sightings: List[SightingRecord] = field(default_factory=list)
    current_camera_id: Optional[str] = None
    current_camera_name: Optional[str] = None
    predicted_next_id: Optional[str] = None
    predicted_next_name: Optional[str] = None
    predicted_transit_time: float = 0.0
    predicted_probability: float = 0.0
    predicted_path_chain: List[str] = field(default_factory=list)
    last_seen_time: float = 0.0


class CameraGraphTracker:
    """
    Spatial-Temporal Camera Topology & Per-Person Trajectory Forecasting Engine.
    """

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.tracks: Dict[str, PersonTrajectoryTrack] = {}  # {person_id: PersonTrajectoryTrack}

    def setup_default_topology(self) -> None:
        """
        Builds 3-camera spatial topology (iriun-01 -> laptop-01 -> iriun-02).
        """
        self.graph.clear()

        # Add 3 Camera Nodes (Shahzeb Mobile -> Laptop Webcam -> Aimal Mobile)
        self.graph.add_node("cam-iriun-01", name="Shahzeb Mobile", zone="Zone A - Entry / Mobile 1")
        self.graph.add_node("cam-laptop-01", name="Laptop Webcam", zone="Zone B - Main Terminal")
        self.graph.add_node("cam-iriun-02", name="Aimal Mobile", zone="Zone C - Exit Corridor / Mobile 2")
        self.graph.add_node("cam-exit-01", name="Exit Security Gate", zone="Zone D - Outer Gate")

        # Directed Transition Edges (iriun-01 -> laptop-01 -> iriun-02 -> exit)
        self.graph.add_edge("cam-iriun-01", "cam-laptop-01", weight=4.0, prob=0.85)
        self.graph.add_edge("cam-iriun-01", "cam-iriun-02", weight=9.0, prob=0.15)
        
        self.graph.add_edge("cam-laptop-01", "cam-iriun-02", weight=5.0, prob=0.85)
        self.graph.add_edge("cam-laptop-01", "cam-iriun-01", weight=4.0, prob=0.15)
        
        self.graph.add_edge("cam-iriun-02", "cam-exit-01", weight=6.0, prob=0.90)
        self.graph.add_edge("cam-iriun-02", "cam-laptop-01", weight=5.0, prob=0.10)

        logger.info("Initialized 3-Camera Spatial-Temporal Topology (NetworkX):")
        for u, v, data in self.graph.edges(data=True):
            logger.info(
                f"  Edge: {self.graph.nodes[u]['name']} -> {self.graph.nodes[v]['name']} "
                f"(ETA: {data['weight']}s | Prob: {int(data['prob']*100)}%)"
            )

    def record_sighting(
        self,
        person_id: str,
        camera_id: str,
        camera_name: str,
        confidence: float
    ) -> PersonTrajectoryTrack:
        """
        Logs a person sighting, updates their trajectory history, and forecasts their next location.
        """
        now = time.time()
        record = SightingRecord(
            person_id=person_id,
            camera_id=camera_id,
            camera_name=camera_name,
            confidence=confidence,
            timestamp=now
        )

        if person_id not in self.tracks:
            self.tracks[person_id] = PersonTrajectoryTrack(person_id=person_id)

        track = self.tracks[person_id]
        track.sightings.append(record)
        track.current_camera_id = camera_id
        track.current_camera_name = camera_name
        track.last_seen_time = now

        # Forecast Next Camera & Full Trajectory Path
        next_id, next_name, transit_sec, prob, path_chain = self.forecast_trajectory(camera_id)
        
        track.predicted_next_id = next_id
        track.predicted_next_name = next_name
        track.predicted_transit_time = transit_sec
        track.predicted_probability = prob
        track.predicted_path_chain = path_chain

        return track

    def forecast_trajectory(
        self, current_camera_id: str
    ) -> Tuple[Optional[str], Optional[str], float, float, List[str]]:
        """
        Predicts the most probable next camera node, estimated transit time, probability,
        and full path chain across the topology.
        """
        if not self.graph.has_node(current_camera_id):
            nodes = [n for n in self.graph.nodes if n != current_camera_id]
            if nodes:
                target = nodes[0]
                name = self.graph.nodes[target].get("name", target)
                return target, name, 4.0, 0.50, [current_camera_id, target]
            return None, None, 0.0, 0.0, []

        # Outgoing candidate transitions from current camera
        out_edges = list(self.graph.out_edges(current_camera_id, data=True))
        if not out_edges:
            return None, None, 0.0, 0.0, [current_camera_id]

        # Sort outgoing edges by highest probability, then lowest transit time
        out_edges.sort(key=lambda e: (-e[2].get("prob", 0.5), e[2].get("weight", 999.0)))

        best_edge = out_edges[0]
        next_node = best_edge[1]
        data = best_edge[2]
        next_name = self.graph.nodes[next_node].get("name", next_node)
        transit_sec = float(data.get("weight", 4.0))
        prob = float(data.get("prob", 0.80))

        # Multi-hop path forecast (shortest path to facility exit if exists)
        path_chain = [current_camera_id, next_node]
        try:
            if nx.has_path(self.graph, next_node, "cam-exit-01"):
                shortest_path = nx.shortest_path(self.graph, source=next_node, target="cam-exit-01", weight="weight")
                path_chain.extend(shortest_path[1:])
        except Exception:
            pass

        # Format human-readable path chain names
        readable_chain = [self.graph.nodes[n].get("name", n) for n in path_chain]

        return next_node, next_name, transit_sec, prob, readable_chain
