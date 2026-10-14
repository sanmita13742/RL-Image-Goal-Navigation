"""
realworld/src/subgoal_planner.py — Topological Visual Graph Subgoal Planner.
=============================================================================
Constructs a k-NN visual roadmap over DINOv3 feature embeddings collected during
exploration. Decomposes long-horizon visual navigation tasks into a chain of
achievable local subgoals:

  Current Frame -> [Local Waypoint 1] -> [Local Waypoint 2] -> ... -> Final Goal

Algorithm:
  1. Offline Graph Construction:
     - Nodes: Exploration frames with valid embeddings [N, 384].
     - Temporal Edges: Adjacent frames within the same segment (t -> t+1).
     - Visual Cross-Edges: Pairs (i, j) with high cosine similarity (sim >= min_sim)
       optionally gated by odometric proximity (distance <= max_dist_m).
     - Edge Weights: Cosine distance (1.0 - cos_sim).
  2. Online Dijkstra Planning:
     - Project current robot observation onto nearest graph node s.
     - Project goal image onto nearest graph node g.
     - Compute shortest path via Dijkstra (heapq).
  3. Dynamic Subgoal Tracking:
     - Hands current waypoint embedding to the actor policy.
     - Automatically switches to next waypoint when sim(phi_robot, phi_waypoint) >= threshold.
"""

import heapq
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger("subgoal_planner")


class VisualGraphPlanner:
    def __init__(
        self,
        phi_vectors: np.ndarray,
        frame_index: Optional[pd.DataFrame] = None,
        k_neighbors: int = 8,
        min_connection_sim: float = 0.75,
        max_connection_dist_m: float = 2.5,
        waypoint_advance_sim: float = 0.75,
        subsample_stride: int = 3,
    ):
        """
        Initialize the visual graph from exploration embeddings.

        Args:
            phi_vectors: [N, 384] L2-normalized DINOv3 embeddings
            frame_index: DataFrame containing metadata ('segment_id', 'pos_x', 'pos_y')
            k_neighbors: Max cross-edges per node in k-NN
            min_connection_sim: Minimum cosine similarity to form a topological edge
            max_connection_dist_m: Max physical distance (if odom available) for edges
            waypoint_advance_sim: Cosine similarity to current waypoint before advancing
            subsample_stride: Stride to subsample dense frames (e.g. 10 Hz -> 3.3 Hz nodes)
        """
        self.waypoint_advance_sim = waypoint_advance_sim
        self.subsample_stride = max(1, subsample_stride)

        # 1. Subsample nodes to keep graph compact and fast
        node_indices = np.arange(0, len(phi_vectors), self.subsample_stride)
        self.node_to_global = node_indices
        self.global_to_node = {g_idx: n_idx for n_idx, g_idx in enumerate(node_indices)}

        self.phi_nodes = phi_vectors[node_indices]  # [M, 384]
        self.num_nodes = len(self.phi_nodes)

        logger.info(
            f"Building Visual Graph: {self.num_nodes:,} nodes (from {len(phi_vectors):,} frames, stride={self.subsample_stride})"
        )

        has_odom = False
        coords = None
        segment_ids = None

        if frame_index is not None:
            sub_df = frame_index.iloc[node_indices].reset_index(drop=True)
            if "pos_x" in sub_df.columns and "pos_y" in sub_df.columns:
                coords = sub_df[["pos_x", "pos_y"]].values
                # Check if coordinates are non-trivial
                if np.ptp(coords[:, 0]) > 0.1 or np.ptp(coords[:, 1]) > 0.1:
                    has_odom = True
            if "segment_id" in sub_df.columns:
                segment_ids = sub_df["segment_id"].values

        # 2. Build adjacency list: {node_idx: [(neighbor_idx, weight), ...]}
        self.adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(self.num_nodes)}

        # Add temporal edges
        for i in range(self.num_nodes - 1):
            same_seg = True
            if segment_ids is not None:
                same_seg = (segment_ids[i] == segment_ids[i + 1])

            if same_seg:
                # Cosine distance
                cos_sim = float(np.dot(self.phi_nodes[i], self.phi_nodes[i + 1]))
                weight = max(0.01, 1.0 - cos_sim)
                self.adj[i].append((i + 1, weight))
                self.adj[i + 1].append((i, weight))

        # Add k-NN topological cross-edges
        # Compute in chunks to handle memory safely
        chunk_size = 500
        for start_idx in range(0, self.num_nodes, chunk_size):
            end_idx = min(self.num_nodes, start_idx + chunk_size)
            chunk_sims = np.dot(self.phi_nodes[start_idx:end_idx], self.phi_nodes.T)  # [chunk, M]

            for local_i, global_i in enumerate(range(start_idx, end_idx)):
                sims = chunk_sims[local_i]
                sims[global_i] = -1.0  # exclude self

                # Mask by physical distance if odom is available
                if has_odom and coords is not None:
                    dists = np.hypot(coords[:, 0] - coords[global_i, 0], coords[:, 1] - coords[global_i, 1])
                    sims[dists > max_connection_dist_m] = -1.0

                top_k_indices = np.argpartition(sims, -k_neighbors)[-k_neighbors:]
                for j in top_k_indices:
                    sim_val = sims[j]
                    if sim_val >= min_connection_sim:
                        weight = max(0.01, 1.0 - sim_val)
                        self.adj[global_i].append((int(j), weight))
                        self.adj[int(j)].append((global_i, weight))

        # Count total edges
        total_edges = sum(len(neighbors) for neighbors in self.adj.values()) // 2
        logger.info(f"Visual Graph constructed: {self.num_nodes} nodes, {total_edges} bidirectional edges.")

        # Active path state
        self.current_path: List[int] = []
        self.current_waypoint_idx: int = 0
        self.final_goal_phi: Optional[np.ndarray] = None

    def find_nearest_node(self, query_phi: np.ndarray) -> Tuple[int, float]:
        """Find graph node with highest cosine similarity to query embedding."""
        if query_phi.ndim == 1:
            sims = np.dot(self.phi_nodes, query_phi)
        else:
            sims = np.dot(self.phi_nodes, query_phi.squeeze())
        best_node = int(np.argmax(sims))
        return best_node, float(sims[best_node])

    def dijkstra(self, start_node: int, goal_node: int) -> Optional[List[int]]:
        """Compute shortest path between start_node and goal_node using Dijkstra."""
        if start_node == goal_node:
            return [start_node]

        dist = {i: float('inf') for i in range(self.num_nodes)}
        prev: Dict[int, Optional[int]] = {i: None for i in range(self.num_nodes)}
        dist[start_node] = 0.0

        pq = [(0.0, start_node)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == goal_node:
                break

            for v, weight in self.adj[u]:
                alt = d + weight
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(pq, (alt, v))

        if dist[goal_node] == float('inf'):
            logger.warning(f"No path found in visual graph between node {start_node} and {goal_node}.")
            return None

        # Reconstruct path
        path = []
        curr = goal_node
        while curr is not None:
            path.append(curr)
            curr = prev[curr]
        path.reverse()
        return path

    def plan(self, current_phi: np.ndarray, goal_phi: np.ndarray) -> bool:
        """
        Plan a topological waypoint route from current robot observation to goal image.
        """
        self.final_goal_phi = goal_phi
        start_node, start_sim = self.find_nearest_node(current_phi)
        goal_node, goal_sim = self.find_nearest_node(goal_phi)

        logger.info(f"Visual Planner: Start node={start_node} (sim={start_sim:.3f}), Goal node={goal_node} (sim={goal_sim:.3f})")

        # Direct navigation check: if start and goal nodes are identical or sim is already high
        if start_node == goal_node:
            self.current_path = [goal_node]
            self.current_waypoint_idx = 0
            return True

        path = self.dijkstra(start_node, goal_node)
        if path is None:
            logger.warning("Falling back to direct goal pursuit (graph disconnected).")
            self.current_path = [goal_node]
            self.current_waypoint_idx = 0
            return False

        self.current_path = path
        self.current_waypoint_idx = 0
        logger.info(f"Visual Planner: Found route with {len(path)} waypoints across the environment.")
        return True

    def get_current_subgoal(self, current_phi: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Returns (subgoal_phi, is_final_goal).
        Automatically advances waypoints as the robot arrives near each node.
        """
        if not self.current_path:
            return self.final_goal_phi, True

        while self.current_waypoint_idx < len(self.current_path) - 1:
            waypoint_node = self.current_path[self.current_waypoint_idx]
            waypoint_phi = self.phi_nodes[waypoint_node]
            sim = float(np.dot(current_phi, waypoint_phi))

            if sim >= self.waypoint_advance_sim:
                self.current_waypoint_idx += 1
                new_node = self.current_path[self.current_waypoint_idx]
                logger.info(
                    f"Reached Waypoint {self.current_waypoint_idx}/{len(self.current_path)} "
                    f"(sim={sim:.3f} >= {self.waypoint_advance_sim:.2f}). "
                    f"Switching to node {new_node}."
                )
            else:
                break

        # Check if at the last waypoint
        is_last_waypoint = (self.current_waypoint_idx >= len(self.current_path) - 1)
        if is_last_waypoint and self.final_goal_phi is not None:
            return self.final_goal_phi, True

        current_node = self.current_path[self.current_waypoint_idx]
        return self.phi_nodes[current_node], False


def load_planner_from_run(run_dir: Path) -> VisualGraphPlanner:
    """Convenience helper to initialize VisualGraphPlanner from processed dataset folder."""
    run_dir = Path(run_dir).resolve()
    processed_dir = run_dir / "processed" if (run_dir / "processed").exists() else run_dir

    phi_path = processed_dir / "dinov3" / "phi_vectors.npy"
    if not phi_path.exists():
        phi_path = processed_dir / "dinov3" / "phi_cache.npy"

    if not phi_path.exists():
        raise FileNotFoundError(f"phi_vectors.npy not found in {processed_dir / 'dinov3'}")

    phi = np.load(phi_path)

    frame_index_path = processed_dir / "frame_index" / "frame_index.parquet"
    frame_index = None
    if frame_index_path.exists():
        frame_index = pd.read_parquet(frame_index_path)

    return VisualGraphPlanner(phi_vectors=phi, frame_index=frame_index)
