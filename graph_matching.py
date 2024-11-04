import networkx as nx
import numpy as np
from typing import Dict, List, Tuple
import cv2
from scipy.optimize import linear_sum_assignment


class FieldGraphMatcher:
    def __init__(self, reference_graph: nx.Graph, detected_graph: nx.Graph):
        self.reference = reference_graph
        self.detected = detected_graph

    def match_graphs(self) -> Dict[int, str]:
        """
        Match detected intersection nodes to reference field points.
        Returns a dictionary mapping detected node indices to reference node labels.
        """
        # Create similarity matrix between all pairs of nodes
        similarity_matrix = self._create_similarity_matrix()

        # Use Hungarian algorithm to find optimal matching
        detected_indices, reference_indices = linear_sum_assignment(-similarity_matrix)

        # Create mapping with confidence scores
        matches = {}
        reference_labels = list(self.reference.nodes())

        for det_idx, ref_idx in zip(detected_indices, reference_indices):
            similarity_score = similarity_matrix[det_idx, ref_idx]
            if similarity_score > 0.5:  # Minimum confidence threshold
                matches[det_idx] = reference_labels[ref_idx]

        return matches

    def _create_similarity_matrix(self) -> np.ndarray:
        """
        Create similarity matrix between all detected and reference nodes.
        Higher values indicate better matches.
        """
        det_nodes = list(self.detected.nodes())
        ref_nodes = list(self.reference.nodes())

        similarity_matrix = np.zeros((len(det_nodes), len(ref_nodes)))

        for i, det_node in enumerate(det_nodes):
            for j, ref_node in enumerate(ref_nodes):
                similarity = self._calculate_node_similarity(det_node, ref_node)
                similarity_matrix[i, j] = similarity

        return similarity_matrix

    def _calculate_node_similarity(self, det_node: int, ref_node: str) -> float:
        """
        Calculate similarity score between a detected node and a reference node
        using multiple features.
        """
        # Get feature vectors
        det_features = self.detected.nodes[det_node]
        ref_features = self.reference.nodes[ref_node]

        # Calculate different similarity components
        scores = []

        # 1. Topology similarity (degree)
        degree_sim = 1 - abs(self.detected.degree[det_node] -
                             self.reference.degree[ref_node]) / 4.0
        scores.append(degree_sim)

        # 2. Position similarity using normalized coordinates
        det_pos = np.array(det_features['norm_pos'])
        ref_pos = np.array([ref_features['x'] / 52.5, ref_features['y'] / 34.0])  # Normalize by field dimensions
        position_sim = 1 - np.linalg.norm(det_pos - ref_pos) / 2.0  # Divide by 2 as max distance in normalized space
        scores.append(position_sim)

        # 3. Corner similarity if available
        if 'is_corner' in det_features and 'is_corner' in ref_features:
            corner_sim = float(det_features['is_corner'] == ref_features['is_corner'])
            scores.append(corner_sim)

        # 4. Local structure similarity (compare neighbor patterns)
        structure_sim = self._calculate_structure_similarity(det_node, ref_node)
        scores.append(structure_sim)

        # Weighted average of similarities
        weights = [0.3, 0.3, 0.2, 0.2]  # Adjust weights based on importance
        similarity = np.average(scores, weights=weights[:len(scores)])

        return max(0, min(1, similarity))  # Ensure result is between 0 and 1

    def _calculate_structure_similarity(self, det_node: int, ref_node: str) -> float:
        """
        Calculate similarity of local graph structure around nodes
        """
        # Get immediate neighbors
        det_neighbors = set(self.detected.neighbors(det_node))
        ref_neighbors = set(self.reference.neighbors(ref_node))

        # Compare number of neighbors
        min_neighbors = min(len(det_neighbors), len(ref_neighbors))
        max_neighbors = max(len(det_neighbors), len(ref_neighbors))
        if max_neighbors == 0:
            return 1.0

        return min_neighbors / max_neighbors

    def get_world_coordinates(self, matches: Dict[int, str]) -> Dict[int, Tuple[float, float]]:
        """
        Convert matched nodes to world coordinates
        """
        world_coords = {}
        for det_idx, ref_label in matches.items():
            ref_pos = self.reference.nodes[ref_label]['pos']
            world_coords[det_idx] = ref_pos

        return world_coords


def visualize_matches(frame, detected_graph, reference_graph, matches):
    """Visualize the matched intersections"""
    result = frame.copy()

    # Draw all detected intersections in red
    for node in detected_graph.nodes():
        x, y = detected_graph.nodes[node]['pos']
        cv2.circle(result, (int(x), int(y)), 5, (0, 0, 255), -1)

    # Draw matched intersections in green with labels
    for det_idx, ref_label in matches.items():
        x, y = detected_graph.nodes[det_idx]['pos']
        cv2.circle(result, (int(x), int(y)), 8, (0, 255, 0), 2)
        cv2.putText(result, ref_label, (int(x) + 10, int(y) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return result