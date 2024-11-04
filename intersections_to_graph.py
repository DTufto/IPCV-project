import numpy as np
import networkx as nx
from typing import List, Tuple, Dict


class DetectedFieldGraph:
    def __init__(self, frame_width: int, frame_height: int):
        self.G = nx.Graph()
        self.frame_width = frame_width
        self.frame_height = frame_height

    def create_graph(self, intersections: List[Tuple[int, int]],
                     line_groups: List[List]) -> nx.Graph:
        """
        Create a graph from detected intersections and line groups

        Args:
            intersections: List of (x,y) intersection points
            line_groups: List of grouped lines from helper.lineGrouper()
        """
        # Reset graph
        self.G.clear()

        # Add nodes for each intersection
        for idx, (x, y) in enumerate(intersections):
            # Calculate normalized coordinates (-1 to 1 range)
            norm_x = (x - self.frame_width / 2) / (self.frame_width / 2)
            norm_y = (y - self.frame_height / 2) / (self.frame_height / 2)

            # Initial node attributes
            attributes = {
                'pos': (x, y),
                'norm_pos': (norm_x, norm_y),
                'x': x,
                'y': y
            }

            self.G.add_node(idx, **attributes)

        # Connect intersections that lie on the same line group
        for group in line_groups:
            # Get all intersections that lie on these parallel lines
            connected_points = self._find_connected_intersections(group, intersections)

            # Add edges between points that are connected by lines
            for i in range(len(connected_points)):
                for j in range(i + 1, len(connected_points)):
                    point1_idx = intersections.index(connected_points[i])
                    point2_idx = intersections.index(connected_points[j])

                    # Calculate Euclidean distance
                    p1 = np.array(connected_points[i])
                    p2 = np.array(connected_points[j])
                    distance = np.linalg.norm(p2 - p1)

                    # Add edge if points are reasonably close
                    max_distance = self.frame_width * 0.5  # Adjust threshold as needed
                    if distance < max_distance:
                        self.G.add_edge(point1_idx, point2_idx, weight=distance)

        # Add node features based on graph structure
        self._add_node_features()

        return self.G

    def _find_connected_intersections(self, line_group: List,
                                      intersections: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Find all intersections that lie on a group of parallel lines"""
        connected_points = []

        for point in intersections:
            # Check if point lies on any line in the group
            for line in line_group:
                if self._point_on_line(point, (-line, +line)):
                    connected_points.append(point)
                    break

        return connected_points

    def _point_on_line(self, point: Tuple[int, int],
                       line: Tuple[Tuple[int, int], Tuple[int, int]],
                       tolerance: int = 10) -> bool:
        """
        Check if a point lies on a line segment within some tolerance
        """
        x, y = point
        (x1, y1), (x2, y2) = line

        # Calculate distances
        d = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

        # Check if point is within line segment bounds
        if d > tolerance:
            return False

        # Check if point lies within the line segment's bounding box
        if min(x1, x2) - tolerance <= x <= max(x1, x2) + tolerance and \
                min(y1, y2) - tolerance <= y <= max(y1, y2) + tolerance:
            return True

        return False

    def _add_node_features(self):
        """Add additional node features based on graph structure"""
        for node in self.G.nodes():
            # Get degree (number of connections)
            degree = self.G.degree[node]

            # Determine if node might be a corner (degree 2, extreme position)
            pos = self.G.nodes[node]['norm_pos']
            is_corner = degree == 2 and (abs(pos[0]) > 0.8 or abs(pos[1]) > 0.8)

            # Update node attributes
            self.G.nodes[node].update({
                'degree': degree,
                'is_corner': is_corner,
            })

    def get_node_features(self, node) -> Dict:
        """Get feature vector for a node to aid in matching"""
        return {
            'degree': self.G.nodes[node]['degree'],
            'x_normalized': self.G.nodes[node]['norm_pos'][0],
            'y_normalized': self.G.nodes[node]['norm_pos'][1],
            'is_corner': int(self.G.nodes[node]['is_corner'])
        }