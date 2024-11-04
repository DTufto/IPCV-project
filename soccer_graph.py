import numpy as np
from dataclasses import dataclass
import networkx as nx


@dataclass
class PenaltyAreaMeasurements:
    """Standard soccer penalty area measurements in meters"""
    PENALTY_AREA_LENGTH = 16.5  # Distance from goal line
    PENALTY_AREA_WIDTH = 40.32  # Total width (16.5m from each goal post)
    GOAL_AREA_LENGTH = 5.5  # Distance from goal line
    GOAL_AREA_WIDTH = 18.32  # Total width (5.5m from each goal post)
    PENALTY_MARK_DIST = 11.0  # Distance of penalty spot from goal line


class PenaltyAreaGraph:
    def __init__(self):
        """Initialize a graph for penalty area from front view perspective"""
        self.G = nx.Graph()
        self.measurements = PenaltyAreaMeasurements()
        self._create_reference_points()
        self._create_graph_connections()

    def _create_reference_points(self):
        """Create intersection points visible in typical front view"""
        m = self.measurements

        # Set origin at the center of goal line at ground level
        self.points = {
            # Goal line intersections with penalty area lines
            'goal_line_left': (0, m.PENALTY_AREA_WIDTH / 2, 0),
            'goal_line_right': (0, -m.PENALTY_AREA_WIDTH / 2, 0),

            # Goal line intersections with goal area lines
            'goal_line_small_left': (0, m.GOAL_AREA_WIDTH / 2, 0),
            'goal_line_small_right': (0, -m.GOAL_AREA_WIDTH / 2, 0),

            # Goal area (5.5m) corners
            'goal_area_left': (m.GOAL_AREA_LENGTH, m.GOAL_AREA_WIDTH / 2, 0),
            'goal_area_right': (m.GOAL_AREA_LENGTH, -m.GOAL_AREA_WIDTH / 2, 0),

            # Penalty area (16.5m) corners
            'penalty_area_left': (m.PENALTY_AREA_LENGTH, m.PENALTY_AREA_WIDTH / 2, 0),
            'penalty_area_right': (m.PENALTY_AREA_LENGTH, -m.PENALTY_AREA_WIDTH / 2, 0),
        }

        # Add nodes to graph with position and semantic attributes
        for label, pos in self.points.items():
            attributes = {
                'pos': pos,  # 3D position (x, y, z)
                'x': pos[0],  # Distance from goal line
                'y': pos[1],  # Lateral distance from center
                'z': pos[2],  # Height from ground
                'is_goal_line': 'goal_line' in label,
                'is_penalty_area': 'penalty_area' in label,
                'is_goal_area': 'goal_area' in label and 'line' not in label,
            }
            self.G.add_node(label, **attributes)

    def _create_graph_connections(self):
        """Create edges between visibly connected points"""
        # Goal line connections
        goal_line_pairs = [
            ('goal_line_left', 'goal_line_small_left'),
            ('goal_line_small_left', 'goal_line_small_right'),
            ('goal_line_small_right', 'goal_line_right')
        ]

        # Goal area connections
        goal_area_pairs = [
            ('goal_line_small_left', 'goal_area_left'),
            ('goal_line_small_right', 'goal_area_right'),
            ('goal_area_left', 'goal_area_right')
        ]

        # Penalty area connections
        penalty_area_pairs = [
            ('goal_line_left', 'penalty_area_left'),
            ('goal_line_right', 'penalty_area_right'),
            ('penalty_area_left', 'penalty_area_right')
        ]

        # Add all edges with real-world distances as weights
        for pair in goal_line_pairs + goal_area_pairs + penalty_area_pairs:
            pos1 = self.G.nodes[pair[0]]['pos']
            pos2 = self.G.nodes[pair[1]]['pos']
            # Calculate 3D Euclidean distance
            distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))
            self.G.add_edge(pair[0], pair[1], weight=distance)

    def get_ground_points(self):
        """Get all points (they're all at ground level in this case)"""
        return self.points

    def get_visible_points(self):
        """Get all points that could be visible as line intersections"""
        return self.points