import heapq
import math
import numpy as np
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
import cv2

@dataclass
class Node:
    x: int
    y: int
    g_cost: float = 0  # Distance from start
    h_cost: float = 0  # Heuristic distance to goal
    f_cost: float = 0  # Total cost (g + h)
    parent: Optional['Node'] = None
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))

@dataclass
class ParkingSpot:
    name: str
    coords: Set[Tuple[int, int]]
    free: bool = True
    
    def get_valid_cells(self):
        return self.coords
    
    def get_entrance_cells(self):
        """Get entrance cells based on parking row location"""
        rows = [r for r, c in self.coords]
        cols = [c for r, c in self.coords]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        
        if min_r < 3:
            # Top row (1–17): entrance from bottom
            return {(max_r + 1, c) for c in range(min_c, max_c + 1)}
        elif 6 <= min_r <= 8:
            # Middle row 1 (18–26): entrance from top
            return {(min_r - 1, c) for c in range(min_c, max_c + 1)}
        elif 9 <= min_r <= 11:
            # Middle row 2 (27–35): entrance from bottom
            return {(max_r + 1, c) for c in range(min_c, max_c + 1)}
        elif min_r >= 15:
            # Bottom row (36–50): entrance from top
            return {(min_r - 1, c) for c in range(min_c, max_c + 1)}
        else:
            # Default fallback
            entrance = set()
            for r, c in self.coords:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                    entrance.add((r + dr, c + dc))
            return entrance

class AStarPathfinder:
    def __init__(self, grid_width: int, grid_height: int):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.obstacles = set()  # Set of (x, y) coordinates that are obstacles
        
        # Define entrance points as specified
        self.entrance_points = [(0, 14), (0, 15), (0, 16), (0, 17)]
        
        # Movement directions (8-directional movement)
        self.directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        # Define forbidden areas
        self.forbidden_areas = self._define_forbidden_areas()
        
        print(f"Pathfinder initialized with {len(self.entrance_points)} entrance points")
        print(f"Forbidden areas: {len(self.forbidden_areas)} cells")
    
    def _define_forbidden_areas(self) -> Set[Tuple[int, int]]:
        """Define all forbidden areas where cars cannot go"""
        forbidden = set()
        
        # Block area from (6,0) to (17,6)
        for r in range(6, 18):  # 18 because range is exclusive at the end
            for c in range(0, 7):  # 7 to include column 6
                forbidden.add((r, c))
        
        # Block area from (6,41) to (17,47)
        for r in range(6, 18):
            for c in range(41, 48):
                forbidden.add((r, c))
        
        # Specific forbidden points
        forbidden_points = [
            (0, 6), (1, 6), (2, 6),
            (0, 27), (1, 27), (2, 27),
            (0, 34), (1, 34), (2, 34),
            (0, 41), (1, 41), (2, 41),
            (9, 20), (10, 20), (11, 20),
            (6, 27), (7, 27), (8, 27),
            (9, 27), (10, 27), (11, 27),
            (6, 20), (7, 20), (8, 20),
            (15, 13), (16, 13), (17, 13),
            (15, 20), (16, 20), (17, 20),
            (15, 27), (16, 27), (17, 27),
            (15, 34), (16, 34), (17, 34)
        ]
        
        for point in forbidden_points:
            forbidden.add(point)
        
        return forbidden
    
    def set_obstacles(self, occupied_spots: List[Tuple[int, int]]):
        """Set obstacle positions (occupied parking spots and forbidden areas)"""
        self.obstacles = set(occupied_spots)
        # Add forbidden areas to obstacles
        self.obstacles.update(self.forbidden_areas)
    
    def heuristic(self, node: Node, goal: Node) -> float:
        """Calculate heuristic distance (Manhattan distance for grid)"""
        return abs(node.x - goal.x) + abs(node.y - goal.y)
    
    def is_walkable(self, r: int, c: int) -> bool:
        """Check if a cell is walkable"""
        return (0 <= r < self.grid_height and 
                0 <= c < self.grid_width and 
                (r, c) not in self.obstacles)
    
    def get_neighbors(self, node: Node) -> List[Tuple[Node, float]]:
        """Get valid neighboring nodes with movement costs"""
        neighbors = []
        r, c = node.x, node.y  # Note: x=row, y=col in our coordinate system
        
        directions = [(-1,0),(1,0),(0,-1),(0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if self.is_walkable(nr, nc):
                # Diagonal movement costs more
                cost = 1.4 if abs(dr) + abs(dc) == 2 else 1.0
                neighbors.append((Node(nr, nc), cost))
        
        return neighbors
    
    def find_path(self, start: Tuple[int, int], goal_area: Set[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
        """Find path from start to any point in goal_area using A* algorithm (simplified version)"""
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        
        while open_set:
            _, current = heapq.heappop(open_set)

            if current in goal_area:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor, move_cost in self.get_neighbors(Node(*current)):
                neighbor_pos = (neighbor.x, neighbor.y)
                tentative_g = g_score[current] + move_cost

                if neighbor_pos not in g_score or tentative_g < g_score[neighbor_pos]:
                    came_from[neighbor_pos] = current
                    g_score[neighbor_pos] = tentative_g
                    f_score = tentative_g + min(
                        abs(neighbor_pos[0] - goal[0]) + abs(neighbor_pos[1] - goal[1])
                        for goal in goal_area
                    )
                    heapq.heappush(open_set, (f_score, neighbor_pos))

        return None

    
    def find_best_entrance_point(self, goal_area: Set[Tuple[int, int]]) -> Tuple[int, int]:
        """Find the best entrance point based on distance to goal"""
        best_entrance = self.entrance_points[0]
        min_distance = float('inf')
        
        for entrance in self.entrance_points:
            if entrance not in self.obstacles:  # Make sure entrance is not blocked
                # Calculate minimum distance to any goal point
                min_dist_to_goal = min(
                    abs(entrance[0] - goal[0]) + abs(entrance[1] - goal[1])
                    for goal in goal_area
                )
                if min_dist_to_goal < min_distance:
                    min_distance = min_dist_to_goal
                    best_entrance = entrance
        
        return best_entrance
    
    def get_path_to_spot(self, spot: ParkingSpot, occupied_spots: List[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
        """Get path from best entrance to a specific parking spot"""
        # Set obstacles (occupied spots + forbidden areas)
        self.set_obstacles(occupied_spots)
        
        # Get entrance cells for the target spot
        goal_area = spot.get_entrance_cells()
        
        # Remove the target spot's entrance from obstacles so we can reach it
        self.obstacles.difference_update(goal_area)
        
        # Find best entrance point
        best_entrance = self.find_best_entrance_point(goal_area)
        
        # Find path
        path = self.find_path(best_entrance, goal_area)
        return path
    
    def find_nearest_free_spot(self, spots: List[ParkingSpot], occupied_spots: List[Tuple[int, int]]) -> Optional[Tuple[ParkingSpot, List[Tuple[int, int]]]]:
        """Find the nearest free parking spot and return both spot and path"""
        free_spots = [spot for spot in spots if spot.free]
        
        if not free_spots:
            return None
        
        best_spot = None
        best_path = None
        best_length = float('inf')
        
        for spot in free_spots:
            path = self.get_path_to_spot(spot, occupied_spots)
            if path and len(path) < best_length:
                best_length = len(path)
                best_spot = spot
                best_path = path
        
        if best_spot and best_path:
            return best_spot, best_path
        
        return None

class PathVisualizer:
    def __init__(self, cell_size: int = 21):
        self.cell_size = cell_size
    
    def draw_path_on_grid(self, grid_image: np.ndarray, path: List[Tuple[int, int]], 
                         entrance_points: List[Tuple[int, int]], destination: Tuple[int, int] = None) -> np.ndarray:
        """Draw the path on the grid image with directional arrows"""
        if not path or len(path) < 2:
            return grid_image
        
        result_image = grid_image.copy()
        
        # Draw path as yellow line with arrows
        for i in range(len(path) - 1):
            start_point = (
                path[i][1] * self.cell_size + self.cell_size // 2,  # col * cell_size
                path[i][0] * self.cell_size + self.cell_size // 2   # row * cell_size
            )
            end_point = (
                path[i + 1][1] * self.cell_size + self.cell_size // 2,
                path[i + 1][0] * self.cell_size + self.cell_size // 2
            )
            
            # Draw yellow line
            cv2.line(result_image, start_point, end_point, (0, 255, 255), 4)  # Yellow line
            
            # Draw directional arrow
            self._draw_arrow(result_image, start_point, end_point)
        
        # Draw entrance points (blue circles)
        for entrance in entrance_points:
            entrance_pixel = (
                entrance[1] * self.cell_size + self.cell_size // 2,
                entrance[0] * self.cell_size + self.cell_size // 2
            )
            cv2.circle(result_image, entrance_pixel, 8, (255, 0, 0), -1)  # Blue circle
        
        # Highlight the actual entrance used (larger blue circle)
        if path:
            actual_entrance = (
                path[0][1] * self.cell_size + self.cell_size // 2,
                path[0][0] * self.cell_size + self.cell_size // 2
            )
            cv2.circle(result_image, actual_entrance, 12, (255, 0, 0), 3)  # Blue circle outline
            cv2.putText(result_image, "START", (actual_entrance[0] - 25, actual_entrance[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw destination marker
        if destination:
            dest_pixel = (
                destination[1] * self.cell_size + self.cell_size // 2,
                destination[0] * self.cell_size + self.cell_size // 2
            )
            cv2.circle(result_image, dest_pixel, 10, (0, 0, 255), -1)  # Red circle
            cv2.putText(result_image, "DEST", (dest_pixel[0] - 20, dest_pixel[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_image
    
    def _draw_arrow(self, image: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]):
        """Draw directional arrow on the path"""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        if dx == 0 and dy == 0:
            return
        
        # Calculate arrow direction
        angle = math.atan2(-dy, dx)  # Negative dy because y increases downward
        
        # Arrow properties
        arrow_length = 8
        arrow_angle = math.pi / 6  # 30 degrees
        
        # Calculate arrow head points
        x1 = end[0] - int(arrow_length * math.cos(angle - arrow_angle))
        y1 = end[1] + int(arrow_length * math.sin(angle - arrow_angle))
        x2 = end[0] - int(arrow_length * math.cos(angle + arrow_angle))
        y2 = end[1] + int(arrow_length * math.sin(angle + arrow_angle))
        
        # Draw arrow head
        cv2.line(image, end, (x1, y1), (0, 0, 255), 2)  # Red arrow
        cv2.line(image, end, (x2, y2), (0, 0, 255), 2)  # Red arrow
    
    def create_path_data(self, path: List[Tuple[int, int]], entrance_points: List[Tuple[int, int]], 
                        destination: Tuple[int, int] = None, spot_name: str = None) -> dict:
        """Create path data for API response"""
        if not path:
            return {
                "path": [],
                "entrance_points": [[p[0], p[1]] for p in entrance_points],
                "entrance_used": None,
                "destination": None,
                "path_length": 0,
                "has_path": False,
                "spot_name": None
            }
        
        return {
            "path": [[p[0], p[1]] for p in path],
            "entrance_points": [[p[0], p[1]] for p in entrance_points],
            "entrance_used": [path[0][0], path[0][1]] if path else None,
            "destination": [destination[0], destination[1]] if destination else None,
            "path_length": len(path),
            "has_path": True,
            "spot_name": spot_name
        }
