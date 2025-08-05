import cv2
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional
import random
import math
from ultralytics import YOLO
import os
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from datetime import datetime
from pathfinding import AStarPathfinder, PathVisualizer, ParkingSpot

@dataclass
class ParkingSlot:
    id: int
    points: List[List[int]]
    occupied: bool = False
    confidence: float = 0.0
    detected_cars: List[Dict] = None
    coverage_percentage: float = 0.0
    
    def __post_init__(self):
        if self.detected_cars is None:
            self.detected_cars = []

@dataclass
class Spot:
    name: str
    coords: List[Tuple[int, int]]
    free: bool = True
    confidence: float = 0.0

class ParkingDetector:
    def __init__(self):
        # Grid configuration
        self.CELL_SIZE = 21
        self.ROWS = 18
        self.COLS = 48
        self.WIDTH = self.COLS * self.CELL_SIZE
        self.HEIGHT = self.ROWS * self.CELL_SIZE
        
        # Coverage threshold for occupancy detection
        self.OCCUPANCY_THRESHOLD = 0.30  # 30% coverage required
        
        # Initialize pathfinding with specific entrance points and forbidden areas
        self.pathfinder = AStarPathfinder(self.COLS, self.ROWS)
        self.path_visualizer = PathVisualizer(self.CELL_SIZE)
        
        # Load YOLO model
        self.yolo_model = self.load_yolo_model()
        
        # Load parking slots from JSON
        self.parking_slots = self.load_parking_slots()

        # Initialize original parking spots for grid mapping (keep existing grid logic)
        self.spots = self._define_parking_spots()
        self.total_spots = len(self.spots)
        
        # Load complete CCTV collage image
        self.collage_image = self.load_collage_image()
        
        # Load background image if available
        self.background_image = None
        try:
            self.background_image = cv2.imread('image4.png')
            if self.background_image is not None:
                self.background_image = cv2.resize(self.background_image, (self.WIDTH, self.HEIGHT))
                print("✓ Grid background image (image4.png) loaded successfully")
            else:
                print("✗ Failed to load image4.png")
        except Exception as e:
            print(f"Error loading background image: {e}")
        
        print(f"Initialized ParkingDetector with {len(self.parking_slots)} slots")
        print(f"Occupancy threshold set to {self.OCCUPANCY_THRESHOLD*100}%")
        print(f"Pathfinding entrance set to: {self.pathfinder.entrance_points}")
    
    def load_yolo_model(self):
        """Load YOLO model for car detection"""
        try:
            # Try to load YOLOv8 model (you can change to yolov8s.pt, yolov8m.pt, etc.)
            model = YOLO('yolo12n.pt')  # nano version for faster inference
            print("YOLO model loaded successfully")
            return model
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            print("Please install ultralytics: pip install ultralytics")
            return None
    
    def load_parking_slots(self):
        """Load parking slots from JSON file"""
        try:
            with open('parking_slots_Collage_Image.json', 'r') as f:
                slots_data = json.load(f)
            
            parking_slots = []
            for slot_data in slots_data:
                slot = ParkingSlot(
                    id=slot_data['id'],
                    points=slot_data['points']
                )
                parking_slots.append(slot)
            
            print(f"Loaded {len(parking_slots)} parking slots from JSON")
            return parking_slots
            
        except Exception as e:
            print(f"Failed to load parking slots: {e}")
            return []
    
    def load_collage_image(self):
        """Load the complete collage image at original resolution"""
        try:
            # Try to load the new collage image first
            collage_path = "Images/Collage_Image3.png"
            if os.path.exists(collage_path):
                collage = cv2.imread(collage_path)
                if collage is not None:
                    height, width = collage.shape[:2]
                    print(f"✓ Loaded complete collage image: {width} x {height} (W x H)")
                    return collage
            
            # Fallback to original collage
            collage_path = "Images/Collage_Image.png"
            if os.path.exists(collage_path):
                collage = cv2.imread(collage_path)
                if collage is not None:
                    height, width = collage.shape[:2]
                    print(f"✓ Loaded fallback collage image: {width} x {height} (W x H)")
                    return collage
            
            print("✗ No collage image found, creating placeholder")
            # Create placeholder image
            placeholder = np.ones((1200, 1600, 3), dtype=np.uint8) * 128
            return placeholder
            
        except Exception as e:
            print(f"Error loading collage image: {e}")
            # Create placeholder image
            placeholder = np.ones((1200, 1600, 3), dtype=np.uint8) * 128
            return placeholder
    
    def get_current_cctv_image(self):
        """Get the complete collage image"""
        return self.collage_image.copy() if self.collage_image is not None else None
    
    def detect_cars_in_image(self, image):
        """Detect cars in image using YOLO"""
        if self.yolo_model is None:
            return []
        
        try:
            # Run YOLO detection
            results = self.yolo_model(image, conf=0.3, classes=[2])  # class 2 is 'car' in COCO dataset
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                        })
            
            return detections
            
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return []
    
    def calculate_polygon_area(self, points):
        """Calculate area of a polygon using shoelace formula"""
        if len(points) < 3:
            return 0
        
        n = len(points)
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        return abs(area) / 2
    
    def calculate_coverage_percentage(self, car_bbox, parking_slot_points):
        """Calculate what percentage of the parking slot is covered by the car"""
        try:
            # Create polygon for parking slot
            slot_polygon = Polygon(parking_slot_points)
            
            # Create rectangle for car bounding box
            x1, y1, x2, y2 = car_bbox
            car_rectangle = box(x1, y1, x2, y2)
            
            # Calculate intersection
            if slot_polygon.is_valid and car_rectangle.is_valid:
                intersection = slot_polygon.intersection(car_rectangle)
                
                # Calculate coverage percentage
                if intersection.area > 0:
                    coverage_percentage = intersection.area / slot_polygon.area
                    return min(coverage_percentage, 1.0)  # Cap at 100%
            
            return 0.0
            
        except Exception as e:
            print(f"Error calculating coverage: {e}")
            return 0.0
    
    def get_spot_grid_coordinates(self, spot: Spot) -> List[Tuple[int, int]]:
        """Get grid coordinates for a parking spot"""
        return [(c, r) for r, c in spot.coords]  # Convert (row, col) to (x, y)
    
    def _convert_spots_to_parking_spots(self) -> List[ParkingSpot]:
        """Convert internal spots to ParkingSpot objects for pathfinding"""
        parking_spots = []
        for spot in self.spots:
            # Convert coordinates from (row, col) to (row, col) set
            coords = {(r, c) for r, c in spot.coords}
            parking_spot = ParkingSpot(
                name=spot.name,
                coords=coords,
                free=spot.free
            )
            parking_spots.append(parking_spot)
        return parking_spots
    
    def calculate_path_to_nearest_spot(self) -> Optional[dict]:
        """Calculate path from entrance to nearest free parking spot"""
        try:
            # Convert spots to ParkingSpot objects
            parking_spots = self._convert_spots_to_parking_spots()
            
            # Get occupied spot coordinates for obstacles
            occupied_spots = []
            for spot in parking_spots:
                if not spot.free:
                    occupied_spots.extend(list(spot.coords))
            
            # Find nearest free spot and path
            result = self.pathfinder.find_nearest_free_spot(parking_spots, occupied_spots)
            
            if result:
                best_spot, path = result
                # Get the destination (last point in path or spot entrance)
                destination = path[-1] if path else None
                
                path_data = self.path_visualizer.create_path_data(
                    path, 
                    self.pathfinder.entrance_points, 
                    destination,
                    best_spot.name
                )
                print(f"Path found: {len(path)} steps to spot {best_spot.name} at {destination}")
                return path_data
            else:
                print("No path found to any free spot")
                return None
            
        except Exception as e:
            print(f"Error calculating path: {e}")
            return None
    
    def detect_cars_in_parking_slots(self):
        """Main detection function that processes complete collage image and updates parking slots"""
        # Get complete collage image at original resolution
        current_image = self.get_current_cctv_image()
        if current_image is None:
            return self.create_empty_grid(), None, {'total_spots': 0, 'occupied_spots': 0, 'free_spots': 0}
    
        # Print image dimensions for debugging
        height, width = current_image.shape[:2]
        print(f"Processing complete collage image at resolution: {width} x {height}")
    
        # Detect cars in the complete image
        car_detections = self.detect_cars_in_image(current_image)
        print(f"Detected {len(car_detections)} cars in the complete image")
    
        # Create annotated image
        annotated_image = current_image.copy()
    
        # Calculate appropriate line thickness and text size based on image resolution
        line_thickness = max(2, int(width / 800))  # Scale line thickness
        text_scale = max(0.4, min(1.0, width / 1200))  # Scale text size
        text_thickness = max(1, int(text_scale * 2))
    
        # Reset all slots to empty
        for slot in self.parking_slots:
            slot.occupied = False
            slot.confidence = 0.0
            slot.detected_cars = []
            slot.coverage_percentage = 0.0
    
        # Check each car detection against parking slots with coverage analysis
        total_confidence = 0.0
        detection_count = 0
    
        for detection in car_detections:
            car_center = detection['center']
            car_bbox = detection['bbox']
            car_confidence = detection['confidence']
        
            # Draw car detection on annotated image with scaled thickness
            x1, y1, x2, y2 = car_bbox
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 255), line_thickness)  # Yellow for cars
            cv2.putText(annotated_image, f'Car {car_confidence:.2f}', 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 255), text_thickness)
        
            # Check coverage for each parking slot
            for slot in self.parking_slots:
                # Calculate coverage percentage
                coverage = self.calculate_coverage_percentage(car_bbox, slot.points)
            
                # Update slot if this car has higher coverage
                if coverage > slot.coverage_percentage:
                    slot.coverage_percentage = coverage
                
                    # Mark as occupied only if coverage exceeds threshold
                    if coverage >= self.OCCUPANCY_THRESHOLD:
                        slot.occupied = True
                        slot.confidence = car_confidence
                        slot.detected_cars = [detection]
                        total_confidence += car_confidence
                        detection_count += 1
                    
                        print(f"Slot {slot.id}: {coverage*100:.1f}% coverage - OCCUPIED")
    
        # Draw parking slots on annotated image with coverage info
        for slot in self.parking_slots:
            points = np.array(slot.points, np.int32)
        
            # Color based on occupancy and coverage
            if slot.occupied:
                color = (0, 0, 255)  # Red for occupied
                status_text = f"#{slot.id}: OCC ({slot.coverage_percentage*100:.1f}%)"
            else:
                if slot.coverage_percentage > 0:
                    color = (0, 165, 255)  # Orange for partial coverage
                    status_text = f"#{slot.id}: PARTIAL ({slot.coverage_percentage*100:.1f}%)"
                else:
                    color = (0, 255, 0)  # Green for free
                    status_text = f"#{slot.id}: FREE"
        
            # Draw parking slot outline with scaled thickness
            cv2.polylines(annotated_image, [points], True, color, line_thickness)
        
            # Add slot ID, status, and coverage percentage
            center_x = int(np.mean([p[0] for p in slot.points]))
            center_y = int(np.mean([p[1] for p in slot.points]))
        
            # Adjust text position and size based on image resolution
            text_offset = max(30, int(width / 40))
            cv2.putText(annotated_image, status_text, (center_x-text_offset, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, text_scale, color, text_thickness)

        # Map YOLO detections to original grid spots for visualization
        self._map_detections_to_grid_spots()

        # Create grid representation using original logic
        grid_data = self.get_latest_detection()[0]  # Use original grid creation
    
        # Calculate statistics
        occupied_count = sum(1 for slot in self.parking_slots if slot.occupied)
        partial_count = sum(1 for slot in self.parking_slots if slot.coverage_percentage > 0 and not slot.occupied)
        total_spots = len(self.parking_slots)
        free_count = total_spots - occupied_count
        avg_confidence = total_confidence / detection_count if detection_count > 0 else 0.0
        avg_coverage = sum(slot.coverage_percentage for slot in self.parking_slots) / total_spots if total_spots > 0 else 0.0
    
        detection_stats = {
            'total_spots': total_spots,
            'occupied_spots': occupied_count,
            'free_spots': free_count,
            'partial_coverage_spots': partial_count,
            'avg_confidence': avg_confidence,
            'avg_coverage': avg_coverage,
            'total_detections': len(car_detections),
            'occupancy_threshold': self.OCCUPANCY_THRESHOLD,
            'image_resolution': f"{width}x{height}"
        }
    
        print(f"Detection Summary: {occupied_count} occupied, {partial_count} partial, {free_count} free (threshold: {self.OCCUPANCY_THRESHOLD*100}%)")
        print(f"Complete image processed at: {width}x{height}")
    
        return grid_data, annotated_image, detection_stats
    
    def display_annotated_image(self, annotated_image, window_name="Complete CCTV Detection"):
        """Display the annotated complete collage image with YOLO detections"""
        if annotated_image is not None:
            # Get original dimensions
            height, width = annotated_image.shape[:2]
            print(f"Displaying complete collage at original resolution: {width} x {height}")
        
            # Resize for display if image is too large for screen
            display_image = annotated_image.copy()
            scale_factor = 1.0
        
            # Scale down if larger than typical screen size
            max_display_width = 1400
            max_display_height = 900
        
            if width > max_display_width or height > max_display_height:
                scale_factor = min(max_display_width/width, max_display_height/height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                display_image = cv2.resize(display_image, (new_width, new_height))
                print(f"Resized for display to: {new_width} x {new_height} (scale: {scale_factor:.2f})")
        
            # Add detection statistics overlay (adjust text size based on image size)
            text_scale = max(0.6, min(1.2, width / 1000))  # Scale text based on image width
            text_thickness = max(1, int(text_scale * 2))
        
            stats_text = [
                f"Complete Collage: {width} x {height}",
                f"Total Slots: {len(self.parking_slots)}",
                f"Occupied: {sum(1 for slot in self.parking_slots if slot.occupied)}",
                f"Free: {sum(1 for slot in self.parking_slots if not slot.occupied)}",
                f"Partial Coverage: {sum(1 for slot in self.parking_slots if slot.coverage_percentage > 0 and not slot.occupied)}",
                f"Threshold: {self.OCCUPANCY_THRESHOLD*100:.0f}%",
                f"Cars Detected: {sum(len(slot.detected_cars) for slot in self.parking_slots)}",
                f"Press 'q' to close, 'r' to refresh, 's' to save"
            ]
        
            # Calculate overlay size based on text
            overlay_height = len(stats_text) * 35 + 30
            overlay_width = 500
        
            # Add semi-transparent background for text
            overlay = display_image.copy()
            cv2.rectangle(overlay, (10, 10), (overlay_width, overlay_height), (0, 0, 0), -1)
            cv2.addWeighted(display_image, 0.7, overlay, 0.3, 0, display_image)
        
            # Add text with appropriate scaling
            for i, text in enumerate(stats_text):
                y_pos = 40 + i * 35
                cv2.putText(display_image, text, (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), text_thickness)
        
            # Display the image
            #cv2.imshow(window_name, display_image)
        
            # Handle key presses (non-blocking)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return False  # Signal to stop display
            elif key == ord('r'):
                print("Manual refresh requested")
            elif key == ord('s'):
                # Save current annotated image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"complete_detection_{timestamp}.jpg"
                cv2.imwrite(filename, annotated_image)  # Save original resolution
                print(f"Complete detection snapshot saved as {filename}")
        
            return True  # Continue display
        return False
    
    def create_grid_from_slots(self):
        """Create 2D grid representation from parking slots"""
        # Initialize empty grid
        grid = [[{"status": "empty", "confidence": 0.0} for _ in range(self.COLS)] for _ in range(self.ROWS)]
        
        # Map parking slots to grid cells (simplified mapping)
        for slot in self.parking_slots:
            # Calculate approximate grid position based on slot center
            center_x = int(np.mean([p[0] for p in slot.points]))
            center_y = int(np.mean([p[1] for p in slot.points]))
            
            # Convert to grid coordinates (this is a simplified mapping)
            # You might need to adjust this based on your actual image dimensions
            grid_col = min(int((center_x / 2000) * self.COLS), self.COLS - 1)  # Assuming image width ~2000px
            grid_row = min(int((center_y / 1500) * self.ROWS), self.ROWS - 1)  # Assuming image height ~1500px
            
            if 0 <= grid_row < self.ROWS and 0 <= grid_col < self.COLS:
                grid[grid_row][grid_col] = {
                    "status": "occupied" if slot.occupied else "empty",
                    "confidence": slot.confidence,
                    "spot_id": str(slot.id),
                    "coverage": slot.coverage_percentage
                }
        
        return grid
    
    def create_empty_grid(self):
        """Create empty grid when no detection is possible"""
        return [[{"status": "empty", "confidence": 0.0} for _ in range(self.COLS)] for _ in range(self.ROWS)]
    
    def get_spots_info(self):
        """Get information about all parking spots"""
        return [
            {
                "id": slot.id,
                "status": "occupied" if slot.occupied else "empty",
                "confidence": slot.confidence,
                "coverage_percentage": slot.coverage_percentage,
                "points": slot.points,
                "detected_cars": len(slot.detected_cars),
                "above_threshold": slot.coverage_percentage >= self.OCCUPANCY_THRESHOLD
            }
            for slot in self.parking_slots
        ]

    def _map_detections_to_grid_spots(self):
        """Map YOLO detections from parking_slots to original grid spots for visualization"""
        # Reset all grid spots
        for spot in self.spots:
            spot.free = True
            spot.confidence = 0.0
        
        # Map occupied parking_slots to corresponding grid spots
        # This is a simplified mapping - you may need to adjust based on your specific layout
        slot_to_spot_mapping = {
            # Top row spots (1-17 map to spots 17-1)
            1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 
            10: "10", 11: "11", 12: "12", 13: "13", 14: "14", 15: "15", 16: "16", 17: "17",
            
            # Center rows (18-35 map to spots 18-35)
            18: "18", 19: "19", 20: "20", 21: "21", 22: "22", 23: "23", 24: "24", 25: "25", 26: "26",
            27: "27", 28: "28", 29: "29", 30: "30", 31: "31", 32: "32", 33: "33", 34: "34", 35: "35",
            
            # Bottom rows (36-50 map to spots 36-50)
            36: "36", 37: "37", 38: "38", 39: "39", 40: "40", 41: "41", 42: "42", 43: "43", 44: "44",
            45: "45", 46: "46", 47: "47", 48: "48", 49: "49", 50: "50"
        }
        
        # Update grid spots based on parking slot occupancy
        for parking_slot in self.parking_slots:
            if parking_slot.id in slot_to_spot_mapping:
                spot_name = slot_to_spot_mapping[parking_slot.id]
                # Find the corresponding grid spot
                for spot in self.spots:
                    if spot.name == spot_name:
                        spot.free = not parking_slot.occupied
                        spot.confidence = parking_slot.confidence
                        break

    def get_latest_detection(self) -> Tuple[List[List[Dict]], np.ndarray]:
        """Get latest parking detection results using original grid logic"""
        # Create 2D grid using original logic
        grid = [[{"status": "empty", "confidence": 0.0} for _ in range(self.COLS)] for _ in range(self.ROWS)]
        
        # Fill grid with spot data using original coordinates
        for spot in self.spots:
            status = "empty" if spot.free else "occupied"
            for (r, c) in spot.coords:
                if 0 <= r < self.ROWS and 0 <= c < self.COLS:
                    grid[r][c] = {
                        "status": status,
                        "confidence": spot.confidence,
                        "spot_id": spot.name
                    }
        
        # Create annotated grid image (not the CCTV image)
        grid_image = self._create_grid_visualization()
        
        return grid, grid_image

    def _create_grid_visualization(self) -> np.ndarray:
        """Create grid visualization image using the original image4.png as base"""
        if self.background_image is not None:
            # Use the original grid image as base
            grid_image = self.background_image.copy()
        else:
            # Fallback: create synthetic grid
            grid_image = np.ones((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8) * 240
        
        # Draw grid lines
        for i in range(self.ROWS + 1):
            y = i * self.CELL_SIZE
            cv2.line(grid_image, (0, y), (self.WIDTH, y), (200, 200, 200), 1)
        
        for j in range(self.COLS + 1):
            x = j * self.CELL_SIZE
            cv2.line(grid_image, (x, 0), (x, self.HEIGHT), (200, 200, 200), 1)

        # Overlay parking spot status on the grid image
        for spot in self.spots:
            # Get bounding box coordinates
            rows = [r for r, c in spot.coords]
            cols = [c for r, c in spot.coords]
            
            if rows and cols:
                min_r, max_r = min(rows), max(rows)
                min_c, max_c = min(cols), max(cols)
                
                # Convert to pixel coordinates
                x1 = min_c * self.CELL_SIZE
                y1 = min_r * self.CELL_SIZE
                x2 = (max_c + 1) * self.CELL_SIZE
                y2 = (max_r + 1) * self.CELL_SIZE
                
                # Choose color based on occupancy (overlay on existing image)
                if not spot.free:  # Occupied
                    # Add red overlay for occupied spots
                    overlay = grid_image[y1:y2, x1:x2].copy()
                    overlay[:, :] = [0, 0, 255]  # Red
                    cv2.addWeighted(grid_image[y1:y2, x1:x2], 0.7, overlay, 0.3, 0, grid_image[y1:y2, x1:x2])
                    
                    # Add border
                    cv2.rectangle(grid_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                else:  # Free
                    # Add green overlay for free spots (lighter)
                    overlay = grid_image[y1:y2, x1:x2].copy()
                    overlay[:, :] = [0, 255, 0]  # Green
                    cv2.addWeighted(grid_image[y1:y2, x1:x2], 0.9, overlay, 0.1, 0, grid_image[y1:y2, x1:x2])
                    
                    # Add border
                    cv2.rectangle(grid_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Calculate and draw path with directional arrows
        path_data = self.calculate_path_to_nearest_spot()
        if path_data and path_data.get('has_path'):
            # Convert path coordinates back to (row, col) format
            path = [(p[0], p[1]) for p in path_data['path']]
            entrance_points = [(p[0], p[1]) for p in path_data['entrance_points']]
            destination = tuple(path_data['destination']) if path_data['destination'] else None
            
            grid_image = self.path_visualizer.draw_path_on_grid(
                grid_image, 
                path, 
                entrance_points,
                destination
            )

        return grid_image

    def _define_parking_spots(self) -> List[Spot]:
        """Define parking spots based on the image4.png layout"""
        spot_map = [
            # Top Row (right to left: 1-17)
            ("17", [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)], False),
            ("16", [(0, 2), (0, 3), (1, 2), (1, 3), (2, 2), (2, 3)], False),
            ("15", [(0, 4), (0, 5), (1, 4), (1, 5), (2, 4), (2, 5)], False),
            ("14", [(0, 7), (0, 8), (1, 7), (1, 8), (2, 7), (2, 8)], False),
            ("13", [(0, 9), (0, 10), (1, 9), (1, 10), (2, 9), (2, 10)], False),
            ("12", [(0, 11), (0, 12), (1, 11), (1, 12), (2, 11), (2, 12)], False),
            ("11", [(0, 23), (0, 24), (1, 23), (1, 24), (2, 23), (2, 24)], False),
            ("10", [(0, 25), (0, 26), (1, 25), (1, 26), (2, 25), (2, 26)], False),
            ("9",  [(0, 28), (0, 29), (1, 28), (1, 29), (2, 28), (2, 29)], False),
            ("8",  [(0, 30), (0, 31), (1, 30), (1, 31), (2, 30), (2, 31)], False),
            ("7",  [(0, 32), (0, 33), (1, 32), (1, 33), (2, 32), (2, 33)], False),
            ("6",  [(0, 35), (0, 36), (1, 35), (1, 36), (2, 35), (2, 36)], False),
            ("5",  [(0, 37), (0, 38), (1, 37), (1, 38), (2, 37), (2, 38)], False),
            ("4",  [(0, 39), (0, 40), (1, 39), (1, 40), (2, 39), (2, 40)], False),
            ("3",  [(0, 42), (0, 43), (1, 42), (1, 43), (2, 42), (2, 43)], False),
            ("2",  [(0, 44), (0, 45), (1, 44), (1, 45), (2, 44), (2, 45)], False),
            ("1",  [(0, 46), (0, 47), (1, 46), (1, 47), (2, 46), (2, 47)], True),
        
            # Center row 1 (19-27)
            ("19", [(6,14), (6, 15), (7, 14), (7,15), (8,14), (8,15)], False),
            ("20", [(6,16), (6,17), (7,16), (7,17), (8,16), (8,17)], False),
            ("21", [(6,18), (6,19), (7,18), (7,19), (8,18), (8,19)], True),
            ("22", [(6,21), (6,22), (7,21), (7,22), (8,21), (8,22)], False),
            ("23", [(6,23), (6,24), (7,23), (7,24), (8,23), (8,24)], False),
            ("24", [(6,25), (6,26), (7,25), (7,26), (8,25), (8,26)], False),
            ("25", [(6,28), (6,29), (7,28), (7,29), (8,28), (8,29)], False),
            ("26", [(6,30), (6,31), (7,30), (7,31), (8,30), (8,31)], False),
            ("27", [(6,32), (6,33), (7,32), (7,33), (8,32), (8,33)], False),
        
            # Center row 2 (28-36)
            ("28", [(9,14), (9,15), (10,14), (10,15), (11,14), (11,15)], False),
            ("29", [(9,16), (9,17), (10,16), (10,17), (11,16), (11,17)], True),
            ("30", [(9,18), (9,19), (10,18), (10,19), (11,18), (11,19)], True),
            ("31", [(9,21), (9,22), (10,21), (10,22), (11,21), (11,22)], True),
            ("32", [(9,23), (9,24), (10,23), (10,24), (11,23), (11,24)], True),
            ("33", [(9,25), (9,26), (10,25), (10,26), (11,25), (11,26)], True),
            ("34", [(9,28), (9,29), (10,28), (10,29), (11,28), (11,29)], True),
            ("35", [(9,30), (9,31), (10,30), (10,31), (11,30), (11,31)], True),
            ("36", [(9,32), (9,33), (10,32), (10,33), (11,32), (11,33)], True),
        
            # Bottom row (37-50)
            ("37", [(15,0), (15,1), (16,0), (16,1), (17,0), (17,1)], False),
            ("38", [(15,2), (15,3), (16,2), (16,3), (17,2), (17,3)], False),
            ("39", [(15,4), (15,5), (16,4), (16,5), (17,4), (17,5)], False),
            ("40", [(15,7), (15,8), (16,7), (16,8), (17,7), (17,8)], False),
            ("41", [(15,9), (15,10), (16,9), (16,10), (17,9), (17,10)], False),
            ("42", [(15,11), (15,12), (16,11), (16,12), (17,11), (17,12)], False),
            ("43", [(15,23), (15,24), (16,23), (16,24), (17,23), (17,24)], False),
            ("44", [(15,25), (15,26), (16,25), (16,26), (17,25), (17,26)], False),
            ("45", [(15,28), (15,29), (16,28), (16,29), (17,28), (17,29)], False),
            ("46", [(15,30), (15,31), (16,30), (16,31), (17,30), (17,31)], False),
            ("47", [(15,32), (15,33), (16,32), (16,33), (17,32), (17,33)], False),
            ("48", [(15,35), (15,36), (16,35), (16,36), (17,35), (17,36)], False),
            ("49", [(15,37), (15,38), (16,37), (16,38), (17,37), (17,38)], False),
            ("50", [(15,39), (15,40), (16,39), (16,40), (17,39), (17,40)], False),
        ]
        
        spots = [Spot(name, coords, free) for name, coords, free in spot_map]
        return spots
