"""
Utility functions for Parking Detection System
"""
import cv2
import numpy as np
import json
import os

def calculate_polygon_iou(polygon1, box):
    """
    Calculate Intersection over Union (IoU) between a polygon and a bounding box.
    
    Args:
        polygon1: List of points defining the polygon
        box: Bounding box coordinates [x1, y1, x2, y2]
    
    Returns:
        float: IoU value between 0 and 1
    """
    x1, y1, x2, y2 = box
    polygon2 = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ], dtype=np.int32)

    poly1 = np.array(polygon1, dtype=np.int32)
    poly2 = polygon2

    # Convert to float for intersection calculation
    poly1f = np.array(poly1, dtype=np.float32)
    poly2f = np.array(poly2, dtype=np.float32)

    inter_area, _ = cv2.intersectConvexConvex(poly1f, poly2f)

    if inter_area == 0:
        return 0.0

    area1 = cv2.contourArea(poly1)
    area2 = cv2.contourArea(poly2)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area

def save_slots_to_json(filename, slots_data):
    """
    Save the list of parking slots and their polygon points to a JSON file.
    
    Args:
        filename: Path to save the JSON file
        slots_data: List of parking slot data
    """
    with open(filename, 'w') as f:
        json.dump(slots_data, f, indent=4)
    print(f"Saved {len(slots_data)} parking slots to {filename}")

def load_slots_from_json(filename):
    """
    Load parking slots from a JSON file.
    
    Args:
        filename: Path to the JSON file
    
    Returns:
        list: List of parking slot data
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Parking slots file not found: {filename}")
    
    with open(filename, 'r') as f:
        slots = json.load(f)
    return slots

def get_image_resolution(image_path):
    """
    Get the resolution of an image.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        tuple: (width, height) of the image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    height, width = img.shape[:2]
    return width, height

def create_output_filename(base_name, suffix=""):
    """
    Create an output filename with timestamp or user input.
    
    Args:
        base_name: Base name for the file
        suffix: Optional suffix to add
    
    Returns:
        str: Generated filename
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if suffix:
        return f"{base_name}_{suffix}_{timestamp}"
    else:
        return f"{base_name}_{timestamp}"

def ensure_directory_exists(directory):
    """
    Ensure that a directory exists, create it if it doesn't.
    
    Args:
        directory: Path to the directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def validate_video_file(video_path):
    """
    Validate that a video file exists and can be opened.
    
    Args:
        video_path: Path to the video file
    
    Returns:
        bool: True if valid, False otherwise
    """
    if not os.path.exists(video_path):
        return False
    
    cap = cv2.VideoCapture(video_path)
    is_valid = cap.isOpened()
    cap.release()
    return is_valid

def get_video_info(video_path):
    """
    Get information about a video file.
    
    Args:
        video_path: Path to the video file
    
    Returns:
        dict: Video information (fps, width, height, frame_count)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    info = {
        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    
    cap.release()
    return info
