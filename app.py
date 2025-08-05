from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import json
import time
import base64
import cv2
import numpy as np
import threading
from datetime import datetime
from parking_detector import ParkingDetector
import io
import os

app = Flask(__name__)
CORS(app)

# Global parking detector instance
detector = ParkingDetector()

# Global state for caching latest data
latest_data = {
    "grid": [],
    "image": "",
    "timestamp": datetime.now().isoformat(),
    "total_spots": 50,
    "occupied_spots": 0,
    "free_spots": 50,
    "path_data": None
}

# Lock for thread safety
data_lock = threading.Lock()

# Background thread for continuous detection
detection_thread = None
stop_detection = False

# Global flag for image display
display_enabled = True
display_window_active = False

def continuous_detection():
    """Continuously detect cars and update parking data"""
    global latest_data, stop_detection, display_enabled, display_window_active
    
    while not stop_detection:
        try:
            # Get latest detection results with YOLO (returns CCTV annotated image)
            cctv_grid_data, cctv_annotated_image, detection_stats = detector.detect_cars_in_parking_slots()
            
            # Get grid visualization (returns grid image for Android app)
            grid_data, grid_image = detector.get_latest_detection()
            
            # Calculate path to nearest parking spot
            path_data = detector.calculate_path_to_nearest_spot()
            
            # Display annotated CCTV image if enabled
            if display_enabled and cctv_annotated_image is not None:
                display_window_active = True
                continue_display = detector.display_annotated_image(cctv_annotated_image)
                if not continue_display:
                    display_enabled = False
                    display_window_active = False
                    print("Display window closed by user")
            
            # Convert grid image to base64 (this is what Android app expects)
            image_base64 = ""
            if grid_image is not None:
                _, buffer = cv2.imencode('.jpg', grid_image)
                image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Also convert CCTV image for debugging/monitoring
            cctv_image_base64 = ""
            if cctv_annotated_image is not None:
                _, buffer = cv2.imencode('.jpg', cctv_annotated_image)
                cctv_image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            with data_lock:
                latest_data = {
                    "grid": grid_data,  # Use grid data for Android app
                    "image": image_base64,  # Grid visualization for Android app
                    "cctv_image": cctv_image_base64,  # CCTV detection image for monitoring
                    "timestamp": datetime.now().isoformat(),
                    "total_spots": detection_stats['total_spots'],
                    "occupied_spots": detection_stats['occupied_spots'],
                    "free_spots": detection_stats['free_spots'],
                    "detection_confidence": detection_stats.get('avg_confidence', 0.0),
                    "path_data": path_data  # Add path data
                }
            
            print(f"Detection Update: {detection_stats['occupied_spots']}/{detection_stats['total_spots']} spots occupied")
            if path_data and path_data.get('has_path'):
                print(f"Path found: {path_data['path_length']} steps to nearest spot")
            
        except Exception as e:
            print(f"Error in continuous detection: {e}")
        
        # Wait before next detection cycle
        time.sleep(3)  # Detect every 3 seconds
    
    # Clean up display window when stopping
    if display_window_active:
        cv2.destroyAllWindows()
        display_window_active = False

def start_background_detection():
    """Start the background detection thread"""
    global detection_thread, stop_detection
    
    if detection_thread is None or not detection_thread.is_alive():
        stop_detection = False
        detection_thread = threading.Thread(target=continuous_detection, daemon=True)
        detection_thread.start()
        print("Background car detection started")

@app.route('/')
def index():
    """API information endpoint"""
    return jsonify({
        "name": "Parking Occupancy API with YOLO Detection & Pathfinding",
        "version": "3.0.0",
        "endpoints": {
            "/snapshot": "Get current parking status (GET)",
            "/stream": "Server-Sent Events stream (GET)",
            "/health": "Health check (GET)",
            "/spots": "Get parking spots configuration (GET)",
            "/detection/start": "Start real-time detection (POST)",
            "/detection/stop": "Stop real-time detection (POST)",
            "/cctv": "Get CCTV detection image (GET)",
            "/display/start": "Start image display window (POST)",
            "/display/stop": "Stop image display window (POST)",
            "/display/status": "Get display status (GET)",
            "/path": "Get path to nearest parking spot (GET)",
            "/path/calculate": "Calculate new path (POST)"
        },
        "status": "running",
        "yolo_enabled": detector.yolo_model is not None,
        "display_enabled": display_enabled,
        "pathfinding_enabled": True
    })

@app.route('/snapshot')
def snapshot():
    """Get current parking occupancy snapshot"""
    try:
        # Force immediate detection (returns CCTV annotated image)
        cctv_grid_data, cctv_annotated_image, detection_stats = detector.detect_cars_in_parking_slots()
        
        # Get grid visualization (returns grid image for Android app)
        grid_data, grid_image = detector.get_latest_detection()
        
        # Calculate path to nearest parking spot
        path_data = detector.calculate_path_to_nearest_spot()
        
        # Convert grid image to base64 (for Android app)
        image_base64 = ""
        if grid_image is not None:
            _, buffer = cv2.imencode('.jpg', grid_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Convert CCTV image to base64 (for monitoring)
        cctv_image_base64 = ""
        if cctv_annotated_image is not None:
            _, buffer = cv2.imencode('.jpg', cctv_annotated_image)
            cctv_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        response_data = {
            "grid": grid_data,  # 18x48 grid for Android app
            "image": image_base64,  # Grid visualization for Android app
            "cctv_image": cctv_image_base64,  # CCTV detection image
            "timestamp": datetime.now().isoformat(),
            "total_spots": detection_stats['total_spots'],
            "occupied_spots": detection_stats['occupied_spots'],
            "free_spots": detection_stats['free_spots'],
            "detection_confidence": detection_stats.get('avg_confidence', 0.0),
            "path_data": path_data  # Include path data
        }
        
        with data_lock:
            latest_data.update(response_data)
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/path')
def get_path():
    """Get current path to nearest parking spot"""
    with data_lock:
        path_data = latest_data.get("path_data")
        return jsonify({
            "path_data": path_data,
            "timestamp": latest_data.get("timestamp", ""),
            "has_path": path_data is not None and path_data.get('has_path', False)
        })

@app.route('/path/calculate', methods=['POST'])
def calculate_path():
    """Calculate new path to nearest parking spot"""
    try:
        path_data = detector.calculate_path_to_nearest_spot()
        
        with data_lock:
            latest_data["path_data"] = path_data
        
        return jsonify({
            "path_data": path_data,
            "timestamp": datetime.now().isoformat(),
            "has_path": path_data is not None and path_data.get('has_path', False),
            "message": "Path calculated successfully" if path_data else "No path found"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stream')
def stream():
    """Server-Sent Events endpoint for real-time updates"""
    def generate():
        while True:
            try:
                with data_lock:
                    # Format as SSE
                    data = json.dumps(latest_data)
                    yield f"data: {data}\n\n"
                
                # Wait before next update
                time.sleep(8)  # Stream every 6 seconds
                
            except GeneratorExit:
                break
            except Exception as e:
                error_data = {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                time.sleep(5)  # Wait longer on error
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Cache-Control'
        }
    )

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "detector_status": "active" if detector else "inactive",
        "yolo_loaded": detector.yolo_model is not None,
        "background_detection": detection_thread is not None and detection_thread.is_alive(),
        "pathfinding_ready": True
    })

@app.route('/spots')
def spots():
    """Get parking spots configuration"""
    return jsonify({
        "spots": detector.get_spots_info(),
        "total_spots": len(detector.parking_slots),
        "grid_dimensions": {
            "rows": detector.ROWS,
            "cols": detector.COLS
        },
        "entrance": detector.pathfinder.entrance
    })

@app.route('/detection/start', methods=['POST'])
def start_detection():
    """Start real-time car detection"""
    try:
        start_background_detection()
        return jsonify({"message": "Real-time detection started", "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/detection/stop', methods=['POST'])
def stop_detection_endpoint():
    """Stop real-time car detection"""
    global stop_detection
    try:
        stop_detection = True
        return jsonify({"message": "Real-time detection stopped", "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/cctv')
def cctv_view():
    """Get current CCTV detection image"""
    with data_lock:
        return jsonify({
            "cctv_image": latest_data.get("cctv_image", ""),
            "timestamp": latest_data.get("timestamp", ""),
            "detection_confidence": latest_data.get("detection_confidence", 0.0)
        })

@app.route('/display/start', methods=['POST'])
def start_display():
    """Start image display window"""
    global display_enabled
    try:
        display_enabled = True
        return jsonify({"message": "Image display started", "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/display/stop', methods=['POST'])
def stop_display():
    """Stop image display window"""
    global display_enabled, display_window_active
    try:
        display_enabled = False
        if display_window_active:
            cv2.destroyAllWindows()
            display_window_active = False
        return jsonify({"message": "Image display stopped", "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/display/status')
def display_status():
    """Get display status"""
    return jsonify({
        "display_enabled": display_enabled,
        "window_active": display_window_active,
        "timestamp": datetime.now().isoformat()
    })

def initialize_app():
    """Initialize the application components"""
    print("Starting Parking Occupancy API with YOLO Detection & Pathfinding...")
    print("Loading YOLO model and parking slots...")
    
    # Initialize detector
    if detector.yolo_model is not None:
        print("✓ YOLO model loaded successfully")
    else:
        print("✗ YOLO model failed to load")
    
    print(f"✓ Loaded {len(detector.parking_slots)} parking slots")
    print(f"✓ Pathfinding initialized with entrance at {detector.pathfinder.entrance_points}")
    
    print("\nEndpoints available:")
    print("  - / (API info)")
    print("  - /snapshot (current status)")
    print("  - /stream (real-time SSE)")
    print("  - /health (health check)")
    print("  - /spots (spots configuration)")
    # ... (keep other endpoint prints if you want)
    
    # Disable display by default in production
    global display_enabled
    display_enabled = False
    
    # Start background detection automatically
    start_background_detection()

# Initialize the app when imported
initialize_app()

# This allows the app to be run directly during development
if __name__ == '__main__':
    # Get port from environment variable (for cloud platforms)
    import os
    port = int(os.environ.get('PORT', 5000))
    
    # Run with debug disabled for production
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"🚀 Starting ParkVision API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode, threaded=True)