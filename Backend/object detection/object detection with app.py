

##_------------------------------------------------------------------

# object_detection_service.py (Formerly object_detection_file.py, now running as a dedicated service)
from flask import Flask, request, jsonify, Response
import cv2
import numpy as np
import json
from datetime import datetime
from ultralytics import YOLO
import threading
import time
import os
import base64

app = Flask(__name__)

# @app.after_request (add CORS headers here too if needed, e.g., for direct frontend access if any)
def add_cors_headers_to_object_service(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response
app.after_request(add_cors_headers_to_object_service)


# Initialize models
try:
    coco_model = YOLO("yolov8s.pt")  # COCO model
    custom_model = YOLO("best.pt")  # Custom model
    print("YOLO models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# Configuration
RESULTS_FILE = "detection_results_object.json" # Distinct name
# CAMERA_WIDTH and CAMERA_HEIGHT are no longer relevant here as it's not capturing

# Global variables to store the latest processed frame and its detections (optional, for /stream or /objresults)
latest_processed_frame = None
latest_detections_object = {"objects": [], "timestamp": ""}
lock = threading.Lock()


def process_frame_for_objects(frame_np):
    """Process a given NumPy frame with both YOLO models"""
    try:
        detections = []
        # Ensure frame is contiguous and valid if coming from external source
        if not isinstance(frame_np, np.ndarray) or frame_np.size == 0:
            print("Received empty or invalid frame for processing.")
            return []

        # Convert to BGR if it's RGB from a web source (though OpenCV often expects BGR)
        # Assuming the input frame_np is already in the correct format (BGR for YOLO)
        # If frontend sends RGB, you might need: frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        coco_results = coco_model(frame_np, verbose=False, device='cpu')
        custom_results = custom_model(frame_np, verbose=False, device='cpu')

        for result in coco_results:
            for box in result.boxes:
                if int(box.cls) == 73:  # Book class (or other classes you want)
                    detections.append({
                        "model": "COCO",
                        "label": "Book",
                        "confidence": round(float(box.conf[0]), 2),
                        "bbox": list(map(int, box.xyxy[0].tolist()))
                    })

        for result in custom_results:
            for box in result.boxes:
                detections.append({
                    "model": "Custom",
                    "label": custom_model.names[int(box.cls)],
                    "confidence": round(float(box.conf[0]), 2),
                    "bbox": list(map(int, box.xyxy[0].tolist()))
                })

        return detections
    except Exception as e:
        print(f"Error in process_frame_for_objects: {e}")
        return []

def save_results(detections_data):
    """Save results to JSON file, skip empty detections"""
    try:
        # Check if detections are empty
        if not detections_data or len(detections_data.get('objects', [])) == 0:
            # print("No object detections to save.") # Comment out to reduce console spam if frequent empty frames
            return

        data = []
        if os.path.exists(RESULTS_FILE):
            try:
                with open(RESULTS_FILE, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                pass

        data.append(detections_data)

        with open(RESULTS_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        # print("Object detection results saved successfully.") # Comment out to reduce console spam
    except Exception as e:
        print(f"Error saving object results: {e}")

@app.route('/objpredict', methods=['POST'])
def obj_predict():
    """Receives a frame, processes it, and returns detections."""
    data = request.get_json()
    image_data_b64 = data.get("frame")

    if not image_data_b64:
        return jsonify({"error": "No image data received"}), 400

    try:
        # Decode base64 image data (assuming it's a JPEG byte string)
        nparr = np.frombuffer(base64.b64decode(image_data_b64), np.uint8)
        frame_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame_np is None:
            return jsonify({"error": "Could not decode image"}), 400

        detections = process_frame_for_objects(frame_np)

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with lock:
            global latest_processed_frame, latest_detections_object
            latest_processed_frame = frame_np.copy() # Store for potential /stream
            latest_detections_object = {
                "objects": detections,
                "timestamp": current_time
            }
        save_results(latest_detections_object)

        return jsonify(latest_detections_object)

    except Exception as e:
        print(f"Error in /objpredict: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for the object detection service"""
    return jsonify({
        "status": "running",
        "model_status": {
            "coco_model": "loaded" if coco_model else "failed",
            "custom_model": "loaded" if custom_model else "failed"
        }
    }), 200

# Optional: Keep /stream and /objresults for direct access if desired,
# but the main flow will be through the Gateway's /process_frame
@app.route('/stream')
def live_stream():
    """Live streaming with object detection from the last received frame"""
    def generate():
        while True:
            with lock:
                if latest_processed_frame is None:
                    # Create placeholder frame if no frame has been processed yet
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "No frames processed yet", (50, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    frame_to_stream = placeholder
                    detections_to_draw = {"objects": []}
                else:
                    frame_to_stream = latest_processed_frame.copy()
                    detections_to_draw = latest_detections_object.copy()

            # Draw detections on frame
            for detection in detections_to_draw.get("objects", []):
                x1, y1, x2, y2 = detection["bbox"]
                label = f"{detection['label']} {detection['confidence']:.2f}"
                color = (0, 255, 0) if detection["model"] == "COCO" else (0, 0, 255)
                cv2.rectangle(frame_to_stream, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_to_stream, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            ret, buffer = cv2.imencode('.jpg', frame_to_stream)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/objresults', methods=['GET'])
def get_obj_results():
    """Get all saved object detection results, excluding empty detections"""
    try:
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, 'r') as f:
                data = json.load(f)
            non_empty = [entry for entry in data if entry.get('objects')]
            if not non_empty:
                return jsonify({"message": "No valid (non-empty) object results found"})
            return jsonify(non_empty)
        return jsonify({"message": "No object results yet"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    if not os.path.exists("yolov8s.pt"):
        print("Error: yolov8s.pt model file not found")
        exit(1)
    if not os.path.exists("best.pt"):
        print("Error: best.pt model file not found")
        exit(1)

    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'w') as f:
            json.dump([], f)

    # This service no longer needs to start a camera capture thread.
    # It just waits for POST requests with frames.
    print("Object Detection Service ready to receive frames on port 5003.")
    app.run(host='0.0.0.0', port=5003, debug=False, threaded=True)