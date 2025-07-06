# import cv2
# import mediapipe as mp
# import pandas as pd
# import math
# import numpy as np
# import time
# import joblib
# import os
# import json
# from datetime import datetime
# import warnings
# from flask import Flask, jsonify, request, Response
# import threading
# warnings.filterwarnings("ignore")
# app = Flask(__name__)
# font = cv2.FONT_HERSHEY_SIMPLEX
# font_scale = 1
# font_thickness = 2
# font_color = (0, 0, 0)  # Black color
# radius = 2
# color = (255, 255, 255)
# thickness = 2
# text_position = (50, 150)
# mp_drawing = mp.solutions.drawing_utils
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# mp_face_mesh = mp.solutions.face_mesh
# mp_face_detection = mp.solutions.face_detection
# point_list = [37, 0, 267, 84, 17, 314]
# relevant_coordinate_indice = [(0, 3), (1, 4), (2, 5), (0, 4), (1, 3)]
# distance_names = ['Distance Left', 'Distance Right', 'Distance Middle', 'Diagonal 1', 'Diagonal 2']
# distances = {name: [] for name in distance_names}
# JSON_FILE_PATH = 'lip_predictions_result.json'
# def face_crop(image):
#     height, width, channels = image.shape
#     with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
#         results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         if not results.detections:
#             return None
#         xmin = int(results.detections[0].location_data.relative_bounding_box.xmin * width)
#         ymin = int(results.detections[0].location_data.relative_bounding_box.ymin * height)
#         width = int(results.detections[0].location_data.relative_bounding_box.width * width)
#         height = int(results.detections[0].location_data.relative_bounding_box.height * height)
#         xmin, ymin = max(0, xmin), max(0, ymin)
#         width = min(width, image.shape[1] - xmin)
#         height = min(height, image.shape[0] - ymin)
#         crop_img = image[ymin:ymin + height, xmin:xmin + width]
#         crop_img = cv2.resize(crop_img, (300, 300), interpolation=cv2.INTER_NEAREST)
#         return crop_img
#
#
# def distance_calculator(pc, coordinates, width, height, distance_name):
#     a, b = coordinates
#     distance = math.sqrt(((pc[a][0] / width - pc[b][0] / width) ** 2 +
#                           ((pc[a][1] / height - pc[b][1] / height) ** 2)))
#     return distance
# def face_mesh_lip_detector(image):
#     with mp_face_mesh.FaceMesh(
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5
#     ) as face_mesh:
#
#         image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#         height, width, _ = image.shape
#         image.flags.writeable = False
#         results = face_mesh.process(image)
#
#         if not results.multi_face_landmarks:
#             return None, None
#
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         lm = results.multi_face_landmarks[0]
#
#         pc = []
#         for i in point_list:
#             x = int(lm.landmark[i].x * width)
#             y = int(lm.landmark[i].y * height)
#             pc.append((x, y))
#             cv2.circle(image, (x, y), radius, color, thickness)
#
#         distance_vector = []
#         for i in range(len(distance_names)):
#             dist = distance_calculator(
#                 pc,
#                 relevant_coordinate_indice[i],
#                 width,
#                 height,
#                 distance_names[i]
#             )
#             distance_vector.append(dist)
#
#         return image, distance_vector
#
#
# def lip_data_collector(image):
#     mod_image = face_crop(image)
#     if mod_image is None:
#         return None, None
#
#     try:
#         return face_mesh_lip_detector(mod_image)
#     except Exception as e:
#         print(f'Error: {e}')
#         return None, None
# def save_prediction_to_json(state, fps, avg_distance):
#     data = []
#     if os.path.exists(JSON_FILE_PATH):
#         try:
#             with open(JSON_FILE_PATH, 'r') as f:
#                 data = json.load(f)
#         except json.JSONDecodeError:
#             pass
#
#     data.append({
#         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "lip_state": state,
#         "FPS": fps,
#         "avg_distance": avg_distance
#     })
#     with open(JSON_FILE_PATH, 'w') as f:
#         json.dump(data, f, indent=4)
#
# def realtime_lip_classifier(model):
#     cap = cv2.VideoCapture(0)
#     prev_time = 0
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         new_time = time.time()
#         fps = int(1 / (new_time - prev_time))
#         prev_time = new_time
#
#         try:
#             mod_image, distance_vector = lip_data_collector(frame)
#
#             if mod_image is not None and distance_vector is not None:
#                 avg_distance = np.mean(distance_vector)
#                 state = "Open" if avg_distance > 0.135 else "Closed"
#
#                 save_prediction_to_json(state, fps, avg_distance)
#
#                 text = f'FPS: {fps} | Mouth: {state}'
#                 cv2.putText(mod_image, text, (25, 25),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#
#                 cv2.imshow("Lip State Detection", mod_image)
#
#         except Exception as e:
#             print(f"Error: {e}")
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
# @app.route('/stream', methods=['GET'])
# def stream_predictions():
#     def generate():
#         cap = cv2.VideoCapture(0)
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             try:
#                 _, distance_vector = lip_data_collector(frame)
#                 if distance_vector is not None:
#                     avg_distance = np.mean(distance_vector)
#                     state = "Open" if avg_distance > 0.135 else "Closed"
#
#                     data = {
#                         'lip_state': state,
#                         'avg_distance': float(avg_distance),
#                         'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#                     }
#                     yield f"data: {json.dumps(data)}\n\n"
#
#
#             except Exception as e:
#                 print(f"Error: {e}")
#
#         cap.release()
#
#     return Response(generate(), mimetype='text/event-stream')
#
#
# @app.route('/predict', methods=['POST'])
# def predict_from_image():
#     """Process a single image upload"""
#     if 'file' not in request.files:
#         return jsonify({"error": "No file provided"}), 400
#
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400
#
#     try:
#         image = cv2.imdecode(np.frombuffer(file.read(), cv2.IMREAD_COLOR))
#         _, distance_vector = lip_data_collector(image)
#
#         if distance_vector is None:
#             return jsonify({"error": "No face detected"}), 400
#
#         avg_distance = np.mean(distance_vector)
#         state = "Open" if avg_distance > 0.135 else "Closed"
#
#         return jsonify({
#             "lip_state": state,
#             "avg_distance": float(avg_distance),
#             "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         })
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
# # Load or train model
# MODEL_PATH = 'lip_movement_model.pkl'
# if os.path.exists(MODEL_PATH):
#     print("Loading saved model...")
#     model = joblib.load(MODEL_PATH)
# else:
#     print("Training model...")
#     model = None  # Placeholder - use your actual trained model
#
# JSON_FILE_PATH = 'lip_predictions_result.json'
#
# @app.route('/')
# def home():
#     return jsonify({
#         "status": "Lip Movement API",
#         "endpoints": {
#             "/health": "GET - Service health status",
#             "/lipresults": "GET - All lip movement results",
#             "/latest": "GET - Latest detection result",
#             "/stream": "GET - Real-time detection stream",
#             "/predict": "POST - Analyze uploaded image"
#         }
#     })
#
# @app.route('/lipresults', methods=['GET'])  # Changed from /libresults
# def get_all_results():
#     try:
#         with open(JSON_FILE_PATH, 'r') as f:
#             data = json.load(f)
#             return jsonify({
#                 "status": "success",
#                 "count": len(data),
#                 "results": data
#             })
#     except FileNotFoundError:
#         return jsonify({"status": "success", "count": 0, "results": []})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
# @app.route('/latest', methods=['GET'])
# def get_latest_result():
#     try:
#         with open(JSON_FILE_PATH, 'r') as f:
#             data = json.load(f)
#             return jsonify(data[-1])  # Return only the latest result
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
# @app.route('/health')
# def health():
#     """Health check endpoint"""
#     return jsonify({
#         "status": "healthy",
#         "service": "lip_movement",
#         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "model_loaded": os.path.exists(MODEL_PATH),
#         "json_file_exists": os.path.exists(JSON_FILE_PATH)
#     }), 200
# if __name__ == '__main__':
#     # Start real-time classifier in background thread
#     threading.Thread(
#         target=realtime_lip_classifier,
#         args=(model,),
#         daemon=True
#     ).start()
#     print(f"Current directory: {os.getcwd()}")
#     print(f"JSON file exists: {os.path.exists(JSON_FILE_PATH)}")
#     app.run(host='0.0.0.0', port=5002, debug=True)
#     # Run Flask app
#     print("🎥 Starting lip movement detection API...")
#

##______________________________________________________________

import cv2
import mediapipe as mp
import pandas as pd
import math
import numpy as np
import time
import joblib
import os
import json
from datetime import datetime
import warnings
from flask import Flask, jsonify, request, Response
import threading
import base64 # Import base64 for decoding image data

warnings.filterwarnings("ignore")
app = Flask(__name__)

# --- CORS Configuration for lip_movement service ---
@app.after_request
def add_cors_headers_to_lip_service(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
font_color = (0, 0, 0)  # Black color
radius = 2
color = (255, 255, 255)
thickness = 2
text_position = (50, 150)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
point_list = [37, 0, 267, 84, 17, 314]
relevant_coordinate_indice = [(0, 3), (1, 4), (2, 5), (0, 4), (1, 3)]
distance_names = ['Distance Left', 'Distance Right', 'Distance Middle', 'Diagonal 1', 'Diagonal 2']
distances = {name: [] for name in distance_names}

JSON_FILE_PATH = 'lip_predictions_result.json'
file_lock = threading.Lock() # Use a lock for JSON file operations


def face_crop(image):
    height, width, channels = image.shape
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return None
        xmin = int(results.detections[0].location_data.relative_bounding_box.xmin * width)
        ymin = int(results.detections[0].location_data.relative_bounding_box.ymin * height)
        width_bbox = int(results.detections[0].location_data.relative_bounding_box.width * width) # Renamed to avoid conflict
        height_bbox = int(results.detections[0].location_data.relative_bounding_box.height * height) # Renamed to avoid conflict
        xmin, ymin = max(0, xmin), max(0, ymin)
        # Ensure crop dimensions do not exceed image bounds
        width_bbox = min(width_bbox, image.shape[1] - xmin)
        height_bbox = min(height_bbox, image.shape[0] - ymin)
        crop_img = image[ymin:ymin + height_bbox, xmin:xmin + width_bbox]
        # Only resize if crop_img is not empty
        if crop_img.shape[0] > 0 and crop_img.shape[1] > 0:
            crop_img = cv2.resize(crop_img, (300, 300), interpolation=cv2.INTER_NEAREST)
        else:
            return None # Return None if crop resulted in empty image
        return crop_img


def distance_calculator(pc, coordinates, width, height, distance_name):
    a, b = coordinates
    # Ensure coordinates are within bounds before division
    if not (0 <= a < len(pc) and 0 <= b < len(pc)):
        raise IndexError(f"Coordinates {a}, {b} out of bounds for point_list length {len(pc)}")

    distance = math.sqrt(((pc[a][0] / width - pc[b][0] / width) ** 2 +
                          ((pc[a][1] / height - pc[b][1] / height) ** 2)))
    return distance

def face_mesh_lip_detector(image):
    with mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh:

        # Mediapipe expects RGB, input is BGR. It also expects non-flipped for analysis.
        # But for display, you might want it flipped. Here, we process as is.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        image_rgb.flags.writeable = False # Read-only for performance
        results = face_mesh.process(image_rgb)
        image_rgb.flags.writeable = True # Make writable again

        if not results.multi_face_landmarks:
            return None, None

        # Draw landmarks on the original image (or a copy if you want to keep original clean)
        # For a service, we don't necessarily need to draw, but keeping for debug/visualization potential.
        # image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) # Convert back to BGR if you want to draw on it

        lm = results.multi_face_landmarks[0] # Get landmarks for the first detected face

        pc = [] # Store pixel coordinates of relevant lip points
        # Only calculate and append points if they are within bounds and exist
        for i in point_list:
            if i < len(lm.landmark):
                x = int(lm.landmark[i].x * width)
                y = int(lm.landmark[i].y * height)
                pc.append((x, y))
                # If drawing, uncomment below
                # cv2.circle(image, (x, y), radius, color, thickness)
            else:
                print(f"Warning: Landmark index {i} out of bounds.")
                return None, None # Indicate failure if critical points are missing

        distance_vector = []
        for i in range(len(distance_names)):
            try:
                dist = distance_calculator(
                    pc,
                    relevant_coordinate_indice[i],
                    width,
                    height,
                    distance_names[i]
                )
                distance_vector.append(dist)
            except IndexError as e:
                print(f"Error calculating distance for {distance_names[i]}: {e}")
                return None, None # Return None if critical distance calculation fails

        return image, distance_vector # Return original image (potentially with drawings) and vector


def lip_data_collector(image):
    mod_image = face_crop(image) # Crop to face region first
    if mod_image is None:
        return None, None

    try:
        # Pass the cropped image to face_mesh_lip_detector
        return face_mesh_lip_detector(mod_image)
    except Exception as e:
        print(f'Error in lip_data_collector (face_mesh_lip_detector call): {e}')
        return None, None


def save_prediction_to_json(data_to_save):
    """
    Saves a single prediction entry to the JSON file.
    Assumes data_to_save is a dictionary ready to be appended.
    """
    with file_lock: # Protect file access
        all_data = []
        if os.path.exists(JSON_FILE_PATH):
            with open(JSON_FILE_PATH, 'r') as f:
                try:
                    all_data = json.load(f)
                except json.JSONDecodeError:
                    # Handle empty or corrupted JSON file
                    all_data = []

        all_data.append(data_to_save)

        with open(JSON_FILE_PATH, 'w') as f:
            json.dump(all_data, f, indent=4)
        # print(f"Saved prediction: {data_to_save['lip_state']}") # For debugging


# --- REMOVED: realtime_lip_classifier function as camera capture is moved to frontend ---
# This service will now receive frames via API

# Global variable to store the latest lip detection result
latest_lip_result = {}
latest_lip_result_lock = threading.Lock()


@app.route('/lippredict', methods=['POST']) # <--- New POST endpoint for receiving frames
def lip_predict():
    """
    Receives a base64 encoded frame from the gateway, processes it for lip movement,
    and returns the detection results.
    """
    global latest_lip_result

    data = request.get_json()
    image_data_b64 = data.get("frame")

    if not image_data_b64:
        return jsonify({"error": "No image data received"}), 400

    try:
        # Decode base64 image data (assuming it's a JPEG byte string)
        nparr = np.frombuffer(base64.b64decode(image_data_b64), np.uint8)
        frame_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Decode as color image

        if frame_np is None:
            return jsonify({"error": "Could not decode image"}), 400

        # Process the frame for lip detection
        # Note: mod_image is the cropped image, we don't need to return it, only distances
        mod_image, distance_vector = lip_data_collector(frame_np)

        if distance_vector is None:
            # No face detected or critical landmarks missing
            result = {
                "lip_state": "No Face Detected",
                "avg_distance": None,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "message": "Could not detect face or lip landmarks."
            }
            with latest_lip_result_lock:
                latest_lip_result = result
            # We might still want to log "no face detected" events, but not saving as 'Open' or 'Closed'
            # save_prediction_to_json(result) # Uncomment if you want to log these
            return jsonify(result)

        avg_distance = np.mean(distance_vector)
        state = "Open" if avg_distance > 0.135 else "Closed" # Use your threshold

        result = {
            "lip_state": state,
            "avg_distance": float(avg_distance), # Ensure float is serializable
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with latest_lip_result_lock:
            latest_lip_result = result # Store the latest result

        # Save to JSON only if a valid state is determined (i.e., face detected)
        save_prediction_to_json(result)

        return jsonify(result)

    except Exception as e:
        print(f"Error in /lippredict: {e}")
        return jsonify({"error": str(e)}), 500


# --- REMOVED: /stream endpoint logic as it's no longer capturing from local camera ---
# If frontend needs a direct stream from *this service*, it would need to
# send frames to *this service* which it then processes and streams back.
# For now, the gateway's /live_feed handles the consolidated stream.
# We will keep a simplified /stream route that returns the latest processed frame
# if you want to visualize what this service is seeing.

@app.route('/stream', methods=['GET'])
def stream_processed_frames():
    """Streams the last processed frame by this service (for debugging/visualization)."""
    def generate():
        while True:
            with latest_lip_result_lock:
                # Assuming `mod_image` (cropped image) from `lip_data_collector` could be stored
                # if you wanted to visualize the cropped face for lip detection.
                # For simplicity, let's just create a placeholder or use the last *received* frame if stored.
                # Currently, `lip_data_collector` returns `mod_image` but it's not stored globally.
                # To stream, you'd need to store `mod_image` globally.
                # For now, generate a placeholder.
                placeholder = np.zeros((300, 300, 3), dtype=np.uint8) # Default size for cropped face
                cv2.putText(placeholder, "Lip Service: No frame to stream", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                ret, buffer = cv2.imencode('.jpg', placeholder)
                frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.5) # Reduced frequency for placeholder
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Removed /predict, as the gateway will handle the primary image forwarding.
# If you need a direct file upload endpoint, keep it, but it's not part of the streaming flow.
# @app.route('/predict', methods=['POST'])
# def predict_from_image():
#     """Process a single image upload (kept for direct image testing)"""
#     # ... (original code) ...


# Load or train model (your model is used for the state classification based on avg_distance)
# The model loading here is for `lip_movement_model.pkl` which seems to be an unused placeholder
# in the current logic where state is determined by a fixed threshold (0.135).
# If you have a trained model that actually classifies based on features, you'd integrate it here.
MODEL_PATH = 'lip_movement_model.pkl'
model = None # Initialize model to None
if os.path.exists(MODEL_PATH):
    try:
        print("Loading saved lip movement model...")
        model = joblib.load(MODEL_PATH)
        print("Lip movement model loaded successfully.")
    except Exception as e:
        print(f"Error loading lip movement model: {e}")
        model = None # Ensure model is None if loading fails
else:
    print("Lip movement model file 'lip_movement_model.pkl' not found. Classification will use threshold.")


@app.route('/')
def home():
    return jsonify({
        "status": "Lip Movement API",
        "endpoints": {
            "/health": "GET - Service health status",
            "/lippredict": "POST - Submit a base64 encoded frame for lip movement detection.",
            "/lipresults": "GET - All saved lip movement results.",
            "/latest": "GET - Latest detection result.",
            "/stream": "GET - Stream last processed frame (placeholder/debug)."
        },
        "instructions": "Send base64 encoded JPEG frame in JSON body: {'frame': 'base64_string'} to /lippredict"
    })

@app.route('/lipresults', methods=['GET'])
def get_all_results():
    try:
        with file_lock: # Ensure thread-safe access
            if os.path.exists(JSON_FILE_PATH):
                with open(JSON_FILE_PATH, 'r') as f:
                    try:
                        data = json.load(f)
                        return jsonify({
                            "status": "success",
                            "count": len(data),
                            "results": data
                        })
                    except json.JSONDecodeError:
                        return jsonify({
                            "status": "error",
                            "message": "No valid data found in JSON file"
                        }), 500
            else:
                return jsonify({"status": "success", "count": 0, "results": []})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/latest', methods=['GET'])
def get_latest_result():
    with latest_lip_result_lock:
        if not latest_lip_result:
            return jsonify({
                "status": "error",
                "message": "No frame processed by this service yet"
            }), 404
        return jsonify({
            "status": "success",
            "prediction": latest_lip_result
        })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "lip_movement",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_loaded": os.path.exists(MODEL_PATH) and (model is not None), # Check if model file exists and is loaded
        "json_file_exists": os.path.exists(JSON_FILE_PATH)
    }), 200

if __name__ == '__main__':
    # Initialize results file if it doesn't exist
    if not os.path.exists(JSON_FILE_PATH):
        with open(JSON_FILE_PATH, 'w') as f:
            json.dump([], f)

    # Removed the realtime_lip_classifier thread start
    print(f"Current directory: {os.getcwd()}")
    print(f"JSON file exists: {os.path.exists(JSON_FILE_PATH)}")
    print("🎥 Lip Movement Detection API ready to receive frames on port 5002.")
    app.run(host='0.0.0.0', port=5002, debug=True, threaded=True) # threaded=True helps handle concurrent requests