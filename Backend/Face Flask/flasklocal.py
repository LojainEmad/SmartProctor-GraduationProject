# # _________________________________________________________________
# # ##this is which i work with and very noce before add photo .
# import app
# from flask import Flask, jsonify, request
# from flask import Flask, request, jsonify, render_template
# import cv2
# import dlib
# import numpy as np
# import json
# import os
# import threading
# import time
# from scipy.spatial import distance
# import base64 # Import base64 for decoding image data
# from datetime import datetime # For timestamping results
# from werkzeug.utils import secure_filename
#
# # Define the shared JSON file
# SHARED_JSON_FILE = "Face_Results.json"
# file_lock = threading.Lock()
# UPLOAD_FOLDER = "static"
# REGISTERED_IMAGE = os.path.join(UPLOAD_FOLDER, "registered_face.jpg")
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#
# def append_to_shared_json(new_data):
#     with file_lock:
#         if os.path.exists(SHARED_JSON_FILE):
#             with open(SHARED_JSON_FILE, "r") as f:
#                 try:
#                     data = json.load(f)
#                 except json.JSONDecodeError:
#                     # If file is empty or corrupted, start with an empty list
#                     data = []
#         else:
#             data = []
#
#         data.append(new_data)
#         with open(SHARED_JSON_FILE, "w") as f:
#             json.dump(data, f, indent=4)
#
# app = Flask(__name__)
#
#
# # Route: Serve the register page
# @app.route('/')
# def serve_register_page():
#     return render_template("register.html")
#
# # Route: Handle form submit
# @app.route('/register', methods=['POST'])
# def register_user():
#     username = request.form.get('username')
#     file = request.files.get('profileImage')
#
#     if not username or not file:
#         return "Username and image required", 400
#     filename = secure_filename("registered_face.jpg")
#     file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#
#
#     return f'''
#     <h3>Registration successful!</h3>
#     <p>Saved as: {filename}</p>
#     <a href="/">Back to Register</a>
#     '''
#
# # --- CORS Configuration for flasklocal (add if needed, though gateway handles it) ---
# @app.after_request
# def add_cors_headers_to_face_service(response):
#     response.headers['Access-Control-Allow-Origin'] = '*'
#     response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
#     response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
#     return response
#
# # Initialize Dlib models (ensure 'shape_predictor_68_face_landmarks.dat' and 'dlib_face_recognition_resnet_model_v1.dat' are in the same directory)
# try:
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#     face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
#     print("Dlib models loaded successfully.")
# except Exception as e:
#     print(f"Error loading Dlib models: {e}")
#     # You might want to exit or handle this more gracefully depending on your application
#     exit(1)
#
#
# # Load known face for recognition (e.g., "Nardeen")
# photo_path = "nardeen.jpeg" # Make sure this image exists in the same directory
# known_faces = []
# known_names = []
#
# if os.path.exists(photo_path):
#     known_image = cv2.imread(photo_path)
#     if known_image is not None:
#         known_image_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
#         known_face_locations = detector(known_image_rgb, 1) # Use 1 for upsampling
#         if len(known_face_locations) > 0:
#             # Take the first detected face for encoding
#             known_encoding = np.array(
#                 face_rec_model.compute_face_descriptor(known_image_rgb, predictor(known_image_rgb, known_face_locations[0]))
#             )
#             known_faces.append(known_encoding)
#             known_names.append("Nardeen")
#             print(f"Known face '{known_names[0]}' loaded successfully.")
#         else:
#             print(f"No face detected in '{photo_path}'. Known face recognition will not work.")
#     else:
#         print(f"Could not load image from '{photo_path}'. Check file path and integrity.")
# else:
#     print(f"Known face image '{photo_path}' not found. Known face recognition will not work.")
#
# # Eye landmark indices for EAR calculation
# LEFT_EYE_IDX = [36, 37, 38, 39, 40, 41]
# RIGHT_EYE_IDX = [42, 43, 44, 45, 46, 47]
# BLINK_THRESHOLD = 0.25 # Adjust as needed
#
# # We no longer need `cap = cv2.VideoCapture(0)` as frames will be received via POST
#
# latest_result = {}
# result_lock = threading.Lock()
#
# # --- Utility function to calculate Eye Aspect Ratio (EAR) ---
# def eye_aspect_ratio(eye):
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     return (A + B) / (2.0 * C)
#
# # --- Core frame processing logic (now accepts a NumPy frame) ---
# def process_frame_for_face_detection(frame):
#     # Convert to grayscale for dlib's detector
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces_dlib = detector(gray) # Detect faces
#
#     frame_data = {"faces_detected": len(faces_dlib), "detections": []}
#
#     # Process each detected face
#     for face in faces_dlib:
#         x, y, w, h = face.left(), face.top(), face.width(), face.height()
#         landmarks = predictor(gray, face) # Get facial landmarks
#         landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()]) # Convert landmarks to NumPy array
#
#         # Face Recognition
#         name = "Unknown"
#         if known_faces: # Only attempt recognition if known faces are loaded
#             face_encoding = np.array(face_rec_model.compute_face_descriptor(frame, landmarks))
#             # Calculate distances to known faces
#             matches = np.linalg.norm(known_faces - face_encoding, axis=1) < 0.6 # Adjust threshold as needed
#             if any(matches):
#                 best_match_index = np.argmin(matches)
#                 name = known_names[best_match_index]
#
#         # Eye Blink Detection (EAR)
#         left_eye = landmarks_np[LEFT_EYE_IDX]
#         right_eye = landmarks_np[RIGHT_EYE_IDX]
#         left_ear = eye_aspect_ratio(left_eye)
#         right_ear = eye_aspect_ratio(right_eye)
#         avg_ear = (left_ear + right_ear) / 2.0
#         gaze_direction = "Center"
#         if avg_ear < BLINK_THRESHOLD:
#             gaze_direction = "Blink"
#
#         # Head Pose Estimation (simplified based on nose position relative to face center)
#         nose_tip = landmarks_np[30]
#         # Approximate face center for head pose
#         if len(landmarks_np) > 16: # Check if points 2 and 14 exist for a more stable center
#             left_cheek = landmarks_np[2]
#             right_cheek = landmarks_np[14]
#             face_center_x = (left_cheek[0] + right_cheek[0]) // 2
#         else: # Fallback if not enough landmarks or different model
#             face_center_x = x + w // 2 # Use bounding box center as fallback
#
#         head_position = "Center"
#         # These pixel thresholds (15) might need adjustment based on frame size and typical head movements
#         if nose_tip[0] > face_center_x + 15:
#             head_position = "Right"
#         elif nose_tip[0] < face_center_x - 15:
#             head_position = "Left"
#
#         # Construct detection result for the current face
#         detection = {
#             "face_position": {"x": x, "y": y, "width": w, "height": h},
#             "name": name,
#             "head_pose": head_position,
#             "eye_gaze": gaze_direction, # Renamed from "blink" to "eye_gaze" for broader meaning
#             "ear": round(avg_ear, 3), # Include EAR value for debugging/analysis
#
#             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Use current time
#         }
#         frame_data["detections"].append(detection)
#
#     return frame_data
#
# # The `capture_frames` thread is removed as frames are now POSTed to `/facepredict`
#
# @app.route('/')
# def home():
#     """Root endpoint that describes available API endpoints"""
#     return jsonify({
#         "status": "Face Detection API",
#         "endpoints": {
#             "/facepredict": "POST - Submit a base64 encoded frame for face detection and get results.",
#             "/latest": "GET - Get the latest detection result processed by this service.",
#             "/health": "GET - Health check endpoint for the face detection service."
#         },
#         "instructions": "Send base64 encoded JPEG frame in JSON body: {'frame': 'base64_string'}"
#     })
#
# @app.route('/facepredict', methods=['POST']) # <--- Changed to POST
# def face_predict():
#     """
#     Receives a base64 encoded frame, processes it for face detection,
#     and returns the results.
#     """
#     global latest_result # Update the global latest_result
#     data = request.get_json()
#     image_data_b64 = data.get("frame")
#
#     if not image_data_b64:
#         return jsonify({"error": "No image data received"}), 400
#
#     try:
#         # Decode base64 image data into a NumPy array
#         nparr = np.frombuffer(base64.b64decode(image_data_b64), np.uint8)
#         frame_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Decode as color image
#
#         if frame_np is None:
#             return jsonify({"error": "Could not decode image"}), 400
#
#         # Process the frame using the existing logic
#         result = process_frame_for_face_detection(frame_np)
#
#         with result_lock:
#             latest_result = result # Store the latest result
#
#         # Log results to JSON file based on your existing conditions
#         # (e.g., if no faces, too many faces, unknown face, or head pose/gaze issues)
#         if result["faces_detected"] == 0 or result["faces_detected"] > 1 or any(
#                 d["name"] == "Unknown" or d["head_pose"] != "Center" or d["eye_gaze"] != "Center"
#                 for d in result["detections"]):
#             append_to_shared_json(result)
#
#         return jsonify(result) # Return the processing result
#
#     except Exception as e:
#         print(f"Error in /facepredict: {e}")
#         return jsonify({"error": str(e)}), 500
#
#
# @app.route('/latest', methods=['GET'])
# def get_latest_prediction():
#     """Return the most recent prediction."""
#     with result_lock:
#         if not latest_result:
#             return jsonify({
#                 "status": "error",
#                 "message": "No frame processed by this service yet"
#             }), 404
#         return jsonify({
#             "status": "success",
#             "prediction": latest_result
#         })
#
# @app.route('/health', methods=['GET'])
# def health():
#     """Health check endpoint."""
#     return jsonify({"status": "Face model is running", "dlib_models_loaded": True}), 200
#
# if __name__ == '__main__':
#     # Ensure dlib model files exist
#     if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
#         print("Error: 'shape_predictor_68_face_landmarks.dat' not found. Download it and place it in the same directory.")
#         print("You can usually find it at: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 (unzip after download)")
#         exit(1)
#     if not os.path.exists("dlib_face_recognition_resnet_model_v1.dat"):
#         print("Error: 'dlib_face_recognition_resnet_model_v1.dat' not found. Download it and place it in the same directory.")
#         print("You can usually find it at: http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2 (unzip after download)")
#         exit(1)
#
#     # Initialize results file if it doesn't exist
#     if not os.path.exists(SHARED_JSON_FILE):
#         with open(SHARED_JSON_FILE, 'w') as f:
#             json.dump([], f)
#
#     print("Face Detection Service ready to receive frames on port 5001.")
#     # The `threaded=True` is generally handled by Flask's internal server for debug mode,
#     # but for production, a WSGI server like Gunicorn handles concurrency.
#     app.run(host='0.0.0.0', port=5001, debug=True) # debug=True is good for development

#################################

# import app
# from flask import Flask, request, jsonify, render_template
# import cv2
# import dlib
# import numpy as np
# import json
# import os
# import threading
# import time
# from scipy.spatial import distance
# import base64
# from datetime import datetime
# from werkzeug.utils import secure_filename
#
# # Constants
# SHARED_JSON_FILE = "Face_Results.json"
# UPLOAD_FOLDER = "static"
# REGISTERED_IMAGE = os.path.join(UPLOAD_FOLDER, "registered_face.jpg")
#
# # Flask app config
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#
# file_lock = threading.Lock()
#
# def append_to_shared_json(new_data):
#     with file_lock:
#         if os.path.exists(SHARED_JSON_FILE):
#             with open(SHARED_JSON_FILE, "r") as f:
#                 try:
#                     data = json.load(f)
#                 except json.JSONDecodeError:
#                     data = []
#         else:
#             data = []
#
#         data.append(new_data)
#         with open(SHARED_JSON_FILE, "w") as f:
#             json.dump(data, f, indent=4)
#
# @app.route('/')
# def serve_register_page():
#     return render_template("register.html")
# #
# # @app.route('/register', methods=['POST'])
# # def register_user():
# #     username = request.form.get('username')
# #     file = request.files.get('profileImage')
# #
# #     if not username or not file:
# #         return "Username and image required", 400
# #
# #     filename = secure_filename("registered_face.jpg")
# #     file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
# #
# #     return f'''
# #     <h3>Registration successful!</h3>
# #     <p>Saved as: {filename}</p>
# #     <a href="/">Back to Register</a>
# #     '''
#
# @app.route('/register', methods=['POST'])
# def register_user():
#     if 'profileImage' not in request.files:
#         return jsonify({"error": "No profile image part"}), 400
#
#     file = request.files['profileImage']
#     username = request.form.get('username')
#
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400
#
#     if file and username:
#         # Securely save the uploaded image
#         filename = secure_filename(file.filename)
#         # You might want to rename the file to something unique,
#         # e.g., using the username or a UUID, to avoid overwrites.
#         image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{username}_{filename}")
#         file.save(image_path)
#
#         # You might also want to save the username in a file if needed for dlib processing
#         # with open(REGISTERED_USERNAME_FILE, 'w') as f:
#         #     f.write(username)
#
#         return jsonify({"message": "Registration data and image received successfully!"}), 200
#     return jsonify({"error": "Invalid data"}), 400
#
# @app.after_request
# def add_cors_headers_to_face_service(response):
#     response.headers['Access-Control-Allow-Origin'] = '*'
#     response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
#     response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
#     return response
#
# # Load Dlib models
# try:
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#     face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
# except Exception as e:
#     print(f"Error loading Dlib models: {e}")
#     exit(1)
#
# LEFT_EYE_IDX = [36, 37, 38, 39, 40, 41]
# RIGHT_EYE_IDX = [42, 43, 44, 45, 46, 47]
# BLINK_THRESHOLD = 0.25
#
# latest_result = {}
# result_lock = threading.Lock()
#
# def eye_aspect_ratio(eye):
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     return (A + B) / (2.0 * C)
#
# def load_registered_face_encoding():
#     if os.path.exists(REGISTERED_IMAGE):
#         image = cv2.imread(REGISTERED_IMAGE)
#         if image is not None:
#             rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             faces = detector(rgb, 1)
#             if len(faces) > 0:
#                 encoding = np.array(face_rec_model.compute_face_descriptor(rgb, predictor(rgb, faces[0])))
#                 return encoding
#     return None
#
# def process_frame_for_face_detection(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces_dlib = detector(gray)
#
#     frame_data = {"faces_detected": len(faces_dlib), "detections": []}
#
#     registered_encoding = load_registered_face_encoding()
#     for face in faces_dlib:
#         x, y, w, h = face.left(), face.top(), face.width(), face.height()
#         landmarks = predictor(gray, face)
#         landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])
#
#         name = "Unknown"
#         if registered_encoding is not None:
#             face_encoding = np.array(face_rec_model.compute_face_descriptor(frame, landmarks))
#             distance_to_registered = np.linalg.norm(face_encoding - registered_encoding)
#             if distance_to_registered < 0.6:
#                 name = "RegisteredUser"
#
#         left_eye = landmarks_np[LEFT_EYE_IDX]
#         right_eye = landmarks_np[RIGHT_EYE_IDX]
#         left_ear = eye_aspect_ratio(left_eye)
#         right_ear = eye_aspect_ratio(right_eye)
#         avg_ear = (left_ear + right_ear) / 2.0
#
#         gaze_direction = "Center"
#         if avg_ear < BLINK_THRESHOLD:
#             gaze_direction = "Blink"
#
#         nose_tip = landmarks_np[30]
#         face_center_x = (landmarks_np[2][0] + landmarks_np[14][0]) // 2
#         head_position = "Center"
#         if nose_tip[0] > face_center_x + 15:
#             head_position = "Right"
#         elif nose_tip[0] < face_center_x - 15:
#             head_position = "Left"
#
#         detection = {
#             "face_position": {"x": x, "y": y, "width": w, "height": h},
#             "name": name,
#             "head_pose": head_position,
#             "eye_gaze": gaze_direction,
#             "ear": round(avg_ear, 3),
#             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }
#         frame_data["detections"].append(detection)
#
#     return frame_data
#
# @app.route('/facepredict', methods=['POST'])
# def face_predict():
#     global latest_result
#     data = request.get_json()
#     image_data_b64 = data.get("frame")
#
#     if not image_data_b64:
#         return jsonify({"error": "No image data received"}), 400
#
#     try:
#         nparr = np.frombuffer(base64.b64decode(image_data_b64), np.uint8)
#         frame_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         if frame_np is None:
#             return jsonify({"error": "Could not decode image"}), 400
#
#         result = process_frame_for_face_detection(frame_np)
#         with result_lock:
#             latest_result = result
#
#         if result["faces_detected"] == 0 or result["faces_detected"] > 1 or any(
#             d["name"] == "Unknown" or d["head_pose"] != "Center" or d["eye_gaze"] != "Center"
#             for d in result["detections"]
#         ):
#             append_to_shared_json(result)
#
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
# @app.route('/latest', methods=['GET'])
# def get_latest_prediction():
#     with result_lock:
#         if not latest_result:
#             return jsonify({"status": "error", "message": "No frame processed yet"}), 404
#         return jsonify({"status": "success", "prediction": latest_result})
#
# @app.route('/health', methods=['GET'])
# def health():
#     return jsonify({"status": "Face model is running", "dlib_models_loaded": True}), 200
#
# if __name__ == '__main__':
#     if not os.path.exists("shape_predictor_68_face_landmarks.dat") or not os.path.exists("dlib_face_recognition_resnet_model_v1.dat"):
#         print("Missing dlib model files.")
#         exit(1)
#     if not os.path.exists(SHARED_JSON_FILE):
#         with open(SHARED_JSON_FILE, 'w') as f:
#             json.dump([], f)
#
#     print("Face Detection Service running on port 5001.")
#     app.run(host='0.0.0.0', port=5001, debug=True)
#
#####################################################
# import app
# from flask import Flask, request, jsonify, render_template, redirect, url_for
# import cv2
# import dlib
# import numpy as np
# import json
# import os
# import threading
# import time
# from scipy.spatial import distance
# import base64
# from datetime import datetime
# from werkzeug.utils import secure_filename
#
# # Constants
# SHARED_JSON_FILE = "Face_Results.json"
# UPLOAD_FOLDER = "static"
# REGISTERED_IMAGE = os.path.join(UPLOAD_FOLDER, "registered_face.jpg")
#
# # Flask app config
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#
# file_lock = threading.Lock()
#
# def append_to_shared_json(new_data):
#     with file_lock:
#         if os.path.exists(SHARED_JSON_FILE):
#             with open(SHARED_JSON_FILE, "r") as f:
#                 try:
#                     data = json.load(f)
#                 except json.JSONDecodeError:
#                     data = []
#         else:
#             data = []
#
#         data.append(new_data)
#         with open(SHARED_JSON_FILE, "w") as f:
#             json.dump(data, f, indent=4)
#
# @app.route('/')
# def serve_register_page():
#     return render_template("register.html")
#
# @app.route('/register', methods=['POST'])
# def register_user():
#     username = request.form.get('username')
#     file = request.files.get('profileImage') # Use .get() to handle cases where profileImage is not sent
#
#     if not username:
#         return jsonify({"error": "Username required"}), 400
#
#     # Only attempt to save file if it exists and has a filename
#     if file and file.filename != '':
#         try:
#             # Save the uploaded image directly as REGISTERED_IMAGE
#             # This will overwrite any previously registered face for the single-face recognition system.
#             file.save(REGISTERED_IMAGE)
#             print(f"Registered face saved to: {REGISTERED_IMAGE}")
#         except Exception as e:
#             return jsonify({"error": f"Failed to save registered image: {e}"}), 500
#     else:
#         # This block will be hit if no file is uploaded (e.g., instructor registration)
#         # or if an empty file input was submitted.
#         print("No profile image provided or selected.")
#         # If your system supports multiple users, you'd integrate database logic here
#         # to save user details and potentially a reference to their unique image.
#
#     return jsonify({"message": "Registration data and image (if provided) received successfully!"}), 200
#
# @app.after_request
# def add_cors_headers_to_face_service(response):
#     response.headers['Access-Control-Allow-Origin'] = '*'
#     response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
#     response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
#     return response
#
# # Load Dlib models
# try:
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#     face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
# except Exception as e:
#     print(f"Error loading Dlib models: {e}")
#     exit(1)
#
# LEFT_EYE_IDX = [36, 37, 38, 39, 40, 41]
# RIGHT_EYE_IDX = [42, 43, 44, 45, 46, 47]
# BLINK_THRESHOLD = 0.25
#
# latest_result = {}
# result_lock = threading.Lock()
#
# def eye_aspect_ratio(eye):
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     return (A + B) / (2.0 * C)
#
# def load_registered_face_encoding():
#     if os.path.exists(REGISTERED_IMAGE):
#         image = cv2.imread(REGISTERED_IMAGE)
#         if image is not None:
#             rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             faces = detector(rgb, 1)
#             if len(faces) > 0:
#                 encoding = np.array(face_rec_model.compute_face_descriptor(rgb, predictor(rgb, faces[0])))
#                 return encoding
#     return None
#
# def process_frame_for_face_detection(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces_dlib = detector(gray)
#
#     frame_data = {"faces_detected": len(faces_dlib), "detections": []}
#
#     registered_encoding = load_registered_face_encoding()
#     for face in faces_dlib:
#         x, y, w, h = face.left(), face.top(), face.width(), face.height()
#         landmarks = predictor(gray, face)
#         landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])
#
#         name = "Unknown"
#         if registered_encoding is not None:
#             face_encoding = np.array(face_rec_model.compute_face_descriptor(frame, landmarks))
#             distance_to_registered = np.linalg.norm(face_encoding - registered_encoding)
#             if distance_to_registered < 0.6:
#                 name = "RegisteredUser"
#
#         left_eye = landmarks_np[LEFT_EYE_IDX]
#         right_eye = landmarks_np[RIGHT_EYE_IDX]
#         left_ear = eye_aspect_ratio(left_eye)
#         right_ear = eye_aspect_ratio(right_eye)
#         avg_ear = (left_ear + right_ear) / 2.0
#
#         gaze_direction = "Center"
#         if avg_ear < BLINK_THRESHOLD:
#             gaze_direction = "Blink"
#
#         nose_tip = landmarks_np[30]
#         face_center_x = (landmarks_np[2][0] + landmarks_np[14][0]) // 2
#         head_position = "Center"
#         if nose_tip[0] > face_center_x + 15:
#             head_position = "Right"
#         elif nose_tip[0] < face_center_x - 15:
#             head_position = "Left"
#
#         detection = {
#             "face_position": {"x": x, "y": y, "width": w, "height": h},
#             "name": name,
#             "head_pose": head_position,
#             "eye_gaze": gaze_direction,
#             "ear": round(avg_ear, 3),
#             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }
#         frame_data["detections"].append(detection)
#
#     return frame_data
#
# @app.route('/facepredict', methods=['POST'])
# def face_predict():
#     global latest_result
#     data = request.get_json()
#     image_data_b64 = data.get("frame")
#
#     if not image_data_b64:
#         return jsonify({"error": "No image data received"}), 400
#
#     try:
#         nparr = np.frombuffer(base64.b64decode(image_data_b64), np.uint8)
#         frame_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         if frame_np is None:
#             return jsonify({"error": "Could not decode image"}), 400
#
#         result = process_frame_for_face_detection(frame_np)
#         with result_lock:
#             latest_result = result
#
#         if result["faces_detected"] == 0 or result["faces_detected"] > 1 or any(
#             d["name"] == "Unknown" or d["head_pose"] != "Center" or d["eye_gaze"] != "Center"
#             for d in result["detections"]
#         ):
#             append_to_shared_json(result)
#
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
# @app.route('/latest', methods=['GET'])
# def get_latest_prediction():
#     with result_lock:
#         if not latest_result:
#             return jsonify({"status": "error", "message": "No frame processed yet"}), 404
#         return jsonify({"status": "success", "prediction": latest_result})
#
# @app.route('/health', methods=['GET'])
# def health():
#     return jsonify({"status": "Face model is running", "dlib_models_loaded": True}), 200
#
# if __name__ == '__main__':
#     # Ensure UPLOAD_FOLDER exists
#     if not os.path.exists(UPLOAD_FOLDER):
#         os.makedirs(UPLOAD_FOLDER)
#
#     if not os.path.exists("shape_predictor_68_face_landmarks.dat") or not os.path.exists("dlib_face_recognition_resnet_model_v1.dat"):
#         print("Missing dlib model files. Please download them and place them in the same directory as flasklocal.py.")
#         print("You can usually find them at:")
#         print(" - http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 (unzip after download)")
#         print(" - http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2 (unzip after download)")
#         exit(1)
#     if not os.path.exists(SHARED_JSON_FILE):
#         with open(SHARED_JSON_FILE, 'w') as f:
#             json.dump([], f)
#
#     print("Face Detection Service running on port 5001.")
#     app.run(host='0.0.0.0', port=5001, debug=True)

############################################################3

# import app
# from flask import Flask, request, jsonify, render_template, redirect, url_for
# import cv2
# import dlib
# import numpy as np
# import json
# import os
# import threading
# import time
# from scipy.spatial import distance
# import base64
# from datetime import datetime
# from werkzeug.utils import secure_filename
#
# # Constants
# SHARED_JSON_FILE = "Face_Results.json"
# UPLOAD_FOLDER = "static"
# REGISTERED_IMAGE = os.path.join(UPLOAD_FOLDER, "registered_face.jpg")
#
# # Flask app config
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#
# file_lock = threading.Lock()
#
# def append_to_shared_json(new_data):
#     with file_lock:
#         if os.path.exists(SHARED_JSON_FILE):
#             with open(SHARED_JSON_FILE, "r") as f:
#                 try:
#                     data = json.load(f)
#                 except json.JSONDecodeError:
#                     data = []
#         else:
#             data = []
#
#         data.append(new_data)
#         with open(SHARED_JSON_FILE, "w") as f:
#             json.dump(data, f, indent=4)
#
# @app.route('/')
# def serve_register_page():
#     return render_template("register.html")
#
# @app.route('/register', methods=['POST'])
# def register_user():
#     username = request.form.get('username')
#     file = request.files.get('profileImage') # Use .get() to handle cases where profileImage is not sent
#
#     if not username:
#         return jsonify({"error": "Username required"}), 400
#
#     # Only attempt to save file if it exists and has a filename
#     if file and file.filename != '':
#         try:
#             # Save the uploaded image directly as REGISTERED_IMAGE
#             # This will overwrite any previously registered face for the single-face recognition system.
#             file.save(REGISTERED_IMAGE)
#             print(f"Registered face saved to: {REGISTERED_IMAGE}")
#         except Exception as e:
#             return jsonify({"error": f"Failed to save registered image: {e}"}), 500
#     else:
#         # This block will be hit if no file is uploaded (e.g., instructor registration)
#         # or if an empty file input was submitted.
#         print("No profile image provided or selected.")
#         # If your system supports multiple users, you'd integrate database logic here
#         # to save user details and potentially a reference to their unique image.
#
#     return jsonify({"message": "Registration data and image (if provided) received successfully!"}), 200
#
# @app.after_request
# def add_cors_headers_to_face_service(response):
#     response.headers['Access-Control-Allow-Origin'] = '*'
#     response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
#     response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
#     return response
#
# # Load Dlib models
# try:
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#     face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
# except Exception as e:
#     print(f"Error loading Dlib models: {e}")
#     exit(1)
#
# LEFT_EYE_IDX = [36, 37, 38, 39, 40, 41]
# RIGHT_EYE_IDX = [42, 43, 44, 45, 46, 47]
# BLINK_THRESHOLD = 0.25
#
# latest_result = {}
# result_lock = threading.Lock()
#
# def eye_aspect_ratio(eye):
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     return (A + B) / (2.0 * C)
#
# def load_registered_face_encoding():
#     if os.path.exists(REGISTERED_IMAGE):
#         image = cv2.imread(REGISTERED_IMAGE)
#         if image is not None:
#             rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             faces = detector(rgb, 1)
#             if len(faces) > 0:
#                 encoding = np.array(face_rec_model.compute_face_descriptor(rgb, predictor(rgb, faces[0])))
#                 return encoding
#     return None
#
# def process_frame_for_face_detection(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces_dlib = detector(gray)
#
#     frame_data = {"faces_detected": len(faces_dlib), "detections": []}
#
#     registered_encoding = load_registered_face_encoding()
#     for face in faces_dlib:
#         x, y, w, h = face.left(), face.top(), face.width(), face.height()
#         landmarks = predictor(gray, face)
#         landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])
#
#         name = "Unknown"
#         if registered_encoding is not None:
#             face_encoding = np.array(face_rec_model.compute_face_descriptor(frame, landmarks))
#             distance_to_registered = np.linalg.norm(face_encoding - registered_encoding)
#             if distance_to_registered < 0.6:
#                 name = "RegisteredUser"
#
#         left_eye = landmarks_np[LEFT_EYE_IDX]
#         right_eye = landmarks_np[RIGHT_EYE_IDX]
#         left_ear = eye_aspect_ratio(left_eye)
#         right_ear = eye_aspect_ratio(right_eye)
#         avg_ear = (left_ear + right_ear) / 2.0
#
#         gaze_direction = "Center"
#         if avg_ear < BLINK_THRESHOLD:
#             gaze_direction = "Blink"
#
#         nose_tip = landmarks_np[30]
#         face_center_x = (landmarks_np[2][0] + landmarks_np[14][0]) // 2
#         head_position = "Center"
#         if nose_tip[0] > face_center_x + 15:
#             head_position = "Right"
#         elif nose_tip[0] < face_center_x - 15:
#             head_position = "Left"
#
#         detection = {
#             "face_position": {"x": x, "y": y, "width": w, "height": h},
#             "name": name,
#             "head_pose": head_position,
#             "eye_gaze": gaze_direction,
#             "ear": round(avg_ear, 3),
#             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }
#         frame_data["detections"].append(detection)
#
#     return frame_data
#
# @app.route('/facepredict', methods=['POST'])
# def face_predict():
#     global latest_result
#     data = request.get_json()
#     image_data_b64 = data.get("frame")
#
#     if not image_data_b64:
#         return jsonify({"error": "No image data received"}), 400
#
#     try:
#         nparr = np.frombuffer(base64.b64decode(image_data_b64), np.uint8)
#         frame_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         if frame_np is None:
#             return jsonify({"error": "Could not decode image"}), 400
#
#         result = process_frame_for_face_detection(frame_np)
#         with result_lock:
#             latest_result = result
#
#         if result["faces_detected"] == 0 or result["faces_detected"] > 1 or any(
#             d["name"] == "Unknown" or d["head_pose"] != "Center" or d["eye_gaze"] != "Center"
#             for d in result["detections"]
#         ):
#             append_to_shared_json(result)
#
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
# @app.route('/latest', methods=['GET'])
# def get_latest_prediction():
#     with result_lock:
#         if not latest_result:
#             return jsonify({"status": "error", "message": "No frame processed yet"}), 404
#         return jsonify({"status": "success", "prediction": latest_result})
#
# @app.route('/health', methods=['GET'])
# def health():
#     return jsonify({"status": "Face model is running", "dlib_models_loaded": True}), 200
#
# if __name__ == '__main__':
#     # Ensure UPLOAD_FOLDER exists
#     if not os.path.exists(UPLOAD_FOLDER):
#         os.makedirs(UPLOAD_FOLDER)
#
#     if not os.path.exists("shape_predictor_68_face_landmarks.dat") or not os.path.exists("dlib_face_recognition_resnet_model_v1.dat"):
#         print("Missing dlib model files. Please download them and place them in the same directory as flasklocal.py.")
#         print("You can usually find them at:")
#         print(" - http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 (unzip after download)")
#         print(" - http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2 (unzip after download)")
#         exit(1)
#     if not os.path.exists(SHARED_JSON_FILE):
#         with open(SHARED_JSON_FILE, 'w') as f:
#             json.dump([], f)
#
#     print("Face Detection Service running on port 5001.")
#     app.run(host='0.0.0.0', port=5001, debug=True)

##############################################################3
#
# ##sh8al
#
# import app
# from flask import Flask, request, jsonify, render_template, redirect, url_for
# import cv2
# import dlib
# import numpy as np
# import json
# import os
# import threading
# import time
# from scipy.spatial import distance
# import base64
# from datetime import datetime
# from werkzeug.utils import secure_filename
#
# # Constants
# SHARED_JSON_FILE = "Face_Results.json"
# UPLOAD_FOLDER = "static"
# REGISTERED_IMAGE = os.path.join(UPLOAD_FOLDER, "registered_face.jpg")
#
# # Flask app config
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#
# file_lock = threading.Lock()
#
#
# def append_to_shared_json(new_data):
#     with file_lock:
#         if os.path.exists(SHARED_JSON_FILE):
#             with open(SHARED_JSON_FILE, "r") as f:
#                 try:
#                     data = json.load(f)
#                 except json.JSONDecodeError:
#                     # If file is corrupted, re-initialize it as an empty list
#                     print(f"Warning: {SHARED_JSON_FILE} is corrupted. Re-initializing.")
#                     data = []
#         else:
#             data = []
#
#         data.append(new_data)
#         with open(SHARED_JSON_FILE, "w") as f:
#             json.dump(data, f, indent=4)
#
#
# @app.route('/')
# def serve_register_page():
#     return render_template("register.html")
#
#
# @app.route('/register', methods=['POST'])
# def register_user():
#     username = request.form.get('username')
#     file = request.files.get('profileImage')  # Use .get() to handle cases where profileImage is not sent
#
#     if not username:
#         return jsonify({"error": "Username required"}), 400
#
#     # Only attempt to save file if it exists and has a filename
#     if file and file.filename != '':
#         try:
#             # Save the uploaded image directly as REGISTERED_IMAGE
#             # This will overwrite any previously registered face for the single-face recognition system.
#             file.save(REGISTERED_IMAGE)
#             print(f"Registered face saved to: {REGISTERED_IMAGE}")
#         except Exception as e:
#             return jsonify({"error": f"Failed to save registered image: {e}"}), 500
#     else:
#         # This block will be hit if no file is uploaded (e.g., instructor registration)
#         # or if an empty file input was submitted.
#         print("No profile image provided or selected.")
#         # If your system supports multiple users, you'd integrate database logic here
#         # to save user details and potentially a reference to their unique image.
#
#     return jsonify({"message": "Registration data and image (if provided) received successfully!"}), 200
#
#
# @app.after_request
# def add_cors_headers_to_face_service(response):
#     response.headers['Access-Control-Allow-Origin'] = '*'
#     response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
#     response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
#     return response
#
#
# # Load Dlib models
# try:
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#     face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
# except Exception as e:
#     print(f"Error loading Dlib models: {e}")
#     # Exit if models fail to load, as the core functionality won't work
#     exit(1)
#
# LEFT_EYE_IDX = [36, 37, 38, 39, 40, 41]
# RIGHT_EYE_IDX = [42, 43, 44, 45, 46, 47]
# BLINK_THRESHOLD = 0.25  # Typically for blinks, EAR goes below this
#
# latest_result = {}
# result_lock = threading.Lock()
#
#
# def eye_aspect_ratio(eye):
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     return (A + B) / (2.0 * C)
#
#
# def load_registered_face_encoding():
#     if os.path.exists(REGISTERED_IMAGE):
#         image = cv2.imread(REGISTERED_IMAGE)
#         if image is not None:
#             rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             faces = detector(rgb, 1)
#             if len(faces) > 0:
#                 # Use the first detected face for encoding
#                 encoding = np.array(face_rec_model.compute_face_descriptor(rgb, predictor(rgb, faces[0])))
#                 return encoding
#     return None
#
#
# def process_frame_for_face_detection(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces_dlib = detector(gray)
#
#     frame_data = {"faces_detected": len(faces_dlib), "detections": []}
#
#     registered_encoding = load_registered_face_encoding()
#     for face in faces_dlib:
#         x, y, w, h = face.left(), face.top(), face.width(), face.height()
#         landmarks = predictor(gray, face)
#         landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])
#
#         name = "Unknown"
#         if registered_encoding is not None:
#             face_encoding = np.array(face_rec_model.compute_face_descriptor(frame, landmarks))
#             # Lower the threshold slightly if needed, or keep 0.6 for common use
#             if np.linalg.norm(face_encoding - registered_encoding) < 0.6:
#                 name = "RegisteredUser"
#
#         left_eye = landmarks_np[LEFT_EYE_IDX]
#         right_eye = landmarks_np[RIGHT_EYE_IDX]
#         left_ear = eye_aspect_ratio(left_eye)
#         right_ear = eye_aspect_ratio(right_eye)
#         avg_ear = (left_ear + right_ear) / 2.0
#
#         # Simple gaze and head pose estimation
#         gaze_direction = "Center"
#         if avg_ear < BLINK_THRESHOLD:
#             gaze_direction = "Blink"
#         # More sophisticated gaze/head pose would require deeper analysis of landmarks or 3D pose estimation
#         # For a basic check:
#         if landmarks_np[42][0] > landmarks_np[39][0] + 5:  # Right eye outer corner > Left eye inner corner + buffer
#             gaze_direction = "Left"  # Looking towards their left (our right)
#         elif landmarks_np[39][0] < landmarks_np[42][0] - 5:  # Left eye inner corner < Right eye outer corner - buffer
#             gaze_direction = "Right"  # Looking towards their right (our left)
#
#         head_position = "Center"
#         # Using nose tip (30) relative to left (1) and right (15) jaw points for simple horizontal pose
#         if landmarks_np[30][0] > landmarks_np[15][0]:  # Nose tip beyond right jaw
#             head_position = "Right"
#         elif landmarks_np[30][0] < landmarks_np[1][0]:  # Nose tip beyond left jaw
#             head_position = "Left"
#         # Using nose tip (30) relative to top (27) and chin (8) for simple vertical pose
#         if landmarks_np[30][1] > landmarks_np[8][1] - 10:  # Nose tip too close to chin
#             head_position = "Down"
#         elif landmarks_np[30][1] < landmarks_np[27][1] + 10:  # Nose tip too close to eyebrow center
#             head_position = "Up"
#
#         # Combine head_position for more descriptive states
#         # (This is a simplification; a full head pose estimation would use solvePnP)
#         if head_position == "Right" and gaze_direction == "Right":
#             head_position = "Turned Right"
#         elif head_position == "Left" and gaze_direction == "Left":
#             head_position = "Turned Left"
#
#         detection = {
#             "face_position": {"x": x, "y": y, "width": w, "height": h},
#             "name": name,
#             "head_pose": head_position,
#             "eye_gaze": gaze_direction,
#             "ear": round(avg_ear, 3),  # Eye Aspect Ratio
#             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }
#         frame_data["detections"].append(detection)
#
#     return frame_data
#
#
# @app.route('/exam_question_details', methods=['POST'])
# def face_predict():
#     global latest_result
#     data = request.get_json()
#     image_data_b64 = data.get("frame")
#
#     if not image_data_b64:
#         return jsonify({"error": "No image data received"}), 400
#
#     try:
#         nparr = np.frombuffer(base64.b64decode(image_data_b64), np.uint8)
#         frame_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         if frame_np is None:
#             return jsonify({"error": "Could not decode image"}), 400
#
#         result = process_frame_for_face_detection(frame_np)
#         with result_lock:
#             latest_result = result
#
#         # --- UPDATED LOGGING LOGIC ---
#         # Log all frames that are processed and have at least one face detected.
#         # This addresses the "not complete protecting" issue where only suspicious frames were logged.
#         if result["faces_detected"] > 0:
#             append_to_shared_json(result)
#         # If you want to log ALL frames (even those with no faces), uncomment the line below:
#         # append_to_shared_json(result)
#         # If you want to revert to only logging suspicious activity, revert to the original if condition:
#         # if result["faces_detected"] == 0 or result["faces_detected"] > 1 or any(
#         #     d["name"] == "Unknown" or d["head_pose"] != "Center" or d["eye_gaze"] != "Center"
#         #     for d in result["detections"]
#         # ):
#         #     append_to_shared_json(result)
#         # --- END UPDATED LOGGING LOGIC ---
#
#         return jsonify(result)
#     except Exception as e:
#         print(f"Error in face_predict: {e}")  # Log the error for debugging
#         return jsonify({"error": str(e)}), 500
#
#
# @app.route('/latest', methods=['GET'])
# def get_latest_prediction():
#     with result_lock:
#         if not latest_result:
#             return jsonify({"status": "error", "message": "No frame processed yet"}), 404
#         return jsonify({"status": "success", "prediction": latest_result})
#
#
# @app.route('/health', methods=['GET'])
# def health():
#     # Attempt to load models briefly to ensure they are accessible
#     try:
#         dlib.get_frontal_face_detector()
#         dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#         dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
#         models_loaded_check = True
#     except Exception:
#         models_loaded_check = False
#     return jsonify({"status": "Face model service is running", "dlib_models_loaded": models_loaded_check}), 200
#
#
# if __name__ == '__main__':
#     # Ensure UPLOAD_FOLDER exists
#     if not os.path.exists(UPLOAD_FOLDER):
#         os.makedirs(UPLOAD_FOLDER)
#
#     # Critical check for Dlib model files
#     if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
#         print("Error: Missing 'shape_predictor_68_face_landmarks.dat'. Please download it.")
#         print("Download link: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 (unzip after download)")
#         exit(1)
#     if not os.path.exists("dlib_face_recognition_resnet_model_v1.dat"):
#         print("Error: Missing 'dlib_face_recognition_resnet_model_v1.dat'. Please download it.")
#         print(
#             "Download link: http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2 (unzip after download)")
#         exit(1)
#
#     # Initialize Face_Results.json if it doesn't exist or is corrupted
#     with file_lock:  # Ensure thread-safe initialization
#         if not os.path.exists(SHARED_JSON_FILE):
#             with open(SHARED_JSON_FILE, 'w') as f:
#                 json.dump([], f)
#         else:
#             with open(SHARED_JSON_FILE, 'r+') as f:
#                 content = f.read()
#                 if not content.strip():
#                     f.seek(0)
#                     json.dump([], f)
#                 else:
#                     try:
#                         json.loads(content)
#                     except json.JSONDecodeError:
#                         print(f"Warning: {SHARED_JSON_FILE} is corrupted. Re-initializing.")
#                         f.seek(0)
#                         f.truncate()
#                         json.dump([], f)
#
#     print("Face Detection Service running on port 5001.")
#     # Set debug=False for production environments to avoid auto-reloading and for better performance
#     app.run(host='0.0.0.0', port=5001, debug=True)

#############################################################333


##_________________________________________________________________

from flask import Flask, jsonify, request
import cv2
import dlib
import numpy as np
import json
import os
import threading
import time
from scipy.spatial import distance
import base64 # Import base64 for decoding image data
from datetime import datetime # For timestamping results

# Define the shared JSON file
SHARED_JSON_FILE = "Face_Results.json"
file_lock = threading.Lock()

def append_to_shared_json(new_data):
    with file_lock:
        if os.path.exists(SHARED_JSON_FILE):
            with open(SHARED_JSON_FILE, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    # If file is empty or corrupted, start with an empty list
                    data = []
        else:
            data = []

        data.append(new_data)
        with open(SHARED_JSON_FILE, "w") as f:
            json.dump(data, f, indent=4)

app = Flask(__name__)

# --- CORS Configuration for flasklocal (add if needed, though gateway handles it) ---
@app.after_request
def add_cors_headers_to_face_service(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response

# Initialize Dlib models (ensure 'shape_predictor_68_face_landmarks.dat' and 'dlib_face_recognition_resnet_model_v1.dat' are in the same directory)
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    print("Dlib models loaded successfully.")
except Exception as e:
    print(f"Error loading Dlib models: {e}")
    # You might want to exit or handle this more gracefully depending on your application
    exit(1)


# Load known face for recognition (e.g., "Nardeen")
photo_path = "Lojain .jpg" # Make sure this image exists in the same directory
known_faces = []
known_names = []

if os.path.exists(photo_path):
    known_image = cv2.imread(photo_path)
    if known_image is not None:
        known_image_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
        known_face_locations = detector(known_image_rgb, 1) # Use 1 for upsampling
        if len(known_face_locations) > 0:
            # Take the first detected face for encoding
            known_encoding = np.array(
                face_rec_model.compute_face_descriptor(known_image_rgb, predictor(known_image_rgb, known_face_locations[0]))
            )
            known_faces.append(known_encoding)
            known_names.append("Lojain")
            print(f"Known face '{known_names[0]}' loaded successfully.")
        else:
            print(f"No face detected in '{photo_path}'. Known face recognition will not work.")
    else:
        print(f"Could not load image from '{photo_path}'. Check file path and integrity.")
else:
    print(f"Known face image '{photo_path}' not found. Known face recognition will not work.")

# Eye landmark indices for EAR calculation
LEFT_EYE_IDX = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_IDX = [42, 43, 44, 45, 46, 47]
BLINK_THRESHOLD = 0.25 # Adjust as needed

# We no longer need `cap = cv2.VideoCapture(0)` as frames will be received via POST

latest_result = {}
result_lock = threading.Lock()

# --- Utility function to calculate Eye Aspect Ratio (EAR) ---
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# --- Core frame processing logic (now accepts a NumPy frame) ---
def process_frame_for_face_detection(frame):
    # Convert to grayscale for dlib's detector
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_dlib = detector(gray) # Detect faces

    frame_data = {"faces_detected": len(faces_dlib), "detections": []}

    # Process each detected face
    for face in faces_dlib:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        landmarks = predictor(gray, face) # Get facial landmarks
        landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()]) # Convert landmarks to NumPy array

        # Face Recognition
        name = "Unknown"
        if known_faces: # Only attempt recognition if known faces are loaded
            face_encoding = np.array(face_rec_model.compute_face_descriptor(frame, landmarks))
            # Calculate distances to known faces
            matches = np.linalg.norm(known_faces - face_encoding, axis=1) < 0.6 # Adjust threshold as needed
            if any(matches):
                best_match_index = np.argmin(matches)
                name = known_names[best_match_index]

        # Eye Blink Detection (EAR)
        left_eye = landmarks_np[LEFT_EYE_IDX]
        right_eye = landmarks_np[RIGHT_EYE_IDX]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        gaze_direction = "Center"
        if avg_ear < BLINK_THRESHOLD:
            gaze_direction = "Blink"

        # Head Pose Estimation (simplified based on nose position relative to face center)
        nose_tip = landmarks_np[30]
        # Approximate face center for head pose
        if len(landmarks_np) > 16: # Check if points 2 and 14 exist for a more stable center
            left_cheek = landmarks_np[2]
            right_cheek = landmarks_np[14]
            face_center_x = (left_cheek[0] + right_cheek[0]) // 2
        else: # Fallback if not enough landmarks or different model
            face_center_x = x + w // 2 # Use bounding box center as fallback

        head_position = "Center"
        # These pixel thresholds (15) might need adjustment based on frame size and typical head movements
        if nose_tip[0] > face_center_x + 15:
            head_position = "Right"
        elif nose_tip[0] < face_center_x - 15:
            head_position = "Left"

        # Construct detection result for the current face
        detection = {
            "face_position": {"x": x, "y": y, "width": w, "height": h},
            "name": name,
            "head_pose": head_position,
            "eye_gaze": gaze_direction, # Renamed from "blink" to "eye_gaze" for broader meaning
            "ear": round(avg_ear, 3), # Include EAR value for debugging/analysis

            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Use current time
        }
        frame_data["detections"].append(detection)

    return frame_data

# The `capture_frames` thread is removed as frames are now POSTed to `/facepredict`

@app.route('/')
def home():
    """Root endpoint that describes available API endpoints"""
    return jsonify({
        "status": "Face Detection API",
        "endpoints": {
            "/facepredict": "POST - Submit a base64 encoded frame for face detection and get results.",
            "/latest": "GET - Get the latest detection result processed by this service.",
            "/health": "GET - Health check endpoint for the face detection service."
        },
        "instructions": "Send base64 encoded JPEG frame in JSON body: {'frame': 'base64_string'}"
    })

@app.route('/facepredict', methods=['POST']) # <--- Changed to POST
def face_predict():
    """
    Receives a base64 encoded frame, processes it for face detection,
    and returns the results.
    """
    global latest_result # Update the global latest_result
    data = request.get_json()
    image_data_b64 = data.get("frame")

    if not image_data_b64:
        return jsonify({"error": "No image data received"}), 400

    try:
        # Decode base64 image data into a NumPy array
        nparr = np.frombuffer(base64.b64decode(image_data_b64), np.uint8)
        frame_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Decode as color image

        if frame_np is None:
            return jsonify({"error": "Could not decode image"}), 400

        # Process the frame using the existing logic
        result = process_frame_for_face_detection(frame_np)

        with result_lock:
            latest_result = result # Store the latest result

        # Log results to JSON file based on your existing conditions
        # (e.g., if no faces, too many faces, unknown face, or head pose/gaze issues)
        if result["faces_detected"] == 0 or result["faces_detected"] > 1 or any(
                d["name"] == "Unknown" or d["head_pose"] != "Center" or d["eye_gaze"] != "Center"
                for d in result["detections"]):
            append_to_shared_json(result)

        return jsonify(result) # Return the processing result

    except Exception as e:
        print(f"Error in /facepredict: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/latest', methods=['GET'])
def get_latest_prediction():
    """Return the most recent prediction."""
    with result_lock:
        if not latest_result:
            return jsonify({
                "status": "error",
                "message": "No frame processed by this service yet"
            }), 404
        return jsonify({
            "status": "success",
            "prediction": latest_result
        })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "Face model is running", "dlib_models_loaded": True}), 200

if __name__ == '__main__':
    # Ensure dlib model files exist
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print("Error: 'shape_predictor_68_face_landmarks.dat' not found. Download it and place it in the same directory.")
        print("You can usually find it at: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 (unzip after download)")
        exit(1)
    if not os.path.exists("dlib_face_recognition_resnet_model_v1.dat"):
        print("Error: 'dlib_face_recognition_resnet_model_v1.dat' not found. Download it and place it in the same directory.")
        print("You can usually find it at: http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2 (unzip after download)")
        exit(1)

    # Initialize results file if it doesn't exist
    if not os.path.exists(SHARED_JSON_FILE):
        with open(SHARED_JSON_FILE, 'w') as f:
            json.dump([], f)

    print("Face Detection Service ready to receive frames on port 5001.")
    # The `threaded=True` is generally handled by Flask's internal server for debug mode,
    # but for production, a WSGI server like Gunicorn handles concurrency.
    app.run(host='0.0.0.0', port=5001, debug=True) # debug=True is good for development

