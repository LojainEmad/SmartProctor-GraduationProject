from flask import Flask, request, jsonify, send_file, Response, render_template
from flask_cors import CORS
import requests
from concurrent.futures import ThreadPoolExecutor
import time
import threading
import io
import base64
import json
from datetime import datetime
import numpy as np
import cv2
import os

app = Flask(__name__)
CORS(app)

# --- Microservice Configuration ---
SERVICES = {
    'face': {
        'url': 'http://127.0.0.1:5001/facepredict',
        'health': 'http://127.0.0.1:5001/health',
        'timeout': 5,
        'enabled': True
    },
    'lip': {
        'url': 'http://127.0.0.1:5002/lippredict',
        'health': 'http://127.0.0.1:5002/health',
        'timeout': 5,
        'enabled': True
    },
    'object': {
        'url': 'http://127.0.0.1:5003/objpredict',
        'health': 'http://127.0.0.1:5003/health',
        'timeout': 30,
        'enabled': True
    },
    'voice': {
        'url': 'http://127.0.0.1:5004/voicepredict',
        'health': 'http://127.0.0.1:5004/health',
        'timeout': 5,
        'enabled': True
    },
    'text_detection': {  # NEW: AI Text Detection Service
        'url': 'http://127.0.0.1:5005/predict',
        'health': 'http://127.0.0.1:5005/health', # Assuming your AI Flask app has a /health endpoint
        'timeout': 10, # Increased timeout for potential AI model load/inference
        'enabled': True
    }
}

config_lock = threading.Lock()
file_write_lock = threading.Lock() # This lock will now also cover REGISTERED_USERS_FILE

latest_combined_results = {}
latest_processed_frame_bytes = None

FINAL_RESULTS_FILE = 'final_results.json'

# --- Helper function to save combined results ---
def save_combined_results_to_json(results, image_data_b64, audio_data_b64):
    """
    Appends the combined results of all services for a single frame to a JSON file.
    Includes the base64 image and audio data.
    """
    with file_write_lock:
        data_to_save = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "detections": results,
            "frame_data": image_data_b64, # Store base64 image data
            "audio_data": audio_data_b64  # Store base64 audio data
        }
        all_results = []
        if os.path.exists(FINAL_RESULTS_FILE):
            try:
                with open(FINAL_RESULTS_FILE, 'r') as f:
                    all_results = json.load(f)
                # Ensure it's a list; if not, initialize as empty list
                if not isinstance(all_results, list):
                    all_results = []
            except json.JSONDecodeError:
                print(f"Warning: {FINAL_RESULTS_FILE} is corrupted or empty. Starting a new log.")
                all_results = []
            except Exception as e:
                print(f"Error reading {FINAL_RESULTS_FILE}: {e}")
                all_results = []

        all_results.append(data_to_save)

        with open(FINAL_RESULTS_FILE, 'w') as f:
            json.dump(all_results, f, indent=4)


# --- Service Health Check Function ---
def check_service_health():
    """
    Checks the health status of all configured microservices.
    Updates the 'enabled' status in the SERVICES dictionary based on health.
    """
    health_status = {}
    with ThreadPoolExecutor(max_workers=len(SERVICES) if SERVICES else 1) as executor:
        futures = {
            service_name: executor.submit(
                requests.get,
                service_config['health'],
                timeout=5
            )
            for service_name, service_config in SERVICES.items()
        }

        for service_name, future in futures.items():
            try:
                response = future.result()
                is_healthy = response.status_code == 200
                with config_lock:
                    SERVICES[service_name]['enabled'] = is_healthy
                health_status[service_name] = is_healthy
            except Exception as e:
                with config_lock:
                    SERVICES[service_name]['enabled'] = False
                health_status[service_name] = False
                print(f"Gateway Health Error: {service_name} service error: {str(e)}")
    return health_status


# --- API Endpoints ---

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """
    Receives a base64 encoded frame (and optional audio) from the frontend.
    Forwards it to all enabled microservices and aggregates results.
    """
    global latest_combined_results, latest_processed_frame_bytes

    data = request.get_json()
    image_data_b64 = data.get("frame")
    audio_data_b64 = data.get("audio")

    if audio_data_b64:
        print(f"Gateway: Received audio data from frontend. Length: {len(audio_data_b64)} bytes.")
    else:
        print("Gateway: No audio data received from frontend in this frame.")


    if not image_data_b64:
        return jsonify({"error": "No image data received"}), 400

    results = {}
    check_service_health()

    with ThreadPoolExecutor(max_workers=len(SERVICES) if SERVICES else 1) as executor:
        futures = {}
        payload_image = {'frame': image_data_b64}
        payload_audio = {'audio': audio_data_b64} if audio_data_b64 else None

        for service_name, service_config in SERVICES.items():
            if service_config['enabled']:
                if service_name in ['face', 'lip', 'object']:
                    futures[service_name] = executor.submit(
                        requests.post, service_config['url'],
                        json=payload_image,
                        timeout=service_config['timeout']
                    )
                elif service_name == 'voice':
                    if payload_audio:
                        print(f"Gateway: Forwarding audio to voice service. Payload length: {len(str(payload_audio))} bytes.")
                        futures[service_name] = executor.submit(
                            requests.post, service_config['url'],
                            json=payload_audio,
                            timeout=service_config['timeout']
                        )
                    else:
                        results[service_name] = {"status": "skipped",
                                                 "reason": "No audio data provided for voice detection"}
                        print("Gateway: Skipping voice service, no audio payload prepared.")
                # The text_detection service is handled by its own endpoint, not here
            else:
                results[service_name] = {"status": "disabled", "reason": "Service not enabled or unhealthy"}
                print(f"Gateway: Service '{service_name}' is disabled or unhealthy.")

        for service_name, future in futures.items():
            try:
                response = future.result()
                if response.status_code == 200:
                    results[service_name] = response.json()
                    if service_name == 'voice':
                        print(f"Voice service response: {response.json()}")
                else:
                    results[service_name] = {
                        "error": f"{service_name} service returned HTTP {response.status_code}: {response.text}"}
                    with config_lock:
                        SERVICES[service_name]['enabled'] = False
            except requests.exceptions.Timeout:
                results[service_name] = {
                    "error": f"{service_name} service timed out after {SERVICES[service_name]['timeout']} seconds"}
                with config_lock:
                    SERVICES[service_name]['enabled'] = False
            except requests.exceptions.RequestException as e:
                results[service_name] = {"error": f"{service_name} service communication error: {str(e)}"}
                with config_lock:
                    SERVICES[service_name]['enabled'] = False
            except Exception as e:
                results[service_name] = {"error": f"An unexpected error occurred with {service_name} service: {str(e)}"}

    with config_lock:
        latest_combined_results = results
        try:
            latest_processed_frame_bytes = base64.b64decode(image_data_b64)
        except Exception as e:
            print(f"Error decoding base64 image for live_feed: {e}")
            latest_processed_frame_bytes = None

    save_combined_results_to_json(results, image_data_b64, audio_data_b64)

    return jsonify(results)

@app.route("/process_text_detection", methods=["POST"]) # NEW: Endpoint for AI text detection
def process_text_detection():
    """
    Receives text from the frontend (e.g., student's answer),
    forwards it to the AI text detection microservice, and returns the result.
    """
    if not request.is_json:
        return jsonify({"error": "Invalid request, JSON expected."}), 400

    data = request.get_json()
    text = data.get("text")

    if not text:
        return jsonify({"error": "No text provided for detection."}), 400

    service_name = 'text_detection'
    with config_lock:
        service_config = SERVICES.get(service_name)

    if not service_config or not service_config['enabled']:
        return jsonify({
            "error": f"{service_name} service is disabled or not configured/healthy."
        }), 503 # Service Unavailable

    try:
        response = requests.post(
            service_config['url'],
            json={'text': text},
            timeout=service_config['timeout']
        )
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        return jsonify(response.json())
    except requests.exceptions.Timeout:
        return jsonify({"error": f"{service_name} service timed out after {service_config['timeout']} seconds"}), 504 # Gateway Timeout
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"{service_name} service communication error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred with {service_name} service: {str(e)}"}), 500


@app.route('/final_results', methods=['GET'])
def get_final_results():
    """
    Returns all combined results stored in final_results.json.
    """
    with file_write_lock:
        if os.path.exists(FINAL_RESULTS_FILE):
            try:
                with open(FINAL_RESULTS_FILE, 'r') as f:
                    data = json.load(f)
                    return jsonify({
                        "status": "success",
                        "count": len(data),
                        "results": data
                    })
            except json.JSONDecodeError:
                return jsonify({
                    "status": "error",
                    "message": "Final results file is corrupted or empty."
                }), 500
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "message": f"Error reading final results file: {str(e)}"
                }), 500
        else:
            return jsonify({
                "status": "success",
                "count": 0,
                "results": []
            })

@app.route('/get_frame_and_audio/<int:timestamp_index>', methods=['GET'])
def get_frame_and_audio(timestamp_index):
    """
    Retrieves the base64 encoded frame and audio data for a specific timestamp index.
    """
    with file_write_lock:
        if os.path.exists(FINAL_RESULTS_FILE):
            try:
                with open(FINAL_RESULTS_FILE, 'r') as f:
                    all_results = json.load(f)
                    if 0 <= timestamp_index < len(all_results):
                        entry = all_results[timestamp_index]
                        return jsonify({
                            "status": "success",
                            "timestamp": entry.get("timestamp"),
                            "frame_data": entry.get("frame_data"),
                            "audio_data": entry.get("audio_data")
                        })
                    else:
                        return jsonify({"status": "error", "message": "Invalid timestamp index"}), 404
            except json.JSONDecodeError:
                return jsonify({"status": "error", "message": "Final results file is corrupted or empty."}), 500
            except Exception as e:
                return jsonify({"status": "error", "message": f"Error reading final results file: {str(e)}"}), 500
        else:
            return jsonify({"status": "error", "message": "Final results file not found"}), 404


@app.route('/report')
def report_page():
    """
    Serves the HTML page for the detection report.
    """
    return render_template('report_details.html') # Ensure this matches your HTML file name


@app.route('/health')
def health_check():
    """Health check endpoint for the API gateway itself and its integrated microservices."""
    try:
        services_health = check_service_health()
        all_services_healthy = all(services_health.values())
        status = {
            'status': 'up' if all_services_healthy else 'degraded',
            'gateway_status': 'running',
            'services': services_health,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'final_results_file_exists': os.path.exists(FINAL_RESULTS_FILE)
        }
        return jsonify(status), (200 if all_services_healthy else 503)
    except Exception as e:
        return jsonify({'error': str(e), 'gateway_status': 'error'}), 500


@app.route('/combined_results', methods=['GET'])
def get_combined_results():
    """Returns the latest aggregated results from all microservices."""
    with config_lock:
        return jsonify(latest_combined_results)


@app.route('/live_feed', methods=['GET'])
def live_feed():
    """Streams the latest frame received by the gateway back to the frontend using MJPEG."""

    def generate():
        while True:
            with config_lock:
                frame_to_send = latest_processed_frame_bytes

            if frame_to_send:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_to_send + b'\r\n')
            else:
                placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder_img, "Waiting for frames...", (150, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                ret, buffer = cv2.imencode('.jpg', placeholder_img)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.05)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/service_status', methods=['GET'])
def service_status():
    """Get current status and configuration of all services (from gateway's internal config)."""
    with config_lock:
        return jsonify({
            service_name: {
                'url': config['url'],
                'health_endpoint': config['health'],
                'enabled': config['enabled'],
                'timeout': config['timeout']
            }
            for service_name, config in SERVICES.items()
        })


@app.route('/')
def home():
    """Provides a basic API endpoint list for easy navigation."""
    return jsonify({
        "message": "API Gateway for Cheating Detection System",
        "description": "Routes video/audio frames to microservices and aggregates results.",
        "endpoints": {
            "/health": "GET - Current health status of the gateway and all connected microservices.",
            "/process_frame": "POST - Main endpoint to send a base64-encoded video frame (and optional audio) for processing. Returns aggregated detection results.",
            "/process_text_detection": "POST - NEW: Endpoint to send text for AI detection. Returns AI detection result.", # NEW
            "/combined_results": "GET - Retrieves the latest aggregated detection results that the gateway has processed.",
            "/final_results": "GET - Retrieves all combined results stored in final_results.json.",
            "/get_frame_and_audio/<timestamp_index>": "GET - Retrieves base64 image and audio for a specific timestamp index from final_results.json.",
            "/live_feed": "GET - Streams the last processed video frame from the gateway as an MJPEG stream (useful for displaying on a dashboard).",
            "/service_status": "GET - Displays the current configuration and enabled status of all microservices as known by the gateway.",
            "/register_user": "POST - Registers a new user and saves their profile image (proxies to Flasklocal, saves image locally).", # New
            "/get_profile_image/<username>": "GET - Retrieves a registered user's profile image.", # Updated/clarified
            "/registered_users": "GET - Retrieves a list of all registered users and their details.", # New
            "/generate_report": "GET - Generates a PDF report of exam events.",
            "/report": "GET - Displays a dynamic HTML report of detection events."
        },
        "note": "Ensure all required microservices (e.g., object_detection_service) are running on their specified ports before starting the gateway for full functionality."
    })


@app.route("/generate_report", methods=["GET"])
def generate_report():
    """
    Generates a placeholder PDF report.
    For a real application, this would fetch logged detection data and create a detailed report.
    Requires 'reportlab' library: pip install reportlab
    """
    try:
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheets
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheets()
        story = []

        story.append(Paragraph("Exam Cheating Detection Report", styles['h1']))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("This is a placeholder report for demonstration purposes.", styles['Normal']))
        story.append(Paragraph(
            "In a production system, this report would contain detailed logs of detected suspicious activities (e.g., 'phone detected at 10:35:12', 'multiple faces detected at 10:40:05').",
            styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Latest combined results from gateway:", styles['h2']))
        with config_lock:
            if latest_combined_results:
                for service, data in latest_combined_results.items():
                    story.append(Paragraph(f"<b>{service.capitalize()} Service:</b>", styles['Normal']))
                    story.append(
                        Paragraph(f"<font face='Courier'><code>{json.dumps(data, indent=2)}</code></font>",
                                  styles['Code']))
                    story.append(Spacer(1, 0.1 * inch))
            else:
                story.append(Paragraph("No detection results processed yet.", styles['Normal']))

        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph("Full Detection History (from final_results.json):", styles['h2']))
        try:
            with file_write_lock:
                if os.path.exists(FINAL_RESULTS_FILE):
                    with open(FINAL_RESULTS_FILE, 'r') as f:
                        full_history = json.load(f)
                        if full_history:
                            for entry in full_history:
                                story.append(Paragraph(f"<font face='Courier'><code>{json.dumps(entry, indent=2)}</code></font>", styles['Code']))
                                story.append(Spacer(1, 0.05 * inch))
                        else:
                            story.append(Paragraph("No historical detection results found.", styles['Normal']))
                else:
                    story.append(Paragraph("final_results.json not found.", styles['Normal']))
        except Exception as e:
            story.append(Paragraph(f"Error loading historical data: {str(e)}", styles['Normal']))


        doc.build(story)
        buffer.seek(0)
        return send_file(buffer, mimetype="application/pdf", as_attachment=True, download_name='exam_report.pdf')
    except ImportError:
        return jsonify({
                           "error": "reportlab library not found. Please install it to generate PDF reports: `pip install reportlab`"}), 500
    except Exception as e:
        print(f"Error during PDF generation: {e}")
        return jsonify({"error": f"An error occurred during report generation: {str(e)}"}), 500


if __name__ == '__main__':
    # Ensure the directory for profile images exists
    # os.makedirs(PROFILE_IMAGES_DIR, exist_ok=True)

    # Initialize final_results.json if it doesn't exist or is corrupted
    with open(FINAL_RESULTS_FILE, 'a+') as f: # Use a+ to create if not exists
        f.seek(0) # Go to beginning of file
        content = f.read()
        if not content.strip(): # If file is empty or only whitespace
            json.dump([], f)
        else:
            try:
                json.loads(content) # Try to load to check if valid JSON
            except json.JSONDecodeError:
                print(f"Warning: {FINAL_RESULTS_FILE} is corrupted. Re-initializing.")
                f.seek(0)
                f.truncate()
                json.dump([], f)


    print("Performing initial health check for services...")
    check_service_health()
    print("API Gateway starting on port 5000.")
    app.run(debug=True, host="0.0.0.0", port=5000)
