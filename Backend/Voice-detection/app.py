# # import datetime
# # import numpy as np
# # import tensorflow_hub as hub
# # import json
# # import base64
# # import io
# # import soundfile as sf
# # from flask import Flask, request, jsonify
# # app = Flask(__name__)
# # model = None
# # class_labels = []
# # TARGET_SAMPLERATE = 16000
# # LOG_FILE = "voice_detections.json"
# # voice_detections_log = [] # In-memory store for detections
# # def load_yamnet_model():
# #     global model, class_labels
# #     try:
# #         print("Voice Detection Service: Loading YAMNet model...")
# #         model = hub.load("https://tfhub.dev/google/yamnet/1")
# #         class_map_path = model.class_map_path().numpy().decode('utf-8')
# #         class_labels = [line.strip().split(',')[-1] for line in open(class_map_path).readlines()[1:]]
# #         print("Voice Detection Service: YAMNet model loaded successfully.")
# #         return True
# #     except Exception as e:
# #         print(f"Voice Detection Service ERROR: Failed to load YAMNet model: {e}")
# #         return False
# # def load_existing_detections():
# #     global voice_detections_log
# #     try:
# #         with open(LOG_FILE, 'r') as f:
# #             content = f.read()
# #             if content:
# #                 voice_detections_log = json.loads(content)
# #                 print(f"Voice Detection Service: Loaded {len(voice_detections_log)} existing detections from {LOG_FILE}.")
# #             else:
# #                 voice_detections_log = []
# #                 print(f"Voice Detection Service: {LOG_FILE} is empty.")
# #     except FileNotFoundError:
# #         voice_detections_log = []
# #         print(f"Voice Detection Service: {LOG_FILE} not found. A new one will be created.")
# #     except json.JSONDecodeError:
# #         print(f"Voice Detection Service Warning: {LOG_FILE} is corrupted or empty. Starting with an empty log.")
# #         voice_detections_log = []
# #     except Exception as e:
# #         print(f"Voice Detection Service ERROR: Failed to load existing detections: {e}")
# #         voice_detections_log = []
# #
# #
# # def save_detection_to_file(detection_entry):
# #     """Appends a new detection to the JSON file."""
# #     global voice_detections_log
# #     voice_detections_log.append(detection_entry)
# #     try:
# #         with open(LOG_FILE, "w") as file:
# #             json.dump(voice_detections_log, file, indent=4)
# #         print(f"Voice Detection Service: Successfully logged to {LOG_FILE}.")
# #     except Exception as e:
# #         print(f"Voice Detection Service ERROR: Failed to save detection to {LOG_FILE}: {e}")
# #
# # @app.route("/voicepredict", methods=["POST"])
# # def voice_predict():
# #     """
# #     Receives a base64 encoded audio chunk, processes it with YAMNet,
# #     and returns the detection result.
# #     """
# #     if model is None:
# #         return jsonify({"error": "Voice model not loaded. Service is not ready."}), 503
# #
# #     print("Voice Detection Service: Received /voicepredict request.")
# #     data = request.get_json()
# #     audio_data_b64 = data.get("audio")
# #
# #     if not audio_data_b64:
# #         print("Voice Detection Service: No audio data provided in request.")
# #         return jsonify({"error": "No audio data provided"}), 400
# #
# #     try:
# #         audio_bytes = base64.b64decode(audio_data_b64)
# #         print(f"Voice Detection Service: Decoded audio bytes (length: {len(audio_bytes)})")
# #         audio_chunk_raw, samplerate = sf.read(io.BytesIO(audio_bytes))
# #         print(f"Voice Detection Service: Audio read by soundfile. Original samplerate: {samplerate}Hz, Shape: {audio_chunk_raw.shape}")
# #         if audio_chunk_raw.ndim > 1:
# #             audio_chunk = np.mean(audio_chunk_raw, axis=1)
# #             print("Voice Detection Service: Converted stereo to mono audio.")
# #         else:
# #             audio_chunk = audio_chunk_raw
# #         if samplerate != TARGET_SAMPLERATE:
# #             try:
# #                 print(f"Voice Detection Service: Resampling audio from {samplerate}Hz to {TARGET_SAMPLERATE}Hz...")
# #                 from scipy.signal import resample as scipy_resample
# #                 audio_chunk = scipy_resample(audio_chunk, num=int(len(audio_chunk) * TARGET_SAMPLERATE / samplerate))
# #
# #             except ImportError:
# #                 print("Voice Detection Service WARNING: 'resampy' not found. Cannot resample audio. YAMNet might perform poorly.")
# #             except Exception as res_e:
# #                 print(f"Voice Detection Service ERROR: Resampling failed: {res_e}")
# #         else:
# #             print("Voice Detection Service: Audio already at target samplerate (16kHz).")
# #         audio_chunk = audio_chunk.astype(np.float32)
# #         print(f"Voice Detection Service: Final audio chunk shape: {audio_chunk.shape}, dtype: {audio_chunk.dtype}")
# #         scores, embeddings, spectrogram = model(audio_chunk)
# #         top_index = np.argmax(scores.numpy()[0])
# #         detected_sound = class_labels[top_index]
# #         confidence = float(scores.numpy()[0, top_index])
# #
# #         timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] # Milliseconds
# #
# #         detection_result = {
# #             "timestamp": timestamp,
# #             "detected_sound": detected_sound,
# #             "confidence": confidence,
# #             "service": "voice-detection"
# #         }
# #         print(f"✅ Voice Detection: [{timestamp}] Detected '{detected_sound}' (Confidence: {confidence:.2f})")
# #         save_detection_to_file(detection_result)
# #         return jsonify(detection_result), 200
# #
# #     except Exception as e:
# #         print(f"⚠️ Voice Detection Service ERROR: An unhandled exception occurred in /voicepredict: {e}")
# #         import traceback
# #         traceback.print_exc()
# #         return jsonify({"error": f"Failed to process audio: {str(e)}"}), 500
# # @app.route("/check_detection", methods=["GET"])
# # def get_latest_detection():
# #     """Endpoint to check the latest detection from the in-memory log."""
# #     if voice_detections_log:
# #         latest_detection = voice_detections_log[-1]
# #         print(f"Voice Detection Service: Latest detection requested: {latest_detection.get('detected_sound', 'N/A')}")
# #         return jsonify(latest_detection), 200
# #     else:
# #         print("Voice Detection Service: No detections yet when /check_detection was called.")
# #         return jsonify({"status": "no detections yet"}), 200
# # @app.route("/health", methods=["GET"])
# # def health_check():
# #     """Health check endpoint for the voice detection microservice."""
# #     status_msg = "ok" if model is not None else "model_loading_failed"
# #     print(f"Voice Detection Service: Health check requested. Status: {status_msg}")
# #     return jsonify({
# #         "status": status_msg,
# #         "service": "voice-detection",
# #         "timestamp": datetime.datetime.now().isoformat(),
# #         "model_loaded": (model is not None)
# #     }), 200
# # if __name__ == '__main__':
# #     print("Voice Detection Service starting up...")
# #     model_loaded_successfully = load_yamnet_model()
# #     load_existing_detections()
# #
# #     if not model_loaded_successfully:
# #         print("Voice Detection Service WARNING: Model failed to load. Service might not function correctly.")
# #
# #     app.run(debug=True, port=5004)
#
#
#
# ################################################
#
# import datetime
# import numpy as np
# import tensorflow_hub as hub
# import json
# import base64
# import io
# import soundfile as sf
# # If resampling is needed (e.g., if your frontend sends audio not at 16kHz)
# # you'll need a resampler. `resampy` is a good choice, install with `pip install resampy`.
# # from resampy import resample
# from flask import Flask, request, jsonify
#
# # --- Flask App Initialization ---
# app = Flask(__name__)
#
# # --- Global Model and Labels ---
# model = None
# class_labels = []
# TARGET_SAMPLERATE = 16000
# LOG_FILE = "voice_detections.json"
# voice_detections_log = [] # In-memory store for detections
#
# # --- Model Loading on Startup ---
# def load_yamnet_model():
#     global model, class_labels
#     try:
#         print("Voice Detection Service: Loading YAMNet model...")
#         model = hub.load("https://tfhub.dev/google/yamnet/1")
#         class_map_path = model.class_map_path().numpy().decode('utf-8')
#         class_labels = [line.strip().split(',')[-1] for line in open(class_map_path).readlines()[1:]]
#         print("Voice Detection Service: YAMNet model loaded successfully.")
#         return True
#     except Exception as e:
#         print(f"Voice Detection Service ERROR: Failed to load YAMNet model: {e}")
#         return False
#
# # --- JSON Log File Management ---
# def load_existing_detections():
#     global voice_detections_log
#     try:
#         with open(LOG_FILE, 'r') as f:
#             content = f.read()
#             if content:
#                 voice_detections_log = json.loads(content)
#                 print(f"Voice Detection Service: Loaded {len(voice_detections_log)} existing detections from {LOG_FILE}.")
#             else:
#                 voice_detections_log = []
#                 print(f"Voice Detection Service: {LOG_FILE} is empty.")
#     except FileNotFoundError:
#         voice_detections_log = []
#         print(f"Voice Detection Service: {LOG_FILE} not found. A new one will be created.")
#     except json.JSONDecodeError:
#         print(f"Voice Detection Service Warning: {LOG_FILE} is corrupted or empty. Starting with an empty log.")
#         voice_detections_log = []
#     except Exception as e:
#         print(f"Voice Detection Service ERROR: Failed to load existing detections: {e}")
#         voice_detections_log = []
#
#
# def save_detection_to_file(detection_entry):
#     """Appends a new detection to the JSON file."""
#     global voice_detections_log
#     voice_detections_log.append(detection_entry)
#     try:
#         with open(LOG_FILE, "w") as file:
#             json.dump(voice_detections_log, file, indent=4)
#         print(f"Voice Detection Service: Successfully logged to {LOG_FILE}.")
#     except Exception as e:
#         print(f"Voice Detection Service ERROR: Failed to save detection to {LOG_FILE}: {e}")
#
# # --- API Endpoints ---
#
# @app.route("/voicepredict", methods=["POST"])
# def voice_predict():
#     """
#     Receives a base64 encoded audio chunk, processes it with YAMNet,
#     and returns the detection result.
#     """
#     if model is None:
#         return jsonify({"error": "Voice model not loaded. Service is not ready."}), 503
#
#     print("Voice Detection Service: Received /voicepredict request.")
#     data = request.get_json()
#     audio_data_b64 = data.get("audio")
#
#     if not audio_data_b64:
#         print("Voice Detection Service: No audio data provided in request.")
#         return jsonify({"error": "No audio data provided"}), 400
#
#     try:
#         # Decode base64 audio data
#         audio_bytes = base64.b64decode(audio_data_b64)
#         print(f"Voice Detection Service: Decoded audio bytes (length: {len(audio_bytes)})")
#
#         # Read audio bytes into a numpy array using soundfile
#         # soundfile is robust and can typically read various formats (WAV, FLAC, etc.)
#         audio_chunk_raw, samplerate = sf.read(io.BytesIO(audio_bytes))
#         print(f"Voice Detection Service: Audio read by soundfile. Original samplerate: {samplerate}Hz, Shape: {audio_chunk_raw.shape}")
#
#         # YAMNet expects mono audio. If stereo, convert to mono by averaging channels.
#         if audio_chunk_raw.ndim > 1:
#             audio_chunk = np.mean(audio_chunk_raw, axis=1)
#             print("Voice Detection Service: Converted stereo to mono audio.")
#         else:
#             audio_chunk = audio_chunk_raw
#
#         # Resample if necessary to TARGET_SAMPLERATE (16kHz for YAMNet)
#         if samplerate != TARGET_SAMPLERATE:
#             # You need `resampy` installed: pip install resampy
#             # from resampy import resample
#             try:
#                 # Assuming resampy is installed. If not, handle or prompt user.
#                 # For basic cases, scipy.signal.resample can also work but resampy is better for audio.
#                 # If using scipy, it's `from scipy.signal import resample`
#                 # audio_chunk = resample(audio_chunk, num=int(len(audio_chunk) * TARGET_SAMPLERATE / samplerate))
#                 print(f"Voice Detection Service: Resampling audio from {samplerate}Hz to {TARGET_SAMPLERATE}Hz...")
#                 # Placeholder for resample call if resampy isn't imported
#                 # audio_chunk = resample(audio_chunk, samplerate, TARGET_SAMPLERATE) # Actual call with resampy
#                 # For now, let's just use scipy if resampy isn't imported to avoid crash,
#                 # but it's not ideal for high quality audio resampling
#                 from scipy.signal import resample as scipy_resample
#                 audio_chunk = scipy_resample(audio_chunk, num=int(len(audio_chunk) * TARGET_SAMPLERATE / samplerate))
#
#             except ImportError:
#                 print("Voice Detection Service WARNING: 'resampy' not found. Cannot resample audio. YAMNet might perform poorly.")
#             except Exception as res_e:
#                 print(f"Voice Detection Service ERROR: Resampling failed: {res_e}")
#                 # Decide if you want to abort or proceed with wrong sample rate
#         else:
#             print("Voice Detection Service: Audio already at target samplerate (16kHz).")
#
#
#         # Ensure audio_chunk is float32 (YAMNet's input requirement)
#         audio_chunk = audio_chunk.astype(np.float32)
#         print(f"Voice Detection Service: Final audio chunk shape: {audio_chunk.shape}, dtype: {audio_chunk.dtype}")
#
#
#         # Process with YAMNet model
#         scores, embeddings, spectrogram = model(audio_chunk)
#         top_index = np.argmax(scores.numpy()[0])
#         detected_sound = class_labels[top_index]
#         confidence = float(scores.numpy()[0, top_index])
#
#         timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] # Milliseconds
#
#         detection_result = {
#             "timestamp": timestamp,
#             "detected_sound": detected_sound,
#             "confidence": confidence,
#             "service": "voice-detection"
#         }
#
#         # Print detection to console
#         print(f"✅ Voice Detection: [{timestamp}] Detected '{detected_sound}' (Confidence: {confidence:.2f})")
#
#         # Save detection to the JSON file
#         save_detection_to_file(detection_result)
#
#         return jsonify(detection_result), 200
#
#     except Exception as e:
#         print(f"⚠️ Voice Detection Service ERROR: An unhandled exception occurred in /voicepredict: {e}")
#         import traceback
#         traceback.print_exc() # Print full traceback for deeper debugging
#         return jsonify({"error": f"Failed to process audio: {str(e)}"}), 500
#
#
# @app.route("/check_detection", methods=["GET"])
# def get_latest_detection():
#     """Endpoint to check the latest detection from the in-memory log."""
#     if voice_detections_log:
#         latest_detection = voice_detections_log[-1]
#         print(f"Voice Detection Service: Latest detection requested: {latest_detection.get('detected_sound', 'N/A')}")
#         return jsonify(latest_detection), 200
#     else:
#         print("Voice Detection Service: No detections yet when /check_detection was called.")
#         return jsonify({"status": "no detections yet"}), 200
#
#
# @app.route("/health", methods=["GET"])
# def health_check():
#     """Health check endpoint for the voice detection microservice."""
#     status_msg = "ok" if model is not None else "model_loading_failed"
#     print(f"Voice Detection Service: Health check requested. Status: {status_msg}")
#     return jsonify({
#         "status": status_msg,
#         "service": "voice-detection",
#         "timestamp": datetime.datetime.now().isoformat(),
#         "model_loaded": (model is not None)
#     }), 200
#
#
# if __name__ == '__main__':
#     print("Voice Detection Service starting up...")
#     model_loaded_successfully = load_yamnet_model()
#     load_existing_detections() # Load any existing log data after model is attempted to be loaded
#
#     if not model_loaded_successfully:
#         print("Voice Detection Service WARNING: Model failed to load. Service might not function correctly.")
#
#     app.run(debug=True, port=5004)


###############################################33

##voice manar

import datetime
import numpy as np
import tensorflow_hub as hub
import json
import base64
import io
import soundfile as sf
from scipy.signal import resample as scipy_resample  # Explicitly import scipy's resample
from flask import Flask, request, jsonify
import os
import traceback  # Import traceback for detailed error logging

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Global Model and Labels ---
model = None
class_labels = []
TARGET_SAMPLERATE = 16000
# Assuming a common default sample rate for incoming raw audio if not explicitly provided
# Modern browsers often record raw PCM at 44100 or 48000 Hz. Let's use 44100 as a common default.
DEFAULT_SOURCE_SAMPLERATE = 44100
LOG_FILE = "voice_detections.jsonl"  # Changed to .jsonl for JSON Lines format


# --- Model Loading on Startup ---
def load_yamnet_model():
    global model, class_labels
    try:
        print("Voice Detection Service: Loading YAMNet model...")
        model = hub.load("https://tfhub.dev/google/yamnet/1")
        class_map_path = model.class_map_path().numpy().decode('utf-8')
        # Filter out empty lines or comments if present
        class_labels = [line.strip().split(',')[-1] for line in open(class_map_path).readlines()[1:] if line.strip()]
        print("Voice Detection Service: YAMNet model loaded successfully.")
        return True
    except Exception as e:
        print(f"Voice Detection Service ERROR: Failed to load YAMNet model: {e}")
        return False


# --- JSONL Log File Management ---
def save_detection_to_file(detection_entry):
    """Appends a new detection to the JSONL file."""
    try:
        with open(LOG_FILE, "a") as file:  # Open in append mode
            file.write(json.dumps(detection_entry) + "\n")  # Write as JSON Line
        print(f"Voice Detection Service: Successfully logged detection to {LOG_FILE}.")
    except Exception as e:
        print(f"Voice Detection Service ERROR: Failed to save detection to {LOG_FILE}: {e}")


# --- API Endpoints ---

@app.route("/voicepredict", methods=["POST"])
def voice_predict():
    """
    Receives a base64 encoded audio chunk, processes it with YAMNet,
    and returns the detection result.
    """
    if model is None:
        print("Voice Detection Service: Model not ready. Returning 503.")
        return jsonify({"error": "Voice model not loaded. Service is not ready."}), 503

    print("Voice Detection Service: Received /voicepredict request.")
    data = request.get_json()
    audio_data_b64 = data.get("audio")

    if not audio_data_b64:
        print("Voice Detection Service: No audio data provided in request.")
        return jsonify({"error": "No audio data provided"}), 400

    audio_chunk = None
    samplerate = None
    processing_error = None

    try:
        audio_bytes = base64.b64decode(audio_data_b64)

        # --- Attempt 1: Try to read as a common audio file format (e.g., WAV) using soundfile ---
        try:
            audio_chunk_raw_sf, samplerate_sf = sf.read(io.BytesIO(audio_bytes))
            # If soundfile successfully read it, check if it's stereo and convert to mono
            if audio_chunk_raw_sf.ndim > 1:
                audio_chunk = np.mean(audio_chunk_raw_sf, axis=1)
            else:
                audio_chunk = audio_chunk_raw_sf
            samplerate = samplerate_sf
            print(
                f"Voice Detection Service: Audio read as file by soundfile. Samplerate: {samplerate}Hz, Shape: {audio_chunk.shape}")
        except Exception as sf_e:
            processing_error = f"Soundfile read failed: {sf_e}"
            print(f"Voice Detection Service: {processing_error}. Attempting raw PCM decode.")

            # --- Attempt 2: If soundfile fails, try to interpret as raw PCM (Float32) ---
            try:
                audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)
                samplerate = DEFAULT_SOURCE_SAMPLERATE  # Use assumed default sample rate for raw PCM
                print(
                    f"Voice Detection Service: Audio interpreted as raw PCM. Assumed Samplerate: {samplerate}Hz, Shape: {audio_chunk.shape}")
            except Exception as np_e:
                processing_error = f"Raw PCM decode failed: {np_e}"
                print(f"Voice Detection Service: {processing_error}.")
                raise ValueError("Could not decode audio data as a supported format (WAV/raw PCM).") from np_e

        if audio_chunk is None or audio_chunk.size == 0:
            print(
                "Voice Detection Service WARNING: Decoded audio chunk is empty or invalid. Skipping YAMNet processing.")
            return jsonify({
                "status": "skipped",
                "reason": "Decoded audio chunk is empty or invalid."
            }), 200

        # Resample if necessary to TARGET_SAMPLERATE (16kHz for YAMNet)
        if samplerate != TARGET_SAMPLERATE:
            # Calculate new length, ensure it's not zero for resampling
            new_length = int(len(audio_chunk) * TARGET_SAMPLERATE / samplerate)
            if new_length == 0:
                print(
                    "Voice Detection Service WARNING: Audio chunk too short for resampling to 16kHz. Skipping YAMNet processing.")
                return jsonify({
                    "status": "skipped",
                    "reason": "Audio chunk too short for meaningful analysis after resampling."
                }), 200

            print(
                f"Voice Detection Service: Resampling audio from {samplerate}Hz to {TARGET_SAMPLERATE}Hz using scipy...")
            audio_chunk = scipy_resample(audio_chunk, num=new_length)
        else:
            print("Voice Detection Service: Audio already at target samplerate (16kHz) or no resampling needed.")

        # Ensure audio_chunk is float32 (YAMNet's input requirement)
        audio_chunk = audio_chunk.astype(np.float32)

        # Process with YAMNet model
        # YAMNet expects a single audio waveform as a tf.Tensor or numpy array.
        scores, embeddings, spectrogram = model(audio_chunk)

        # YAMNet returns scores per frame within the audio. Average over frames to get overall scores.
        mean_scores = np.mean(scores.numpy(), axis=0)
        top_index = np.argmax(mean_scores)
        detected_sound = class_labels[top_index]
        confidence = float(mean_scores[top_index])

        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Milliseconds

        detection_result = {
            "timestamp": timestamp_str,
            "detected_sound": detected_sound,
            "confidence": confidence,
            "service": "voice-detection"
        }

        # Print detection to console
        print(f"✅ Voice Detection: [{timestamp_str}] Detected '{detected_sound}' (Confidence: {confidence:.2f})")

        # Save detection to the JSONL file
        save_detection_to_file(detection_result)

        return jsonify(detection_result), 200

    except Exception as e:
        print(f"⚠️ Voice Detection Service ERROR: An unhandled exception occurred in /voicepredict: {e}")
        traceback.print_exc()  # Print full traceback for deeper debugging
        return jsonify({
                           "error": f"Failed to process audio: {str(e)}. Detailed error: {processing_error if processing_error else 'N/A'}"}), 500


@app.route("/check_detection", methods=["GET"])
def get_latest_detection():
    """
    Endpoint to check the latest detection.
    """
    latest_detection_entry = None
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                # Read lines in reverse to find the last valid JSON object
                for line in reversed(list(f)):
                    if line.strip():
                        try:
                            latest_detection_entry = json.loads(line)
                            break  # Found the last valid entry
                        except json.JSONDecodeError:
                            continue  # Skip corrupted lines
        except Exception as e:
            print(f"Voice Detection Service ERROR: Failed to read latest detection from {LOG_FILE}: {e}")

    if latest_detection_entry:
        print(
            f"Voice Detection Service: Latest detection requested: {latest_detection_entry.get('detected_sound', 'N/A')}")
        return jsonify(latest_detection_entry), 200
    else:
        print("Voice Detection Service: No detections yet when /check_detection was called.")
        return jsonify({"status": "no detections yet"}), 200


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for the voice detection microservice."""
    status_msg = "ok" if model is not None else "model_loading_failed"
    print(f"Voice Detection Service: Health check requested. Status: {status_msg}")
    return jsonify({
        "status": status_msg,
        "service": "voice-detection",
        "timestamp": datetime.datetime.now().isoformat(),
        "model_loaded": (model is not None)
    }), 200


if __name__ == '__main__':
    print("Voice Detection Service starting up...")
    model_loaded_successfully = load_yamnet_model()

    if not model_loaded_successfully:
        print("Voice Detection Service WARNING: Model failed to load. Service might not function correctly.")

    # Ensure log file exists or is created before running the app
    if not os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'w') as f:
                pass  # Just create an empty file
            print(f"Voice Detection Service: Created empty log file: {LOG_FILE}")
        except Exception as e:
            print(f"Voice Detection Service ERROR: Could not create log file {LOG_FILE}: {e}")

    app.run(debug=True, host="0.0.0.0", port=5004)
