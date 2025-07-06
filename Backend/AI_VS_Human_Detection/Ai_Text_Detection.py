## Ai text detection before the flask

# from flask import Flask, request, jsonify
# import os
# import torch
# import torch.nn.functional as F
# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
# from datetime import datetime
#
# # Disable symlink warnings on Windows
# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
#
# # Load model and tokenizer once
# model = DistilBertForSequenceClassification.from_pretrained("ai_text_detector")
# tokenizer = DistilBertTokenizer.from_pretrained("ai_text_detector")
# model.eval()
#
# # Flask app
# app = Flask(__name__)
#
# # Prediction function
# def predict_text_category(text):
#     encoding = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
#     with torch.no_grad():
#         output = model(encoding["input_ids"], attention_mask=encoding["attention_mask"])
#     probabilities = F.softmax(output.logits, dim=1)
#     predicted_label = torch.argmax(probabilities, dim=1).item()
#     return predicted_label  # 1 = AI-generated, 0 = Human-written
#
# # JSON prediction endpoint (similar to lip model)
# @app.route("/predict", methods=["POST"])
# def predict():
#     if not request.is_json:
#         return jsonify({"error": "Invalid request, JSON expected."}), 400
#
#     data = request.get_json()
#     text = data.get("text")
#
#     if not text:
#         return jsonify({"error": "No text provided."}), 400
#
#     try:
#         prediction = predict_text_category(text)
#         label = "AI-generated" if prediction == 1 else "Human-written"
#
#         return jsonify({
#             "prediction": label,
#             "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         })
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
# # Default route
# @app.route("/", methods=["GET"])
# def home():
#     return "🧠 AI Text Detection API is Running!"
#
# # Run app
# if __name__ == "__main__":
#     app.run(debug=True, port=5005)

##----------------------------------------------------------


from flask import Flask, request, jsonify
import os
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datetime import datetime

# Disable symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load model and tokenizer once
model = DistilBertForSequenceClassification.from_pretrained("ai_text_detector")
tokenizer = DistilBertTokenizer.from_pretrained("ai_text_detector")
model.eval()

# Flask app
app = Flask(__name__)

# Prediction function
def predict_text_category(text):
    encoding = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    with torch.no_grad():
        output = model(encoding["input_ids"], attention_mask=encoding["attention_mask"])
    probabilities = F.softmax(output.logits, dim=1)
    predicted_label = torch.argmax(probabilities, dim=1).item()
    return predicted_label  # 1 = AI-generated, 0 = Human-written

# JSON prediction endpoint (similar to lip model)
@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Invalid request, JSON expected."}), 400

    data = request.get_json()
    text = data.get("text")

    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        prediction = predict_text_category(text)
        label = "AI-generated" if prediction == 1 else "Human-written"

        return jsonify({
            "prediction": label,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add this to AI_text_detection.py
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'AI Text Detection service is healthy'}), 200

# Default route
@app.route("/", methods=["GET"])
def home():
    return "🧠 AI Text Detection API is Running!"

# Run app
if __name__ == "__main__":
    app.run(debug=True, port=5005)
