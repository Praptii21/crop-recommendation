from flask import Blueprint, request, jsonify
import os
from disease_detect.disease import predict_disease

disease_bp = Blueprint("disease", __name__)

UPLOAD_FOLDER = "backend/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@disease_bp.route("/detect-disease", methods=["POST"])
def detect_disease():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    result = predict_disease(filepath)

    return jsonify(result)
