from flask import Blueprint, request, jsonify
import pickle, pandas as pd, os, numpy as np

# === Blueprint ===
ml_crop_bp = Blueprint("ml_crop", __name__)

# === Base directory for relative paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../crops_model")

# === Load Models & Encoders ===
try:
    encoder = pickle.load(open(os.path.join(MODEL_DIR, "encoder.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))
    model_gbc = pickle.load(open(os.path.join(MODEL_DIR, "model_gbc.pkl"), "rb"))
    yield_model = pickle.load(open(os.path.join(MODEL_DIR, "yield_model.pkl"), "rb"))
    le_crop = pickle.load(open(os.path.join(MODEL_DIR, "le_crop.pkl"), "rb"))
    le_season = pickle.load(open(os.path.join(MODEL_DIR, "le_season.pkl"), "rb"))
    le_state = pickle.load(open(os.path.join(MODEL_DIR, "le_state.pkl"), "rb"))
except Exception as e:
    print("Error loading ML models:", e)

# helper to safely transform with LabelEncoder-like objects
def safe_transform(encoder_obj, value, fallback=0):
    try:
        return int(encoder_obj.transform([value])[0])
    except Exception:
        return fallback

# === Prediction Function ===
def predict_crop_and_yield(N, P, K, temperature, humidity, ph, rainfall, season, state, area, fertilizer, pesticide):
    input_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    input_scaled = scaler.transform(input_df)

    recommendations = []

    try:
        # === Crop Prediction (Top 3) ===
        probs = model_gbc.predict_proba(input_scaled)[0]
        classes = model_gbc.classes_
        top_idx = np.argsort(probs)[::-1][:3]   # Top 3 crops
        top_crops = encoder.inverse_transform(classes[top_idx])

        # === Predict yield for each top crop ===
        for crop in top_crops:
            crop_for_yield = safe_transform(le_crop, crop, fallback=0)
            season_encoded = safe_transform(le_season, season, fallback=0)
            state_encoded = safe_transform(le_state, state, fallback=0)

            features = [[crop_for_yield, season_encoded, state_encoded, area, rainfall, fertilizer, pesticide]]

            try:
                predicted_yield_val = yield_model.predict(features)[0]
                predicted_yield = round(float(predicted_yield_val), 2)
            except Exception:
                predicted_yield = None

            # === Final response structure ===
            recommendations.append({
                "name": str(crop),
                "yield": predicted_yield,
                "profit": 1500,  # dummy static value
                "sustainability": "moderate",  # dummy
                "description": "Sample description for testing.",  # dummy
                "image": f"assets/images/{crop.lower()}.png"  # dummy image path
            })

    except Exception as e:
        return {"error": f"Prediction error: {e}"}

    return {"recommendations": recommendations}

# === Route ===
@ml_crop_bp.route("/recommend-crop", methods=["GET", "POST"])
def recommend_crop():
    try:
        if request.method == "GET":
            data = {
                "N": float(request.args.get("N", 0)),
                "P": float(request.args.get("P", 0)),
                "K": float(request.args.get("K", 0)),
                "ph": float(request.args.get("ph", 7)),
                "temperature": float(request.args.get("temperature", 0)),
                "humidity": float(request.args.get("humidity", 0)),
                "rainfall": float(request.args.get("rainfall", 0)),
                "fertilizer": float(request.args.get("fertilizer", 0)),
                "pesticide": float(request.args.get("pesticide", 0)),
                "area": float(request.args.get("area", 1)),
                "season": request.args.get("season", "Kharif"),
                "state": request.args.get("state", "Unknown"),
            }
        else:  # POST
            data = request.json

        result = predict_crop_and_yield(
            data["N"], data["P"], data["K"],
            data["temperature"], data["humidity"], data["ph"], data["rainfall"],
            data["season"], data["state"], data["area"], data["fertilizer"], data["pesticide"]
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})
