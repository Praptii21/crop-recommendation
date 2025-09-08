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
    # keep going — route will error appropriately if models are missing

# helper to safely transform with LabelEncoder-like objects
def safe_transform(encoder_obj, value, fallback=0):
    try:
        return int(encoder_obj.transform([value])[0])
    except Exception:
        return fallback

# === Prediction Function (returns top 3 + yield for best) ===
def predict_crop_and_yield(N, P, K, temperature, humidity, ph, rainfall, season, state, area, fertilizer, pesticide):
    # prepare input
    input_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    input_scaled = scaler.transform(input_df)

    # get top 3 recommendations (if model supports predict_proba)
    recommendations = []
    try:
        if hasattr(model_gbc, "predict_proba"):
            probs = model_gbc.predict_proba(input_scaled)[0]   # probabilities
            classes = model_gbc.classes_
            top_idx = np.argsort(probs)[::-1][:3]
            # classes[top_idx] are encoded labels; encoder.inverse_transform expects encoded labels
            crops = encoder.inverse_transform(classes[top_idx])
            top_probs = probs[top_idx] * 100  # to percentage
            recommendations = [
                {"crop": str(crop), "confidence": round(float(conf), 2)}
                for crop, conf in zip(crops, top_probs)
            ]
        else:
            # fallback: only top1 via predict
            crop_encoded = model_gbc.predict(input_scaled)
            crop_name = encoder.inverse_transform(crop_encoded)[0]
            recommendations = [{"crop": str(crop_name), "confidence": None}]
    except Exception as e:
        # If something goes wrong with prob prediction, fallback to single predict
        try:
            crop_encoded = model_gbc.predict(input_scaled)
            crop_name = encoder.inverse_transform(crop_encoded)[0]
            recommendations = [{"crop": str(crop_name), "confidence": None}]
        except Exception as inner_e:
            return {
                "recommendations": [],
                "expected_yield": "Yield prediction not available",
                "inputs_used": {
                    "N": N, "P": P, "K": K,
                    "temperature": temperature,
                    "humidity": humidity,
                    "ph": ph,
                    "rainfall": rainfall,
                    "season": season,
                    "state": state,
                    "area": area,
                    "fertilizer": fertilizer,
                    "pesticide": pesticide
                },
                "error": f"Model prediction error: {e}; fallback error: {inner_e}"
            }

    # pick best crop (first in recommendations) for yield prediction
    best_crop = recommendations[0]["crop"] if recommendations else None

    # safe transforms for encoders
    crop_for_yield = safe_transform(le_crop, best_crop, fallback=0) if best_crop else 0
    season_encoded = safe_transform(le_season, season, fallback=0)
    state_encoded = safe_transform(le_state, state, fallback=0)

    features = [[crop_for_yield, season_encoded, state_encoded, area, rainfall, fertilizer, pesticide]]
    try:
        predicted_yield_val = yield_model.predict(features)[0]
        predicted_yield = f"{predicted_yield_val:.2f} tons/hectare"
    except Exception:
        predicted_yield = "Yield prediction not available for this crop"

    return {
        "recommendations": recommendations,
        "expected_yield": predicted_yield,
        "inputs_used": {
            "N": N, "P": P, "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
            "season": season,
            "state": state,
            "area": area,
            "fertilizer": fertilizer,
            "pesticide": pesticide
        }
    }

# === Route ===
@ml_crop_bp.route("/recommend-crop", methods=["GET", "POST"])
def recommend_crop():
    try:
        # For GET: get data from query parameters
        if request.method == "GET":
            data = {
                "N": float(request.args.get("N", 0)),
                "P": float(request.args.get("P", 0)),
                "K": float(request.args.get("K", 0)),
                "temperature": float(request.args.get("temperature", 0)),
                "humidity": float(request.args.get("humidity", 0)),
                "ph": float(request.args.get("ph", 0)),
                "rainfall": float(request.args.get("rainfall", 0)),
                "season": request.args.get("season", "Kharif"),
                "state": request.args.get("state", "Unknown"),
                "area": float(request.args.get("area", 1)),
                "fertilizer": float(request.args.get("fertilizer", 0)),
                "pesticide": float(request.args.get("pesticide", 0))
            }
        else:
            data = request.json

        # call prediction and assign to result (so it's defined)
        result = predict_crop_and_yield(
            data["N"], data["P"], data["K"],
            data["temperature"], data["humidity"], data["ph"], data["rainfall"],
            data["season"], data["state"], data["area"], data["fertilizer"], data["pesticide"]
        )

        # build HTML to show top 3 recommendations
        recs_html = ""
        if result.get("recommendations"):
            for rec in result["recommendations"]:
                conf_text = f" ({rec['confidence']}%)" if rec.get("confidence") is not None else ""
                recs_html += f"<li>{rec['crop']}{conf_text}</li>"
        else:
            recs_html = "<li>No recommendations available</li>"

        best_crop = result["recommendations"][0]["crop"] if result.get("recommendations") else "N/A"
        return f"""
        <h2>Crop Recommendation Result</h2>
        <ul>
            {recs_html}
        </ul>
        <p><b>Expected Yield (for best crop {best_crop}):</b> {result['expected_yield']}</p>
        <p><b>Inputs Used:</b> {result['inputs_used']}</p>
        """

    except Exception as e:
        return f"<p>Error: {e}</p>"
