from flask import Blueprint, request, jsonify

crop_bp = Blueprint("crop", __name__)

@crop_bp.route("/recommend-crop-basic", methods=["POST", "GET"])
def recommend_crop():
    if request.method == "POST":
        data = request.get_json()
        #soil = data.get("soil", "unknown")
        #weather = data.get("weather", "unknown")
    else: 
        soil = request.args.get("soil", "clay")
        weather = request.args.get("weather", "sunny")

    return jsonify({
        "recommended_crop": "Rice 🌾",
        "soil": soil,
        "weather": weather
    })
