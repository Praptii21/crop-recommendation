from flask import Blueprint, jsonify

data_bp = Blueprint("data", __name__)

@data_bp.route("/soil", methods=["GET"])
def soil_data():
    return jsonify({
        "ph": 6.5,
        "moisture": "Good",
        "type": "Loamy"
    })

@data_bp.route("/weather", methods=["GET"])
def weather_data():
    return jsonify({
        "temperature": "30°C",
        "rainfall": "Moderate",
        "forecast": "Cloudy"
    })

@data_bp.route("/market-prices", methods=["GET"])
def market_prices():
    return jsonify({
        "crop": "Wheat",
        "price_per_quintal": "₹2100"
    })
