from flask import Flask, jsonify
from routes.database import db
from routes.disease import disease_bp
from routes.ml_crop import ml_crop_bp 
from routes.data import data_bp
from routes.soil_data import soil_bp
from routes.auth import auth_bp  # ✅ NEW import for login

from flask_jwt_extended import JWTManager  # ✅ JWT for login tokens

app = Flask(__name__)

# 🔹 SQLite database config
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///agroassist.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# 🔹 Secret key for JWT
app.config["JWT_SECRET_KEY"] = "super-secret-key"   # ⚠ change to strong key in production

# 🔹 initialize DB + JWT
db.init_app(app)
jwt = JWTManager(app)

# 🔹 Register blueprints
app.register_blueprint(disease_bp)
app.register_blueprint(data_bp)
app.register_blueprint(ml_crop_bp)
app.register_blueprint(soil_bp)
app.register_blueprint(auth_bp, url_prefix="/auth")   # ✅ Login routes

@app.route("/")
def home():
    return jsonify({"message": "KrishiSathi Backend Running 🚀"})

if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # create tables if not already
    app.run(debug=True)
