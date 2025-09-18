from flask import Blueprint, request, jsonify
from routes.database import db
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import create_access_token
from sqlalchemy import Column, Integer, String

auth_bp = Blueprint("auth", __name__)

# User model
class User(db.Model):
    id = Column(Integer, primary_key=True)
    phone = Column(String(15), unique=True, nullable=False)
    password = Column(String(200), nullable=False)

# Register user
@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    phone = data.get("phone")
    password = data.get("password")

    if not phone or not password:
        return jsonify({"status": "fail", "message": "Missing phone or password"}), 400

    if User.query.filter_by(phone=phone).first():
        return jsonify({"status": "fail", "message": "Phone already registered"}), 409

    hashed_pw = generate_password_hash(password)
    new_user = User(phone=phone, password=hashed_pw)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"status": "success", "message": "User registered successfully"}), 201

# Login user
@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    phone = data.get("phone")
    password = data.get("password")

    user = User.query.filter_by(phone=phone).first()

    if user and check_password_hash(user.password, password):
        token = create_access_token(identity=phone)
        return jsonify({
            "status": "success",
            "message": "Login successful",
            "token": token
        }), 200

    return jsonify({"status": "fail", "message": "Invalid credentials"}), 401
