from flask import Flask, jsonify
from database import db 
from routes.disease import disease_bp
from routes.ml_crop import ml_crop_bp 

app = Flask(__name__)

# 🔹 SQLite database config
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///agroassist.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# 🔹 initialize DB
db.init_app(app)


app.register_blueprint(disease_bp)
app.register_blueprint(ml_crop_bp)

@app.route("/")
def home():
    return jsonify({"message": "AgroAssist Backend Running 🚀"})

if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # create tables if not already
    app.run(debug=True)

