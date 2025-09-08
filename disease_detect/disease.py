import os
import tensorflow as tf
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model
MODEL_PATH = os.path.join(BASE_DIR, "final_disease_model_robust.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names
CLASS_PATH = os.path.join(BASE_DIR, "class_names.txt")
with open(CLASS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

def predict_disease(image):
    img = tf.keras.preprocessing.image.load_img(image, target_size=(224, 224))  
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    return {
        "disease": predicted_class,
        "confidence": round(float(confidence) * 100, 2)
    }
