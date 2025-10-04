
# Krishi Sathi: Empowering Farmers, Sustaining Futures
An AI-powered mobile application to provide farmers with real-time, data-driven agricultural recommendations and solutions.

# OVERVIEW
Krishi Sathi is a smart agriculture solution designed to assist Indian farmers in making informed decisions. The application provides comprehensive, personalized guidance by integrating an AI/ML backend with a user-friendly, cross-platform mobile app. Our goal is to empower farmers with the right knowledge, at the right time, in their own language.

# FEATURES

Krishi Sathi offers a suite of features designed to simplify farming and increase productivity:
AI-Based Crop Recommendation: Recommends the top 3 crops based on soil nutrients, pH, rainfall, temperature, and location data. The recommendations also include projected yield and profit margins.
Disease Detection: Farmers can upload images of crop leaves to instantly detect pests or diseases and receive treatment options .
Advisory Chatbot: Provides simple voice and text support in local languages to answer farming-related questions.
Real-Time Data: Integrates external APIs to provide real-time market price trends and local weather forecasts.
Offline-First Design: The app uses caching to ensure farmers can still access stored recommendations and data even in areas with low connectivity.
Multilingual Support: The user interface and chatbot are available in multiple languages to overcome language barriers.


# Technical Stack

Our solution is built on a three-layered architecture.
1. AI/ML Models (Python)
Crop Recommendation: Gradient Boosting Classifier 
Yield Prediction: Random Forest Regressor 
Disease Detection: Convolutional Neural Network (CNN) 
Libraries: Pandas, NumPy, Scikit-learn, TensorFlow/Keras 
2. APIs & Data Integration (Backend)
Framework: Flask (Python) 
Data integration: Bhuvan APIs, SoilGrids, OpenWeather API, and Mandi/MSP data from Data.gov.in 
Communication: REST APIs with JSON
3. Frontend (Mobile App)
Framework: Flutter (Dart) for a cross-platform (Android/iOS) mobile app 
Data Handling: SQLite caching for offline-first functionality 
