import joblib
import numpy as np

# Define the path to the saved model
model_path = "model.pkl"

try:
    # Load the trained XGBoost model
    model = joblib.load(model_path)
    print("✅ Model Loaded Successfully!")

    # Example: Make a prediction with new data
    # Replace this with real input features (ensure it matches the trained model's feature count)
    new_data = np.array([[0.5, 0.7, 0.2]])  # Example input, adjust as needed

    # Predict using the loaded model
    prediction = model.predict(new_data)
    print(f"🔮 Predicted Value: {prediction[0]:.4f}")

except FileNotFoundError:
    print("❌ model.pkl not found! Train and save the model first by running train_xgb_model.py.")
