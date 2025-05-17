import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Sample Data (Replace this with actual dataset)
data = {
    "feature1": np.random.rand(100),
    "feature2": np.random.rand(100),
    "feature3": np.random.rand(100),
    "target": np.random.rand(100) * 10
}

df = pd.DataFrame(data)

# Splitting features and target
X = df.drop(columns=["target"])
y = df["target"]

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBRegressor
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "model.pkl")
print("✅ Model Trained and Saved Successfully!")

# Evaluate Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"📊 Model Evaluation:\n - MAE: {mae:.4f}\n - MSE: {mse:.4f}")
