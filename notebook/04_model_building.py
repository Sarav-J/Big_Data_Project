# 04_model_building.ipynb

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Load the processed (Silver) data ---
silver_path = r"/Users/sarav/Desktop/Smart City Sensor Analytics/Big-data-project/data/processed/pollution_silver.csv"
df = pd.read_csv(silver_path)

print("✅ Dataset Loaded:", df.shape)
print(df.head())

# --- Feature Selection ---
# Use pollutant concentration columns as input features
features = ['no2', 'o3', 'so2', 'co']

# Target variable — choose one AQI column (for example, NO2 AQI)
target = 'NO2 AQI'

# Drop rows with missing values
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Predictions ---
y_pred = model.predict(X_test)

# --- Evaluation ---
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation Results:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# --- Optional: Compare Actual vs Predicted ---
comparison = pd.DataFrame({'Actual AQI': y_test.values, 'Predicted AQI': y_pred})
print("\n--- Sample Predictions ---")
print(comparison.head())

# --- Save Model (optional) ---
import joblib
joblib.dump(model, "linear_regression_aqi.pkl")
print("\n💾 Model saved to '../models/linear_regression_aqi.pkl'")
