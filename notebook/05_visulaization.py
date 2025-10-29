# 05_visualization_fixed.py
# 🎯 Purpose: Visualize pollution trends, correlations, and model predictions

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import r2_score

# --- Load Silver Dataset ---
silver_path = r"/Users/sarav/Desktop/Smart City Sensor Analytics/Big-data-project/data/processed/pollution_silver.csv"
df = pd.read_csv(silver_path)
print("✅ Dataset Loaded for Visualization:", df.shape)

# --- Basic Overview ---
print(df.head())

# ✅ Helper function to save & show plots
def save_and_show(fig_name):
    output_folder = r"/Users/sarav/Desktop/Smart City Sensor Analytics/outputs"
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(f"{output_folder}/{fig_name}.png", bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"📸 Saved: {fig_name}.png")

# --------------------------------------------------
# 🌍 1️⃣ Pollution Trends Over Time
# --------------------------------------------------
plt.figure(figsize=(12, 6))
df['date'] = pd.to_datetime(df['date'])
df.groupby('date')[['no2', 'o3', 'so2', 'co']].mean().plot(figsize=(12, 6))
plt.title("Average Pollutant Levels Over Time", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Pollutant Concentration")
plt.legend(['NO2', 'O3', 'SO2', 'CO'])
plt.grid(True)
save_and_show("pollution_trends")

# --------------------------------------------------
# 📈 2️⃣ State-wise Pollution Comparison
# --------------------------------------------------
state_avg = df.groupby('State')[['no2', 'o3', 'so2', 'co']].mean().sort_values('no2', ascending=False).head(10)
state_avg.plot(kind='bar', figsize=(12, 6))
plt.title("Top 10 States by Average Pollution Levels", fontsize=14)
plt.ylabel("Concentration Level")
plt.xlabel("State")
plt.grid(True)
save_and_show("statewise_pollution")

# --------------------------------------------------
# 🔄 3️⃣ Correlation Heatmap
# --------------------------------------------------
plt.figure(figsize=(10, 6))
sns.heatmap(df[['no2', 'o3', 'so2', 'co', 'NO2 AQI', 'O3 AQI', 'SO2 AQI', 'CO AQI']].corr(),
            annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Pollutants and AQI")
save_and_show("correlation_heatmap")

# --------------------------------------------------
# 🤖 4️⃣ Model Prediction Visualization
# --------------------------------------------------
model_path = r"/Users/sarav/Desktop/Smart City Sensor Analytics/Big-data-project/notebook/linear_regression_aqi.pkl"
model = joblib.load(model_path)

features = ['no2', 'o3', 'so2', 'co']
df = df.dropna(subset=features + ['NO2 AQI'])
X = df[features]
y = df['NO2 AQI']
y_pred = model.predict(X)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y, y=y_pred, alpha=0.5)
plt.xlabel("Actual NO2 AQI")
plt.ylabel("Predicted NO2 AQI")
plt.title(f"Actual vs Predicted NO2 AQI (R² = {r2_score(y, y_pred):.2f})")
plt.grid(True)
save_and_show("model_prediction")

# --------------------------------------------------
# 🌆 5️⃣ City-level Pollution Comparison
# --------------------------------------------------
city_pollution = df.groupby('City')[['no2', 'o3', 'so2', 'co']].mean().sort_values('no2', ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=city_pollution.index, y=city_pollution['no2'],
            hue=city_pollution.index, palette="viridis", legend=False)
plt.title("Top 10 Most Polluted Cities (by NO2)", fontsize=14)
plt.ylabel("Average NO2 Concentration")
plt.xticks(rotation=45)
save_and_show("city_pollution")
