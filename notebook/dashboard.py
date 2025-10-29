# Pollution Analytics Dashboard

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import r2_score

#  Setup

st.set_page_config(page_title="Smart City Pollution Dashboard", layout="wide")
st.title("🌆 Smart City Pollution Analytics Dashboard")
st.markdown("Visualizing air quality trends, state comparisons, and model predictions")


# Load Dataset

silver_path = r"/Users/sarav/Desktop/Smart City Sensor Analytics/Big-data-project/data/processed/pollution_silver.csv"
df = pd.read_csv(silver_path)
df['date'] = pd.to_datetime(df['date'])
st.sidebar.header("🔍 Filter Options")

#  Sidebar filters
states = st.sidebar.multiselect("Select States", df['State'].unique(), default=df['State'].unique()[:5])
df_filtered = df[df['State'].isin(states)]

# 1️⃣ Pollution Trends
st.subheader("🌍 Average Pollutant Levels Over Time")
fig, ax = plt.subplots(figsize=(10, 5))
df_filtered.groupby('date')[['no2', 'o3', 'so2', 'co']].mean().plot(ax=ax)
plt.title("Average Pollutant Levels Over Time")
plt.ylabel("Concentration")
st.pyplot(fig)

#  2️⃣ State-wise Comparison
st.subheader("📈 Top 10 States by Average Pollution Levels")
state_avg = df.groupby('State')[['no2', 'o3', 'so2', 'co']].mean().sort_values('no2', ascending=False).head(10)
fig, ax = plt.subplots(figsize=(10, 5))
state_avg.plot(kind='bar', ax=ax)
plt.title("Top 10 States by Average NO2")
st.pyplot(fig)

# 3️⃣ Correlation Heatmap
st.subheader("🔄 Correlation Between Pollutants and AQI")
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(df[['no2', 'o3', 'so2', 'co', 'NO2 AQI', 'O3 AQI', 'SO2 AQI', 'CO AQI']].corr(),
            annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)


# 4️⃣ Model Prediction (NO2 AQI)

st.subheader("🤖 Model Prediction: Actual vs Predicted NO2 AQI")

model_path = r"/Users/sarav/Desktop/Smart City Sensor Analytics/Big-data-project/notebook/linear_regression_aqi.pkl"
model = joblib.load(model_path)

features = ['no2', 'o3', 'so2', 'co']
df_model = df.dropna(subset=features + ['NO2 AQI'])
X = df_model[features]
y = df_model['NO2 AQI']
y_pred = model.predict(X)

fig, ax = plt.subplots(figsize=(6, 5))
sns.scatterplot(x=y, y=y_pred, alpha=0.5, ax=ax)
ax.set_xlabel("Actual NO2 AQI")
ax.set_ylabel("Predicted NO2 AQI")
ax.set_title(f"Actual vs Predicted NO2 AQI (R² = {r2_score(y, y_pred):.2f})")
st.pyplot(fig)


# 5️⃣ City-Level Comparison

st.subheader("🌆 Top 10 Most Polluted Cities (by NO2)")
city_pollution = df.groupby('City')[['no2', 'o3', 'so2', 'co']].mean().sort_values('no2', ascending=False).head(10)
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=city_pollution.index, y=city_pollution['no2'], hue=city_pollution.index,
            palette="viridis", legend=False, ax=ax)
plt.title("Top 10 Cities by NO2")
plt.xticks(rotation=45)
st.pyplot(fig)

st.success("✅ Dashboard Loaded Successfully!")
