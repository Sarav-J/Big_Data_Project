# 🥇 Gold Layer - EDA for Smart City Pollution Analysis

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Silver Layer Data
silver_file = r"/Users/sarav/Desktop/Smart City Sensor Analytics/Big-data-project/data/processed/pollution_silver.csv"
df = pd.read_csv(silver_file)

print("✅ Silver Data Loaded Successfully!")
print(df.head())
print("\nColumns available:", df.columns.tolist())

# -------------------------------
# 1️ Basic Summary
# -------------------------------
print("\n--- Basic Statistics ---")
print(df[['no2', 'o3', 'so2', 'co']].describe())

# -------------------------------
# 2️ Average Pollution per City
# -------------------------------
city_avg = df.groupby('City')[['no2', 'o3', 'so2', 'co']].mean().reset_index()
top_cities = city_avg.sort_values(by='no2', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_cities, x='no2', y='City', palette='coolwarm')
plt.title("Top 10 Cities with Highest NO₂ Levels")
plt.xlabel("Average NO₂ Level (ppm)")
plt.ylabel("City")
plt.show()

# -------------------------------
# 3️ Pollutant Correlation
# -------------------------------
plt.figure(figsize=(6, 4))
corr = df[['no2', 'o3', 'so2', 'co']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Pollutant Correlation Heatmap")
plt.show()

# -------------------------------
# 4️ Pollution Trend Over Time
# -------------------------------
df['date'] = pd.to_datetime(df['date'], errors='coerce')
trend = df.groupby('date')[['no2', 'o3', 'so2', 'co']].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.plot(trend['date'], trend['no2'], label='NO₂')
plt.plot(trend['date'], trend['o3'], label='O₃')
plt.plot(trend['date'], trend['so2'], label='SO₂')
plt.plot(trend['date'], trend['co'], label='CO')
plt.title("Average Pollutant Levels Over Time")
plt.xlabel("Date")
plt.ylabel("Concentration (ppm)")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# 5️ State-wise Pollution Comparison
# -------------------------------
state_avg = df.groupby('State')[['no2', 'o3', 'so2', 'co']].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(data=state_avg.sort_values(by='no2', ascending=False).head(15),
            x='no2', y='State', palette='Spectral')
plt.title("Top 15 States by Average NO₂ Concentration")
plt.xlabel("Average NO₂ (ppm)")
plt.ylabel("State")
plt.show()

print("✅ EDA Completed Successfully!")
