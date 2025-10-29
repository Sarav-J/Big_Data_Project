# 🥈 Silver Layer - Data Cleaning & Preprocessing

import pandas as pd
import os

# Load Bronze layer
bronze_path = rbronze_output = r'/Users/sarav/Desktop/Smart City Sensor Analytics/Big-data-project/data/processed/pollution_bronze.csv'

df = pd.read_csv(bronze_path)
print("✅ Bronze data loaded successfully!")
print("Shape:", df.shape)

# --- Cleaning Steps ---

# 1. Handle missing values
missing_before = df.isnull().sum().sum()
df = df.dropna(subset=['Date Local', 'NO2 Mean', 'O3 Mean', 'SO2 Mean', 'CO Mean'])
missing_after = df.isnull().sum().sum()
print(f"🧹 Removed missing rows: {missing_before - missing_after}")

# 2. Convert date column
df['Date Local'] = pd.to_datetime(df['Date Local'], errors='coerce')

# 3. Rename columns (optional, cleaner names)
df.rename(columns={
    'Date Local': 'date',
    'NO2 Mean': 'no2',
    'O3 Mean': 'o3',
    'SO2 Mean': 'so2',
    'CO Mean': 'co'
}, inplace=True)

# 4. Drop redundant columns if any
df = df.drop(columns=['NO2 Units', 'O3 Units', 'SO2 Units', 'CO Units'], errors='ignore')

# 5. Save Silver version
silver_output = "/Users/sarav/Desktop/Smart City Sensor Analytics/Big-data-project/data/processed/pollution_silver.csv"
df.to_csv(silver_output, index=False)

print(f"\n✅ Silver layer data saved → {silver_output}")
print("Shape after cleaning:", df.shape)
df.head()
