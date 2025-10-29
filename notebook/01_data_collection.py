

import pandas as pd
import os
from IPython.display import display   # ✅ Add this line

# File path (relative)
file_path = r'/Users/sarav/Desktop/Smart City Sensor Analytics/Big-data-project/data/raw/pollution_us.csv'

# Check if file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ File not found at {file_path}. Please verify!")

# Load dataset
df = pd.read_csv(file_path)
print("✅ Dataset Loaded Successfully!")
print("Shape:", df.shape)

# Display first few rows
display(df.head())   # ✅ Now this will work

# Basic info
print("\n--- Dataset Info ---")
df.info()

# Save a bronze copy (raw data preserved)
bronze_output = r'/Users/sarav/Desktop/Smart City Sensor Analytics/Big-data-project/data/processed/pollution_bronze.csv'
df.to_csv(bronze_output, index=False)
print(f"\n💾 Raw data saved as Bronze layer → {bronze_output}")
