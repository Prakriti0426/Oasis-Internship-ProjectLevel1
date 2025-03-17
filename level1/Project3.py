import pandas as pd
import json
import zipfile
import numpy as np

# Load the JSON file (Category mapping for YouTube Data)
with open("CA_category_id.json", "r") as f:
    category_data = json.load(f)
    categories = {str(item["id"]): item["snippet"]["title"] for item in category_data["items"]}

# Extract and load the Airbnb NYC dataset
zip_path = "AB_NYC_2019.csv.zip"
with zipfile.ZipFile(zip_path, 'r') as z:
    csv_file = z.namelist()[0]  # Get CSV filename
    df = pd.read_csv(z.open(csv_file))

# 1. Data Integrity Check
def check_integrity(df):
    print("Initial Dataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nDuplicate Rows:", df.duplicated().sum())

check_integrity(df)

# 2. Handling Missing Values
def handle_missing_values(df):
    df.fillna({
        "name": "Unknown",  # Replace missing names with 'Unknown'
        "host_name": "Unknown",  # Replace missing host names
        "reviews_per_month": df["reviews_per_month"].median(),  # Fill numeric missing values with median
    }, inplace=True)
    df.dropna(subset=["latitude", "longitude", "price"], inplace=True)  # Drop essential missing values
    return df

df = handle_missing_values(df)

# 3. Removing Duplicates
def remove_duplicates(df):
    df.drop_duplicates(inplace=True)
    return df

df = remove_duplicates(df)

# 4. Standardizing Formatting
def standardize_data(df):
    df["price"] = df["price"].astype(float)
    df["name"] = df["name"].str.title().str.strip()
    df["host_name"] = df["host_name"].str.title().str.strip()
    return df

df = standardize_data(df)

# 5. Outlier Detection (Using IQR method)
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

df = detect_outliers(df, "price")

# Final Data Overview
print("\nCleaned Dataset Info:")
print(df.info())
print("\nSample Data:")
print(df.head())

# Save cleaned data
df.to_csv("Cleaned_AB_NYC_2019.csv", index=False)
