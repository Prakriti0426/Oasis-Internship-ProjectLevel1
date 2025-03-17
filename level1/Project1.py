import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("retail_sales_dataset.csv")  # Ensure the file is in the same directory

# Convert 'Date' column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Display basic dataset info
print("Dataset Overview:")
print(df.info())
print("\nFirst 5 Rows:\n", df.head())

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Basic statistics
print("\nDescriptive Statistics:\n", df.describe())

# Extract month for time series analysis
df["Month"] = df["Date"].dt.to_period("M")
monthly_sales = df.groupby("Month")["Total Amount"].sum()

# Plot Monthly Sales Trend
plt.figure(figsize=(12, 6))
monthly_sales.plot(marker='o', linestyle='-', color='b')
plt.xlabel("Month")
plt.ylabel("Total Sales (₹)")
plt.title("Monthly Sales Trend")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Customer Demographics Analysis
gender_sales = df.groupby("Gender")["Total Amount"].sum()
age_bins = [0, 18, 25, 35, 45, 55, 65, 100]
age_labels = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
df["Age Group"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=False)
age_sales = df.groupby("Age Group")["Total Amount"].sum()

# Product Analysis
product_sales = df.groupby("Product Category")["Total Amount"].sum().sort_values(ascending=False)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.barplot(x=gender_sales.index, y=gender_sales.values, ax=axes[0], palette="coolwarm")
axes[0].set_title("Total Sales by Gender")
axes[0].set_ylabel("Total Sales (₹)")
axes[0].set_xlabel("Gender")

sns.barplot(x=product_sales.index, y=product_sales.values, ax=axes[1], palette="viridis")
axes[1].set_title("Total Sales by Product Category")
axes[1].set_ylabel("Total Sales (₹)")
axes[1].set_xlabel("Product Category")
axes[1].tick_params(axis='x', rotation=45)

sns.barplot(x=age_sales.index, y=age_sales.values, ax=axes[2], palette="magma")
axes[2].set_title("Total Sales by Age Group")
axes[2].set_ylabel("Total Sales (₹)")
axes[2].set_xlabel("Age Group")

plt.tight_layout()
plt.show()

# Correlation Analysis - Exclude non-numeric columns
numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns

plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Business Recommendations
print("\nBusiness Recommendations:")
print("1. Stock high-demand products more frequently.")
print("2. Offer discounts to peak-age customers.")
print("3. Focus marketing efforts on the most active gender group.")
print("4. Optimize inventory based on monthly sales trends.")
print("5. Identify correlations between features to improve forecasting.")

print("\nAnalysis Completed! 🚀")
