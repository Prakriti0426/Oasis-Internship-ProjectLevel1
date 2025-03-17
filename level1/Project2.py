import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# ------------------- Load Dataset -------------------
df = pd.read_csv("customer_segmentation_data.csv")  # Make sure this file exists

# ------------------- Data Cleaning -------------------
# Check missing values
print("Missing Values:\n", df.isnull().sum())

# Fill missing values (replace NaN with mean for numerical & mode for categorical)
for column in df.columns:
    if df[column].dtype == "object":  # Categorical column
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:  # Numerical column
        df[column].fillna(df[column].mean(), inplace=True)

# Convert categorical data to numeric
label_encoders = {}
for column in df.select_dtypes(include=["object"]).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# ------------------- Feature Scaling -------------------
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# ------------------- Apply K-Means Clustering -------------------
# Choose optimal number of clusters using Elbow Method
inertia = []
K_range = range(1, 11)  # Checking for 1 to 10 clusters

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()

# Apply K-Means with best K (assume 4 for now)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(df_scaled)

# ------------------- Cluster Visualization -------------------
pca = PCA(n_components=2)  # Reduce to 2D for visualization
df_pca = pca.fit_transform(df_scaled)
df["PCA1"] = df_pca[:, 0]
df["PCA2"] = df_pca[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df["PCA1"], y=df["PCA2"], hue=df["Cluster"], palette="viridis", s=100)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Customer Segmentation using K-Means")
plt.legend(title="Cluster")
plt.show()
