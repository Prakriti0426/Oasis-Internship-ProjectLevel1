# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ------------------- Load Dataset -------------------
df = pd.read_csv("creditcard.csv")

# ------------------- Data Exploration -------------------
print("Data Overview:")
print(df.describe())
print(df.info())

# ------------------- Data Preprocessing -------------------
# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Feature scaling
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

# Splitting data into features and labels
X = df.drop('Class', axis=1)
y = df['Class']

# ------------------- Train Test Split -------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nData Split into Training and Testing Sets.")

# ------------------- Model Training -------------------
print("\nTraining Model...")
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, verbose=1)
model.fit(X_train, y_train)

# ------------------- Model Evaluation -------------------
print("\nEvaluating Model...")
y_pred = model.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ------------------- Feature Importance -------------------
plt.figure(figsize=(10, 6))
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title("Feature Importance from RandomForest Classifier")
plt.show()
