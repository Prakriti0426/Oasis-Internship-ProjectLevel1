# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report

# ------------------- Load Datasets -------------------
apps_df = pd.read_csv('apps.csv')
reviews_df = pd.read_csv('user_reviews.csv')

print("Apps Data Overview:\n", apps_df.head())
print("\nUser Reviews Data Overview:\n", reviews_df.head())

# ------------------- Data Cleaning -------------------
# Cleaning apps.csv
apps_df.dropna(inplace=True)
apps_df = apps_df[apps_df['Rating'] <= 5]

# Cleaning user_reviews.csv
reviews_df.dropna(subset=['Sentiment'], inplace=True)

print("\nData Cleaned Successfully!")

# ------------------- Category Exploration -------------------
plt.figure(figsize=(12, 6))
sns.countplot(y='Category', data=apps_df, order=apps_df['Category'].value_counts().index)
plt.title('App Count by Category')
plt.xlabel('Number of Apps')
plt.ylabel('Category')
plt.show()

# ------------------- Metrics Analysis -------------------
# Rating Distribution
plt.figure(figsize=(10, 5))
sns.histplot(apps_df['Rating'], bins=30, kde=True, color='blue')
plt.title('App Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Convert Price column to float
apps_df['Price'] = apps_df['Price'].str.replace('$', '').astype(float)

# Price Distribution
plt.figure(figsize=(12, 6))
sns.boxplot(x='Category', y='Price', data=apps_df)
plt.xticks(rotation=90)
plt.title('Price Distribution by Category')
plt.show()


positive_reviews = " ".join(review for review in reviews_df[reviews_df['Sentiment'] == 'Positive']['Translated_Review'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Positive Reviews Word Cloud')
plt.show()

# Sentiment Distribution
sns.countplot(x='Sentiment', data=reviews_df, palette='cool')
plt.title('Sentiment Distribution')
plt.show()

print("Analysis Completed! Insights Generated.")
