import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
df = pd.read_csv('wine_review.csv')

# Display the first few rows of the dataframe
print(df.head())

# Basic properties of the dataset
print("\nDataframe Info:")
df.info()

# Summary statistics for numerical variables
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Review the distribution of key variables
print("\nDistribution of Points:")
print(df['points'].value_counts())

print("\nDistribution of Price:")
# plt.figure(figsize=(10, 6))
# sns.histplot(df['price'], kde=True)
# plt.title('Distribution of Price')
# plt.savefig('price_distribution.png')

# Handling missing values - example by dropping them
df_cleaned = df.dropna()

# Handling outliers - example using IQR for 'price'
Q1 = df_cleaned['price'].quantile(0.25)
Q3 = df_cleaned['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_cleaned = df_cleaned[(df_cleaned['price'] >= lower_bound) & (df_cleaned['price'] <= upper_bound)]

# Apply log transformation to the price to reduce skewness
df_cleaned['log_price'] = np.log1p(df_cleaned['price'])

# Visualize the new distribution of log-transformed price
# plt.figure(figsize=(10, 6))
# sns.histplot(df_cleaned['log_price'], kde=True)
# plt.title('Distribution of Log-transformed Price')
# plt.savefig('log_price_distribution_after_outliers.png')

# Display the cleaned dataframe
print("\nCleaned Dataframe:")
print(df_cleaned.head())


from sklearn.feature_extraction.text import TfidfVectorizer

# Example of using TF-IDF to vectorize 'description'
tfidf = TfidfVectorizer(max_features=100)  # consider top 100 terms
description_tfidf = tfidf.fit_transform(df['description'])

# For simplicity in models, just check the shape and the type of matrix created
print(description_tfidf.shape)

# Binning price into categories
df_cleaned['price_bin'] = pd.cut(df_cleaned['price'], bins=[0, 20, 40, 60, 80, 100, np.inf], labels=['0-20', '20-40', '40-60', '60-80', '80-100', '>100'])


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Splitting data
X = df_cleaned[['log_price']]  # initially using log_price as the only predictor
y = df_cleaned['superior_rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Building a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluating the model
print(classification_report(y_test, predictions))
