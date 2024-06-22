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
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], kde=True)
plt.title('Distribution of Price')
plt.savefig('price_distribution.png')

# Handling missing values - example by dropping them
df_cleaned = df.dropna()

# Handling outliers - example using IQR for 'price'
Q1 = df_cleaned['price'].quantile(0.25)
Q3 = df_cleaned['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_cleaned = df_cleaned[(df_cleaned['price'] >= lower_bound) & (df_cleaned['price'] <= upper_bound)]

# Display the cleaned dataframe
print("\nCleaned Dataframe:")
print(df_cleaned.head())

# Apply log transformation to the price to reduce skewness
df_cleaned['log_price'] = np.log1p(df_cleaned['price'])

# Visualize the new distribution of log-transformed price
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['log_price'], kde=True)
plt.title('Distribution of Log-transformed Price')
plt.savefig('log_price_distribution_after_outliers.png')

from textblob import TextBlob

# Extracting sentiment from descriptions
df['sentiment'] = df['description'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Check how sentiment correlates with points
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sentiment', y='points', data=df)
plt.title('Sentiment Polarity vs. Points')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Points')
plt.show()

