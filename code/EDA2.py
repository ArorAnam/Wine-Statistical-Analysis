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

# Distribution of 'points'
plt.figure(figsize=(10, 6))
sns.histplot(df['points'], bins=20, kde=False)
plt.title('Distribution of Points')
plt.xlabel('Points')
plt.ylabel('Frequency')
# plt.show()
plt.savefig('points_distribution.png')

# Distribution of 'price' after log transformation
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['log_price'], kde=True)
plt.title('Distribution of Log-transformed Price')
plt.xlabel('Log Price')
plt.ylabel('Frequency')
# plt.show()
plt.savefig('log_price_distribution_transformed.png')

# Count of different 'varieties'
plt.figure(figsize=(12, 8))
sns.countplot(y='variety', data=df, order = df['variety'].value_counts().index)
plt.title('Frequency of Wine Varieties')
plt.xlabel('Count')
plt.ylabel('Variety')
# plt.show()
plt.savefig('variety_count.png')

# Relationship between 'price' and 'points'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='log_price', y='points', data=df_cleaned)
plt.title('Log-Price vs Points')
plt.xlabel('Log Price')
plt.ylabel('Points')
# plt.show()
plt.savefig('log_price_vs_points.png')

# Average points per variety
variety_points = df.groupby('variety')['points'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 8))
variety_points.plot(kind='bar')
plt.title('Average Points by Wine Variety')
plt.xlabel('Variety')
plt.ylabel('Average Points')
# plt.show()
plt.savefig('average_points_per_variety.png')
