import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import category_encoders as ce
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


df = pd.read_csv('wine_review.csv')

print(df.head())
df.info()

df['sentiment'] = df['description'].apply(lambda x: TextBlob(x).sentiment.polarity)


plt.figure(figsize=(10, 6))
sns.scatterplot(x='sentiment', y='points', data=df)
plt.title('Sentiment Polarity vs. Points')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Points')
plt.savefig('sentiment_vs_points_model_py.png')


encoder = ce.TargetEncoder(cols=['variety'])
df['variety_encoded'] = encoder.fit_transform(df['variety'], df['points'])


df['log_price'] = np.log1p(df['price'])
poly = PolynomialFeatures(degree=2, include_bias=False)
df_poly = poly.fit_transform(df[['log_price']])
df_poly_features = pd.DataFrame(df_poly, columns=poly.get_feature_names_out(['log_price']))
df = pd.concat([df, df_poly_features], axis=1)


X = df[['log_price', 'variety_encoded', 'sentiment'] + list(df_poly_features.columns)]
y = df['points']
y_class = df['superior_rating']


X_train, X_test, y_train, y_test, y_class_train, y_class_test = train_test_split(
    X, y, y_class, test_size=0.2, random_state=42)


regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)
predictions_reg = regressor.predict(X_test)
mse = mean_squared_error(y_test, predictions_reg)
print(f"Mean Squared Error: {mse}")


classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
classifier.fit(X_train, y_class_train)
predictions_class = classifier.predict(X_test)
print(classification_report(y_class_test, predictions_class))


scores = cross_val_score(classifier, X_train, y_class_train, cv=5)
print(f"Average Accuracy: {scores.mean()}")
