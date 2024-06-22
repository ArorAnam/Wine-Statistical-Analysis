import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
import category_encoders as ce

# Load the data
df = pd.read_csv('wine_review.csv')
df['log_price'] = np.log1p(df['price'])

# Encode 'variety' using target encoding
encoder = ce.TargetEncoder(cols=['variety'])
df['variety_encoded'] = encoder.fit_transform(df['variety'], df['points'])

# Prepare features and target variable
X = df[['log_price', 'variety_encoded']]
y = df['superior_rating']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a classification model (Gradient Boosting Classifier)
model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Plot 1: Probability Plot (ROC Curve)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
# plt.show()
plt.savefig('roc_curve.png')

# Plot 2: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
# plt.show()
plt.savefig('confusion_matrix.png')

# Plot 3: Classification Decision Boundary
plt.figure(figsize=(10, 6))

# Create a mesh to plot
x_min, x_max = X_train['log_price'].min() - 1, X_train['log_price'].max() + 1
y_min, y_max = X_train['variety_encoded'].min() - 1, X_train['variety_encoded'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Predict classifications across the grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
plt.scatter(X_test['log_price'], X_test['variety_encoded'], c=y_test, edgecolors='k', marker='o', cmap='coolwarm')
plt.xlabel('Log Price')
plt.ylabel('Variety (Encoded)')
plt.title('Classification Decision Boundary')
plt.grid(True)
# plt.show()
plt.savefig('classification_boundary.png')

# Print classification report for more details
print(classification_report(y_test, y_pred))
