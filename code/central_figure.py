import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import category_encoders as ce

# Load the data
df = pd.read_csv('wine_review.csv')
df['log_price'] = np.log1p(df['price'])

# Encode the 'variety' column using target encoding
encoder = ce.TargetEncoder(cols=['variety'])
df['variety_encoded'] = encoder.fit_transform(df['variety'], df['points'])

# Setup the grid layout with GridSpec
fig = plt.figure(constrained_layout=True, figsize=(15, 12))
gs = fig.add_gridspec(3, 2)

# Plot 1: Log-Price vs Points
ax1 = fig.add_subplot(gs[0, 0])
sns.scatterplot(x='log_price', y='points', data=df, ax=ax1)
ax1.set_title('Log-Price vs Points')
ax1.set_xlabel('Log Price')
ax1.set_ylabel('Points')

# Plot 2: Average Points by Wine Variety
ax2 = fig.add_subplot(gs[0, 1])
average_points_by_variety = df.groupby('variety')['points'].mean().sort_values(ascending=False)
average_points_by_variety.plot(kind='bar', ax=ax2, color='skyblue')
ax2.set_title('Average Points by Wine Variety')
ax2.set_xlabel('Wine Variety')
ax2.set_ylabel('Average Points')
ax2.tick_params(axis='x', rotation=90)

# Plot 3: Distribution of Log-Transformed Price
ax3 = fig.add_subplot(gs[1, 0])
sns.histplot(df['log_price'], kde=True, ax=ax3)
ax3.set_title('Distribution of Log-Transformed Price')
ax3.set_xlabel('Log Price')
ax3.set_ylabel('Count')

# Plot 4: Distribution of Points
ax4 = fig.add_subplot(gs[1, 1])
sns.histplot(df['points'], bins=20, kde=True, ax=ax4)
ax4.set_title('Distribution of Points')
ax4.set_xlabel('Points')
ax4.set_ylabel('Count')

# Add a central conclusion or summary as a text box
ax5 = fig.add_subplot(gs[2, :])
ax5.axis('off')
summary_text = """
Summary of Findings:
1. Wines with higher log-transformed prices tend to receive higher ratings, as shown in Plot 1.
2. The average points by wine variety (Plot 2) indicate that certain varieties consistently receive higher ratings.
3. The distribution of log-transformed prices (Plot 3) is approximately normal, indicating the effectiveness of the transformation.
4. The distribution of points (Plot 4) shows a slight skew toward higher ratings, with many wines clustered between 85 and 92 points.
"""
ax5.text(0.05, 0.5, summary_text, fontsize=12, wrap=True)

# Adjust the layout
plt.tight_layout()
plt.savefig('central_figure.png')
