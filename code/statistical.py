import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import category_encoders as ce
import numpy as np

# Load the data
df = pd.read_csv('wine_review.csv')
df['log_price'] = np.log1p(df['price'])

# Encode the 'variety' column using target encoding
encoder = ce.TargetEncoder(cols=['variety'])
df['variety_encoded'] = encoder.fit_transform(df['variety'], df['points'])

# Select top 10 most common wine varieties for focused analysis
top_varieties = df['variety'].value_counts().nlargest(10).index
df_top_varieties = df[df['variety'].isin(top_varieties)]

# Setup the figure
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Plot 1: Box Plot for Wine Variety Scores
sns.boxplot(x='variety', y='points', data=df_top_varieties, ax=axes[0], palette='Set2')
axes[0].set_title('Box Plot of Wine Scores by Variety')
axes[0].set_xlabel('Wine Variety')
axes[0].set_ylabel('Points')
axes[0].tick_params(axis='x', rotation=45)

# Plot 2: Violin Plot for Wine Variety Scores
sns.violinplot(x='variety', y='points', data=df_top_varieties, ax=axes[1], palette='Set3')
axes[1].set_title('Violin Plot of Wine Scores by Variety')
axes[1].set_xlabel('Wine Variety')
axes[1].set_ylabel('Points')
axes[1].tick_params(axis='x', rotation=45)

# Plot 3: Swarm Plot for Wine Variety Scores
sns.swarmplot(x='variety', y='points', data=df_top_varieties, ax=axes[2], palette='Set1')
axes[2].set_title('Swarm Plot of Wine Scores by Variety')
axes[2].set_xlabel('Wine Variety')
axes[2].set_ylabel('Points')
axes[2].tick_params(axis='x', rotation=45)

# Adjust layout
plt.tight_layout()
plt.savefig('wine_variety_scores.png')
