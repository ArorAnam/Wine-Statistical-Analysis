import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('wine_review.csv')

# Melt the DataFrame for easier plotting
descriptors = ["Crisp", "Dry", "Finish", "Firm", "Fresh", "Fruit", "Full", "Rich", "Round", "Soft", "Sweet"]
df_melt = df.melt(id_vars=["points", "superior_rating"], value_vars=descriptors, var_name="Descriptor", value_name="Presence")

# Plot 1: Distribution of points based on descriptor presence
plt.figure(figsize=(15, 8))
sns.boxplot(x="Descriptor", y="points", hue="Presence", data=df_melt, palette="Set2")
plt.title("Distribution of Points by Descriptor Presence")
plt.xlabel("Descriptor")
plt.ylabel("Points")
plt.legend(title="Presence", loc="upper right")
plt.grid(True, linestyle='--')
# plt.show()
plt.savefig('points_by_descriptor.png')

# Plot 2: Proportion of superior ratings based on descriptor presence
plt.figure(figsize=(15, 8))
sns.barplot(x="Descriptor", y="superior_rating", hue="Presence", data=df_melt, palette="Set1", estimator=np.mean)
plt.title("Proportion of Superior Ratings by Descriptor Presence")
plt.xlabel("Descriptor")
plt.ylabel("Proportion of Superior Rating")
plt.legend(title="Presence", loc="upper right")
plt.grid(True, linestyle='--')
# plt.show()
plt.savefig('superior_rating_by_descriptor.png')
