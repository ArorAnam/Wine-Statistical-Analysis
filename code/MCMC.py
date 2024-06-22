import pymc3 as pm
import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('wine_review.csv')

sample_size = 500
df_sample = df.sample(n=sample_size, random_state=42)

import category_encoders as ce
encoder = ce.TargetEncoder(cols=['variety'])
df_sample['variety_encoded'] = encoder.fit_transform(df_sample['variety'], df_sample['points'])

df_sample['log_price'] = np.log1p(df_sample['price'])

X_bayesian = df_sample[['log_price', 'variety_encoded']].values
y_bayesian = df_sample['points'].values

X_mean = np.mean(X_bayesian, axis=0)
X_std = np.std(X_bayesian, axis=0)
X_bayesian_normalized = (X_bayesian - X_mean) / X_std

def run():
    with pm.Model() as model:
        beta0 = pm.Normal('beta0', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=10)
        beta2 = pm.Normal('beta2', mu=0, sigma=10)

        mu = beta0 + beta1 * X_bayesian_normalized[:, 0] + beta2 * X_bayesian_normalized[:, 1]

        sigma = pm.HalfNormal('sigma', sigma=1)

        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_bayesian)

        trace = pm.sample(1000, tune=1000, return_inferencedata=True)

        return trace


if __name__ == '__main__':
    trace = run()
    az.plot_trace(trace)
    plt.savefig('trace_plot.png')

    summary = az.summary(trace)
    print(summary)
