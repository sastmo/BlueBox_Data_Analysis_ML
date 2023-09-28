# https://www.youtube.com/watch?v=589nCGeWG1w&list=PL9q4h4aGdqQ5eCT5N21Eb0QMe5Ksus14X&index=74
# https://chat.openai.com/share/b9f83dab-eae3-45fe-a133-035e3ce8128c

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import LabelEncoder

# Create a synthetic dataset
data = {'Category': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B', 'C'],
        'Feature1': [10, 20, 15, 25, 12, 30, 18, 22, 28],
        'Feature2': [0.5, 0.8, 0.7, 0.6, 0.9, 0.4, 0.3, 0.6, 0.8],
        'Target': [1, 0, 1, 1, 0, 1, 0, 1, 1]}
df = pd.DataFrame(data)

# One-Hot Encoding ****************************************************
one_hot_encoded = pd.get_dummies(df, columns=['Category'], prefix='OH')

print(one_hot_encoded)

# Label Encoding ******************************************************
label_encoder = LabelEncoder()
df['LabelEncoded'] = label_encoder.fit_transform(df['Category'])

print(df)

# Target Encoding ****************************************************
target_means = df.groupby('Category')['Target'].mean()
df['TargetEncoded'] = df['Category'].map(target_means)

print(df)

# Bayesian Mean Target Encoding *************************************

# Overall mean
overall_mean = df['Target'].mean()

# Custom-defined weight for overall mean
m = 2  # Weight for Overall Mean

# Overall mean
overall_mean = df['Target'].mean()

# Calculate n as the count of occurrences for each category
n_values = df['Category'].value_counts().to_dict()
n_values = {category: n_values.get(category, 0) for category in df['Category']}

# Calculate Bayesian Mean Target Encoding using a loop
bayesian_encoded = []
for category in df['Category']:
    n = n_values[category]
    category_mean = df[df['Category'] == category]['Target'].mean()
    weighted_mean = (n * category_mean + m * overall_mean) / (n + m)
    bayesian_encoded.append(weighted_mean)

df['BayesianTargetEncoded'] = bayesian_encoded

print(df)

# K-Fold Target Encoding ****************************************

# Custom-defined weight for optional mean
n = 2  # Weight for Optional Mean (e.g., category mean)
m = 2  # Weight for Overall Mean

# K-Fold Target Encoding
kf = KFold(n_splits=3, shuffle=True, random_state=42)
df['KFoldTargetEncoded'] = np.nan
for train_idx, val_idx in kf.split(df):
    train_weights = df.iloc[train_idx].groupby('Category')['Target'].count()
    train_means = df.iloc[train_idx].groupby('Category')['Target'].mean()
    overall_mean = df.iloc[train_idx]['Target'].mean()  # Calculate overall mean for the current fold
    for idx in val_idx:
        category = df.loc[idx, 'Category']
        n = train_weights.get(category, 0)
        weighted_mean = (n * train_means.get(category, overall_mean) +
                         m * overall_mean) / (n + m)
        df.loc[idx, 'KFoldTargetEncoded'] = weighted_mean

print(df)

# Leave-One-Out Target Encoding  ****************************************
loo = LeaveOneOut()
df['LOOTargetEncoded'] = np.nan
for train_idx, val_idx in loo.split(df):
    train_weights = df.iloc[train_idx].groupby('Category')['Target'].count()
    train_means = df.iloc[train_idx].groupby('Category')['Target'].mean()
    overall_mean = df.iloc[train_idx]['Target'].mean()  # Calculate overall mean for the current fold
    for idx in val_idx:
        category = df.loc[idx, 'Category']
        n = train_weights.get(category, 0)
        weighted_mean = (n * train_means.get(category, overall_mean) +
                         m * overall_mean) / (n + m)
        df.loc[idx, 'LOOTargetEncoded'] = weighted_mean

print(df)
