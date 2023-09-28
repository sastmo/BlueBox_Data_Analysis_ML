# https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205
# https://chat.openai.com/share/3be710a5-ddde-4987-bd97-be4324c23635

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Split the data into training and evaluation sets
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the number of folds for K-fold cross-validation
K = 5
kf = KFold(n_splits=K)

# Initialize an empty DataFrame to store data for each fold
weak_learner_data = pd.DataFrame()

# Train weak learners using K-fold cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    # Create and train a weak learner (decision tree)
    weak_learner = DecisionTreeClassifier(max_depth=2)
    weak_learner.fit(X_fold_train, y_fold_train)

    # Generate predictions for the fold's validation set
    fold_predictions = weak_learner.predict(X_fold_val)

    # Create a DataFrame to store fold data
    fold_data = pd.DataFrame(X_fold_val, columns=[f'feature_{i}' for i in range(X_fold_val.shape[1])])
    fold_data['fold_predictions'] = fold_predictions

    # Append the fold's data to the overall DataFrame
    weak_learner_data = pd.concat([weak_learner_data, fold_data])

# Display the shape of the generated DataFrame
print("Shape of weak_learner_data:", weak_learner_data.shape)

# Extract features and labels for meta-model training
X_meta = weak_learner_data.drop(columns=['fold_predictions'])
y_meta = weak_learner_data['fold_predictions']

# Initialize a Gradient Boosting classifier as the meta-model
meta_model = GradientBoostingClassifier()

# Train the meta-model (Gradient Boosting) on weak learners' predictions
meta_model.fit(X_meta, y_meta)

# Predict using the meta-model
meta_predictions = meta_model.predict(X_eval)

# Evaluate the meta-model's performance
meta_accuracy = accuracy_score(y_eval, meta_predictions)
print(f"Meta-Model Accuracy: {meta_accuracy:.2f}")

# Plot actual values and meta-model predictions
plt.figure(figsize=(10, 6))
plt.plot(y_eval, label='Actual')
plt.plot(meta_predictions, label='Predicted (Meta-Model)')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.title('Actual vs. Predicted (Meta-Model) Values')
plt.legend()
plt.show()
