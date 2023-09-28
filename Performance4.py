import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Example predicted probabilities
predicted_probs = np.array([0.65, 0.72, 0.58, 0.85, 0.42])

# Example threshold determined through tuning
threshold = 0.7

# Apply the threshold to get revised predictions
revised_predictions = (predicted_probs >= threshold).astype(int)

# Example true labels
true_labels = np.array([1, 1, 0, 1, 0])

# Calculate evaluation metrics using the revised predictions
precision = precision_score(true_labels, revised_predictions)
recall = recall_score(true_labels, revised_predictions)
f1 = f1_score(true_labels, revised_predictions)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

import numpy as np
from sklearn.metrics import fbeta_score

# Example predicted probabilities
predicted_probs = np.array([0.65, 0.72, 0.58, 0.85, 0.42])

# Example true labels
true_labels = np.array([1, 1, 0, 1, 0])

# Define beta for F-beta score
beta = 2  # Adjust beta based on your preference (higher value prioritizes Recall)

# Define a range of threshold values to search
thresholds = np.linspace(0, 1, 100)

# Initialize variables to track the maximum F-beta score and the corresponding threshold
max_fbeta_score = 0
best_threshold = 0

# Iterate through thresholds and calculate F-beta scores
for threshold in thresholds:
    predictions = (predicted_probs >= threshold).astype(int)
    fbeta = fbeta_score(true_labels, predictions, beta=beta)  # Pass beta as a keyword argument
    if fbeta > max_fbeta_score:
        max_fbeta_score = fbeta
        best_threshold = threshold

print(f"Best Threshold: {best_threshold:.2f}")
print(f"Max F-beta Score: {max_fbeta_score:.2f}")

from sklearn.metrics import confusion_matrix

# Example confusion matrix
conf_matrix = [[900, 50],  # TP, FN
               [100, 9500]]  # FP, TN

# Define costs
C_FP = 100  # False Positive Cost
C_FN = 50  # False Negative Cost

# Calculate expected costs
expected_cost = (conf_matrix[0][0] * 0 +
                 conf_matrix[0][1] * C_FN +
                 conf_matrix[1][0] * C_FP +
                 conf_matrix[1][1] * 0)

print("Expected Cost:", expected_cost)

from sklearn.metrics import confusion_matrix

# Example confusion matrix
conf_matrix = [[100, 10],  # TP, FN
               [5, 1000]]  # FP, TN

# Define costs
C_FP = 200  # False Positive Cost
C_FN = 5000  # False Negative Cost

# Calculate expected costs for different threshold values
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for threshold in thresholds:
    # Calculate TP, FN, FP, TN based on the threshold
    predictions = (predicted_probs > threshold).astype(int)
    conf_matrix = confusion_matrix(true_labels, predictions)

    # Calculate expected cost
    expected_cost = (conf_matrix[0][0] * 0 +
                     conf_matrix[0][1] * C_FN +
                     conf_matrix[1][0] * C_FP +
                     conf_matrix[1][1] * 0)

    print(f"Threshold: {threshold}, Expected Cost: {expected_cost}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, fbeta_score

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Calculate predicted probabilities on the validation set
probs = model.predict_proba(X_val)[:, 1]

# Define costs
C_FP = 100  # False Positive Cost
C_FN = 50  # False Negative Cost

# Calculate precision and recall
precision, recall, thresholds = precision_recall_curve(y_val, probs)

# Calculate expected costs for different thresholds
expected_costs = (C_FP * (1 - precision) * recall + C_FN * precision * (1 - recall))

# Find the threshold that minimizes the expected cost
optimal_threshold_idx = np.argmin(expected_costs)
optimal_threshold = thresholds[optimal_threshold_idx]

# Calculate confusion matrix with the optimal threshold
predictions = (probs >= optimal_threshold).astype(int)
conf_matrix = confusion_matrix(y_val, predictions)

# Calculate F-beta score with the optimal threshold
beta = 2  # Adjust based on your preference
fbeta = fbeta_score(y_val, predictions, beta=beta)

# Print results
print("Optimal Threshold:", optimal_threshold)
print("Confusion Matrix:\n", conf_matrix)
print("F-beta Score:", fbeta)

# Plot precision-recall curve
plt.plot(recall, precision, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()
