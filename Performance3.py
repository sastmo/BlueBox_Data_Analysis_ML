# https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
# https://python.plainenglish.io/how-to-improve-your-classification-models-with-threshold-tuning-db35c31bf018
# https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc
# https://chat.openai.com/share/28592dc0-f454-4ac8-9183-1117867262d9

# Import required libraries
import numpy as np
from matplotlib import pyplot
from numpy import sqrt, argmax, arange
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, accuracy_score, f1_score, \
    precision_recall_curve
import matplotlib.pyplot as plt

# Generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)

# Split into train/test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)

# Fit a logistic regression model
model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)

# Predict probabilities
y_probs = model.predict_proba(testX)

# Keep probabilities for the positive outcome only
y_positive_probs = y_probs[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(testy, y_positive_probs)
roc_auc = auc(fpr, tpr)

# Unbalanced Data set
# Calculate the G-Mean for each threshold
gmeans = sqrt(tpr * (1 - fpr))

# Locate the index of the largest G-Mean
best_threshold_idx = argmax(gmeans)
best_threshold = thresholds[best_threshold_idx]

# Balanced Data set
'''# Calculate F1-score for each threshold
f1_scores = [f1_score(testy, (y_probs[:, 1] >= t).astype(int)) for t in thresholds]

# Locate the index of the largest F1-score
best_threshold_idx_bl = np.argmax(f1_scores)
best_threshold_bl = thresholds[best_threshold_idx_bl]

print('Best Threshold for F1-Score Optimization:', best_threshold_bl)
'''
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[best_threshold_idx], gmeans[best_threshold_idx]), "\n")

# Plot the ROC curve for the model
plt.figure()
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
plt.plot(fpr, tpr, marker='.', label=f'Logistic (AUC = {roc_auc:.2f})')
plt.scatter(fpr[best_threshold_idx], tpr[best_threshold_idx], marker='o', color='black',
            label=f'Best Threshold (G-Mean = {gmeans[best_threshold_idx]:.2f})')
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.scatter(fpr[best_threshold_idx_bl], tpr[best_threshold_idx_bl], color='red', marker='o', label='Best Threshold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Calculate confusion matrix
y_pred = model.predict(testX)
conf_matrix = confusion_matrix(testy, y_pred)

print("Confusion Matrix:")
print(conf_matrix, "\n")

# Calculate accuracy score
accuracy = accuracy_score(testy, y_pred)
print(f"Accuracy: {accuracy:.6f}")

# calculate Youden's J statistic
J = tpr - fpr
# get the best threshold index
ix = argmax(J)
best_thresh = thresholds[ix]
print('Best Threshold=%f' % best_thresh, "\n")

# Optimal Threshold for Precision-Recall Curve ***************************************************

# calculate pr-curve
precision, recall, thresholds = precision_recall_curve(testy, y_positive_probs)
# calculate F1-score
fscores = (2 * precision * recall) / (precision + recall)
# locate the index of the largest F1-score
ix = argmax(fscores)
best_thresh = thresholds[ix]
print("precision * recall-->>")
print('Best Threshold=%f' % best_thresh, "\n")

# plot the precision-recall curve for the model
no_skill = len(testy[testy == 1]) / len(testy)
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
pyplot.plot(recall, precision, marker='.', label='Logistic')
pyplot.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')

# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
pyplot.legend()
# show the plot
pyplot.show()


# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')


# define thresholds
thresholds = arange(0, 1, 0.001)
# evaluate each threshold
scores = [f1_score(testy, to_labels(y_positive_probs, t)) for t in thresholds]
# get best threshold
ix = argmax(scores)
best_threshold = thresholds[ix]
best_f1_score = scores[ix]
print('Best Threshold=%.3f, F-Score=%.5f' % (best_threshold, best_f1_score))

# plot F1-Score vs. Threshold
plt.plot(thresholds, scores, marker='.')
plt.xlabel('Threshold')
plt.ylabel('F1-Score')
plt.title('F1-Score vs. Threshold')
plt.axvline(x=best_threshold, color='r', linestyle='--', label='Best Threshold')
plt.legend()
plt.show()

