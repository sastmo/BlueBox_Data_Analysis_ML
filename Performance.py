# https://stats.stackexchange.com/questions/210700/how-to-choose-between-roc-auc-and-f1-score#:~:text=ROC%20%2F%20AUC%20is%20the%20same,PR%20but%20not%20ROC%2FAUC.


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

# Data preparation
corpus = [
    'We enjoyed our stay so much. The weather was not great, but everything else was perfect.',
    'Going to think twice before staying here again. The wifi was spotty and the rooms smaller than advertised',
    'The perfect place to relax and recharge.',
    'Never had such a relaxing vacation.',
    'The pictures were misleading, so I was expecting the common areas to be bigger. But the service was good.',
    'There were no clean linens when I got to my room and the breakfast options were not that many.',
    'Was expecting it to be a bit far from historical downtown, but it was almost impossible to drive through those narrow roads',
    'I thought that waking up with the chickens was fun, but I was wrong.',
    'Great place for a quick getaway from the city. Everyone is friendly and polite.',
    'Unfortunately it was raining during our stay, and there weren\'t many options for indoors activities. Everything was great, but there was literally no other oprionts besides being in the rain.',
    'The town festival was postponed, so the area was a complete ghost town. We were the only guests. Not the experience I was looking for.',
    'We had a lovely time. It\'s a fantastic place to go with the children, they loved all the animals.',
    'A little bit off the beaten track, but completely worth it. You can hear the birds sing in the morning and then you are greeted with the biggest, sincerest smiles from the owners. Loved it!',
    'It was good to be outside in the country, visiting old town. Everything was prepared to the upmost detail'
    'staff was friendly. Going to come back for sure.',
    'They didn\'t have enough staff for the amount of guests. It took some time to get our breakfast and we had to wait 20 minutes to get more information about the old town.',
    'The pictures looked way different.',
    'Best weekend in the countryside I\'ve ever had.',
    'Terrible. Slow staff, slow town. Only good thing was being surrounded by nature.',
    'Not as clean as advertised. Found some cobwebs in the corner of the room.',
    'It was a peaceful getaway in the countryside.',
    'Everyone was nice. Had a good time.',
    'The kids loved running around in nature, we loved the old town. Definitely going back.',
    'Had worse experiences.',
    'Surprised this was much different than what was on the website.',
    'Not that mindblowing.'
]  # List of reviews
targets = [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0]  # Corresponding labels (0 or 1)

# Splitting the dataset
train_features, test_features, train_targets, test_targets = train_test_split(corpus, targets, test_size=0.25,
                                                                              random_state=123)

# Turning the corpus into a tf-idf array
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, norm='l1')
train_features = vectorizer.fit_transform(train_features)
test_features = vectorizer.transform(test_features)


# Build Multi-Layer Perceptron with 3 hidden layers, each with 5 neurons
def buildMLPerceptron(train_features, train_targets, num_neurons=2):
    classifier = MLPClassifier(hidden_layer_sizes=num_neurons, max_iter=35, activation='relu', solver='sgd', verbose=10,
                               random_state=762, learning_rate='invscaling')
    classifier.fit(train_features, train_targets)
    return classifier


ml_percetron_model = buildMLPerceptron(train_features, train_targets, num_neurons=5)


# Plotting the ROC curve and calculating AUC
def plot_roc(model, test_features, test_targets):
    random_forests_model = RandomForestClassifier(random_state=42)
    random_forests_model.fit(train_features, train_targets)

    rfc_disp = RocCurveDisplay.from_estimator(random_forests_model, test_features, test_targets)
    model_disp = RocCurveDisplay.from_estimator(model, test_features, test_targets, ax=rfc_disp.ax_)
    model_disp.figure_.suptitle("ROC curve: Multilayer Perceptron vs Random Forests")

    plt.show()


plot_roc(ml_percetron_model, test_features, test_targets)




# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# Generate example dataset
np.random.seed(42)
num_samples = 1000
features = np.random.rand(num_samples, 5)
labels = np.random.randint(2, size=num_samples)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build Logistic Regression model
model = LogisticRegression()

# Fit model
model.fit(X_train_scaled, y_train)

# Predict probabilities
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# ROC Analysis and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Calculate F1-score
threshold = 0.5  # Chosen threshold for classification
y_pred = (y_pred_proba > threshold).astype(int)
f1 = f1_score(y_test, y_pred)

print("F1-Score:", f1)

# Adjust threshold for desired sensitivity and specificity
new_threshold = 0.6  # Adjust threshold
new_y_pred = (y_pred_proba > new_threshold).astype(int)
new_f1 = f1_score(y_test, new_y_pred)

print("F1-Score with New Threshold:", new_f1)

# Cost-sensitive learning
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
model_cost_sensitive = LogisticRegression(class_weight={0: class_weights[0], 1: class_weights[1]})
model_cost_sensitive.fit(X_train_scaled, y_train)

y_pred_proba_cost_sensitive = model_cost_sensitive.predict_proba(X_test_scaled)[:, 1]
cost_sensitive_f1 = f1_score(y_test, y_pred_proba_cost_sensitive > threshold)

print("F1-Score with Cost-Sensitive Learning:", cost_sensitive_f1)
