# https://towardsdatascience.com/machine-learning-part-17-boosting-algorithms-adaboost-in-python-d00faac6c464
# https://www.datacamp.com/tutorial/adaboost-classifier-python

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np


# Create a dictionary of AdaBoost models with varying max_depths of DecisionTree base estimator
def get_models():
    models = {}
    for i in range(1, 11):
        base = DecisionTreeClassifier(max_depth=i)
        models[str(i)] = AdaBoostClassifier(base_estimator=base)
    return models


# Evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


# Train and evaluate an AdaBoost classifier using GridSearchCV for hyperparameter tuning
def train_and_evaluate_adaboost(X_train, X_test, y_train, y_test, base_estimator=None, param_grid=None):
    if base_estimator is None:
        base_estimator = AdaBoostClassifier()

    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 1]
        }

    grid_search = GridSearchCV(estimator=AdaBoostClassifier(base_estimator),
                               param_grid=param_grid,
                               scoring='accuracy',
                               cv=5)

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    best_accuracy = accuracy_score(y_test, y_pred)

    return best_accuracy, grid_search.best_params_


def main():
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Train and evaluate AdaBoost with default base estimator
    default_accuracy, default_best_params = train_and_evaluate_adaboost(X_train, X_test, y_train, y_test)
    print("Default AdaBoost Accuracy:", default_accuracy)
    print("Best Parameters for Default AdaBoost:", default_best_params)

    # Train and evaluate AdaBoost with Support Vector Classifier as base estimator
    svc = SVC(probability=True, kernel='linear')
    svc_accuracy, svc_best_params = train_and_evaluate_adaboost(X_train, X_test, y_train, y_test, base_estimator=svc)
    print("AdaBoost with SVC Accuracy:", svc_accuracy)
    print("Best Parameters for AdaBoost with SVC:", svc_best_params)

    # Get the models to evaluate
    models = get_models()

    # Evaluate the models and store results
    results, names = [], []
    for name, model in models.items():
        scores = evaluate_model(model, X, y)
        results.append(scores)
        names.append(name)
        print(f'>{name} Mean Accuracy: {np.mean(scores):.3f} (Std: {np.std(scores):.3f})')

    # Plot model performance for comparison using boxplots
    plt.boxplot(results, labels=names, showmeans=True)
    plt.xlabel('Max Depth of Decision Tree (Weak Learner)')
    plt.ylabel('Accuracy')
    plt.title('AdaBoost Performance with Different Weak Learner Depths')
    plt.show()


if __name__ == "__main__":
    main()
