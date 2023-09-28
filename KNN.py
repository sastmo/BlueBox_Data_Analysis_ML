# https://www.analyticsvidhya.com/blog/2018/08/k-nearest-neighbor-introduction-regression-python/
# https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn
# https://chat.openai.com/share/03ca700a-0b93-4323-83e7-ecd315557155

# Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Reading the dataset
loan_data = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\Binary_target.csv")

loan_dataset = pd.DataFrame(loan_data)
print(loan_dataset.columns)
# Data Pre-Processing


# 1. Checking for missing values
missing_values = loan_dataset.isna().sum()
print(missing_values)

# 2. Filling missing values
fill_values = {
    'Gender': loan_dataset['Gender'].mode().values[0],
    'Married': loan_dataset['Married'].mode().values[0],
    'Dependents': loan_dataset['Dependents'].mode().values[0],
    'Self_Employed': loan_dataset['Self_Employed'].mode().values[0],
    'LoanAmount': loan_dataset['LoanAmount'].mean(),
    'Loan_Amount_Term': loan_dataset['Loan_Amount_Term'].mode().values[0],
    'Credit_History': loan_dataset['Credit_History'].mode().values[0]
}
loan_dataset.fillna(fill_values, inplace=True)

# 3. Removing the Id column
loan_dataset.drop('Loan_ID', axis=1, inplace=True)

# Exploratory Data Analysis
'''
# Gender vs Loan Approval
plt.figure(figsize=(10, 6))
sns.countplot(x='Gender', hue='Loan_Status', data=loan_dataset)
plt.title("Loan Approval by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# Marital Status vs Loan Approval
plt.figure(figsize=(10, 6))
sns.countplot(x='Married', hue='Loan_Status', data=loan_dataset)
plt.title("Loan Approval by Marital Status")
plt.xlabel("Marital Status")
plt.ylabel("Count")
plt.show()

# Education vs Loan Approval
plt.figure(figsize=(10, 6))
sns.countplot(x='Education', hue='Loan_Status', data=loan_dataset)
plt.title("Loan Approval by Education")
plt.xlabel("Education")
plt.ylabel("Count")
plt.show()

# Self-Employment vs Loan Approval
plt.figure(figsize=(10, 6))
sns.countplot(x='Self_Employed', hue='Loan_Status', data=loan_dataset)
plt.title("Loan Approval by Self-Employment")
plt.xlabel("Self-Employed")
plt.ylabel("Count")
plt.show()

# Credit History vs Loan Approval
plt.figure(figsize=(10, 6))
sns.countplot(x='Credit_History', hue='Loan_Status', data=loan_dataset)
plt.title("Loan Approval by Credit History")
plt.xlabel("Credit History")
plt.ylabel("Count")
plt.show()

# Property Area vs Loan Approval
plt.figure(figsize=(10, 6))
sns.countplot(x='Property_Area', hue='Loan_Status', data=loan_dataset)
plt.title("Loan Approval by Property Area")
plt.xlabel("Property Area")
plt.ylabel("Count")
plt.show()

# Loan Approval Status
plt.figure(figsize=(6, 4))
loan_dataset['Loan_Status'].value_counts().plot(kind='bar')
plt.title("Loan Approval Status")
plt.xlabel("Loan Status")
plt.ylabel("Count")
plt.show()

# Applicant Income Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=loan_dataset, x='ApplicantIncome', kde=True)
plt.title("Applicant Income Distribution")
plt.xlabel("Applicant Income")
plt.ylabel("Density")
plt.show()

# Applicant Income Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x=loan_dataset['ApplicantIncome'])
plt.title("Applicant Income Box Plot")
plt.xlabel("Applicant Income")
plt.show()

# Coapplicant Income Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=loan_dataset, x='CoapplicantIncome', kde=True)
plt.title("Coapplicant Income Distribution")
plt.xlabel("Coapplicant Income")
plt.ylabel("Density")
plt.show()

# Coapplicant Income Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x=loan_dataset['CoapplicantIncome'])
plt.title("Coapplicant Income Box Plot")
plt.xlabel("Coapplicant Income")
plt.show()

# Loan Amount Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=loan_dataset, x='LoanAmount', kde=True)
plt.title("Loan Amount Distribution")
plt.xlabel("Loan Amount")
plt.ylabel("Density")
plt.show()

# Loan Amount Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x=loan_dataset['LoanAmount'])
plt.title("Loan Amount Box Plot")
plt.xlabel("Loan Amount")
plt.show()

# Correlation Matrix
plt.figure(figsize=(12, 8))
correlation_matrix = loan_dataset.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()
'''

# Step 3: Deal with categorical variables and drop the id columns
loan_dataset.drop(['Gender', 'Dependents', 'Education', 'Self_Employed'], axis=1,
                  inplace=True)

print(loan_dataset.columns)

# Selecting columns for one-hot encoding
columns_for_encoding = ['Married', 'Credit_History', 'Property_Area', 'Loan_Status']

# Perform one-hot encoding using pandas get_dummies
loan_dataset_encoded = pd.get_dummies(loan_dataset, columns=columns_for_encoding, drop_first=True)

# Select only numeric columns
numeric_columns = loan_dataset_encoded.select_dtypes(include=['number'])

# Filter out rows with any non-numeric values
loan_dataset_encoded_numeric = loan_dataset_encoded.dropna(subset=numeric_columns.columns, how='any')

# Step 4: Create a train and a test set
x = loan_dataset_encoded_numeric.iloc[:, :-1]
y = loan_dataset_encoded_numeric.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25, random_state=38, stratify=y)

# Step 5: Preprocessing – Scaling the features
# List of columns with numerical features for scaling
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Scale only the numerical features
X_train_scaled[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test_scaled[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Combine scaled numerical columns with non-numeric categorical columns
X_train = pd.concat([X_train_scaled, X_train.select_dtypes(include=['object'])], axis=1)
X_test = pd.concat([X_test_scaled, X_test.select_dtypes(include=['object'])], axis=1)


# Step 6: Let us have a look at the error rate for different k values

# 1. K Curve
def calculate_k_curve(X_train, X_test, Y_train, Y_test):
    rmse_val = []
    for K in range(1, 21):
        model = KNeighborsRegressor(n_neighbors=K)
        model.fit(X_train, Y_train)
        pred = model.predict(X_test)
        error = np.sqrt(mean_squared_error(Y_test, pred))
        rmse_val.append(error)
        print('RMSE value for k =', K, 'is:', error)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 21), rmse_val, marker='o')
    plt.title('RMSE vs K Value')
    plt.xlabel('K')
    plt.ylabel('RMSE')
    plt.show()


# 2. Using GridSearchCV
# Define the parameter grid
def find_best_k_using_grid_search(X_train, Y_train):
    params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]}
    knn = KNeighborsRegressor()
    grid_search = GridSearchCV(knn, params, cv=5)
    grid_search.fit(X_train, Y_train)
    best_params = grid_search.best_params_
    print("Best parameters:", best_params)


# 3. Using Accuracy and Cross Validation

def plot_accuracy_curve(X, y, k_values):
    scores = []

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn, X, y, cv=5)
        scores.append(np.mean(score))

    sns.lineplot(x=k_values, y=scores, marker='o')
    plt.xlabel("K Values")
    plt.ylabel("Accuracy Score")
    plt.title("Accuracy vs K Value")
    plt.show()


calculate_k_curve(X_train, X_test, Y_train, Y_test)
find_best_k_using_grid_search(X_train, Y_train)
k_values = [i for i in range(1, 31)]
plot_accuracy_curve(X_train, Y_train, k_values)

# Model Building - KNN
optimized_k = 6
print("Optimal K value:", optimized_k)
knn = KNeighborsClassifier(n_neighbors=optimized_k)
knn.fit(X_train, Y_train)
prediction_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(Y_test, prediction_knn)

# Model Building - Decision Tree
dTree = tree.DecisionTreeClassifier()
dTree.fit(X_train, Y_train)
prediction_dt = dTree.predict(X_test)
accuracy_dt = accuracy_score(Y_test, prediction_dt)

# Displaying results
print('KNN Accuracy: {:.2f}%'.format(accuracy_knn * 100))
print('Decision Tree Accuracy: {:.2f}%'.format(accuracy_dt * 100))


# Printing confusion matrices and classification reports
def display_metrics(y_true, y_pred, title):
    con_mat = confusion_matrix(y_true, y_pred)

    # Create a custom confusion matrix with "True" and "False" labels
    labels = ['True', 'False']
    con_mat_df = pd.DataFrame(con_mat, index=labels, columns=labels)

    sns.heatmap(con_mat_df, annot=True, fmt="d")
    plt.title(title + ' Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    print(classification_report(y_true, y_pred))


display_metrics(Y_test, prediction_knn, 'KNN')
display_metrics(Y_test, prediction_dt, 'Decision Tree')
