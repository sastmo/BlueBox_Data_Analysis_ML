# https://chat.openai.com/share/425b05de-61cf-4523-9da7-4bc6bb8ad446

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Step 1: Load Iris dataset
data = load_iris()
X = data.data

# Step 5: Center the data using mean
X_centered = X - np.mean(X, axis=0)

# Step 6: Standardize features
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X_centered)

# Step 2: Use PLSRegression for PCA (NIPALS-like algorithm)
num_components = X.shape[1]  # Set the number of components

# Create a placeholder target variable for fitting
y_placeholder = np.zeros((X.shape[0], 1))  # Replace with actual target if available

# Create a PLSRegression model with the number of components
pls_model = PLSRegression(n_components=num_components)

# Apply PCA with selected number of components
pca = PCA(n_components=num_components)  # Creating an instance of PCA class
X_pca = pca.fit_transform(X_standardized)

# Step 4: Scree plot (bar chart)
explained_variances = (pca.explained_variance_ / np.sum(pca.explained_variance_))
cumulative_explained_variances = np.cumsum(explained_variances)
plt.bar(range(1, X.shape[1]+1), explained_variances)
# plt.plot(range(1, X.shape[1]+1), cumulative_explained_variances, marker='o', color='red')
plt.xlabel('Number of Principal Components')
plt.ylabel('Proportion of Explained Variance')
plt.title('Scree Plot')
plt.show()

# Step 7: Select number of PCA based on scree plot
num_components = 2  # Based on the elbow in the scree plot

# Apply PCA with selected number of components
pca = PCA(n_components=num_components)
X_pca = pca.fit_transform(X_standardized)

# Plot data based on selected principal components with legend and real names
colors = ['r', 'g', 'b']
species = ['Setosa', 'Versicolor', 'Virginica']
feature_names = data.feature_names

for target, color, species_name in zip(range(len(species)), colors, species):
    plt.scatter(X_pca[data.target == target, 0], X_pca[data.target == target, 1], color=color, label=species_name)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Data Plot in PCA Space')
plt.legend()
plt.show()

# Print proportion of explained variance
print(f'Proportion of Explained Variance (PCA with {num_components} components):')
print(pca.explained_variance_ratio_)

# Get the coefficients (loadings) of variables in PC1
loadings_pc1 = pca.components_[0]

# Create a list of variable names
variable_names = data.feature_names

# Create a dictionary to store variable names and their loadings
variable_loadings = dict(zip(variable_names, loadings_pc1))

# Sort variables by their absolute loadings
sorted_loadings = sorted(variable_loadings.items(), key=lambda x: abs(x[1]), reverse=True)

# Print the variables and their loadings for PC1
for variable, loading in sorted_loadings:
    print(f"{variable}: {loading:.4f}")