# https://chat.openai.com/share/425b05de-61cf-4523-9da7-4bc6bb8ad446

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

# Step 2: Use SVD for PCA
U, S, Vt = np.linalg.svd(X_standardized, full_matrices=False)

# Step 3: LOOCV for optimal number of principal components
loo = LeaveOneOut()
explained_variances = []
for i in range(1, X.shape[1] + 1):
    pca = PCA(n_components=i)
    variances = []
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X_standardized[train_idx], X_standardized[test_idx]
        pca.fit(X_train)
        variances.append(pca.explained_variance_ratio_[i - 1])
    explained_variances.append(np.mean(variances))

# Step 4: Scree plot (bar chart)
plt.bar(range(1, X.shape[1] + 1), explained_variances)
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

    # Calculate mean and covariance matrix of each category
    mean = np.mean(X_pca[data.target == target, :], axis=0)
    cov_matrix = np.cov(X_pca[data.target == target, :].T)

    # Create and plot ellipse or rectangle
    if num_components == 2:  # For 2D PCA space
        ellipse = Ellipse(mean, width=np.sqrt(cov_matrix[0, 0]) * 2, height=np.sqrt(cov_matrix[1, 1]) * 2, fill=False,
                          color=color, linestyle='dashed')
        plt.gca().add_patch(ellipse)
    else:
        # Create a bounding box for higher-dimensional PCA
        rect = plt.Rectangle((mean[0] - 0.5, mean[1] - 0.5), width=1, height=1, fill=False, color=color,
                             linestyle='dashed')
        plt.gca().add_patch(rect)

plt.xlabel(feature_names[0] + "[PC1]")  # Use the first original variable name
plt.ylabel(feature_names[1] + "[PC2]")  # Use the second original variable name
plt.title('Data Plot in PCA Space with Ellipses/Bounding Boxes')
plt.legend()
plt.show()

# Print proportion of explained variance
print(f'Proportion of Explained Variance (PCA with {num_components} components):')
print(pca.explained_variance_ratio_)
