# https://towardsdatascience.com/an-introduction-to-t-sne-with-python-example-5a3a293108d1
# https://www.datacamp.com/tutorial/introduction-t-sne

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Generate custom classification dataset
X, y = make_classification(
    n_features=6,
    n_classes=3,
    n_samples=1500,
    n_informative=2,
    random_state=5,
    n_clusters_per_class=1,
)

# 3D Scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', alpha=0.8)
plt.title("3D Scatter plot of Custom Classification dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
ax.set_zlabel("Feature 3")
plt.colorbar(scatter)
plt.show()

# PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure()
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title("PCA visualization of Custom Classification dataset")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.colorbar(scatter)
plt.show()

# t-SNE visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure()
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.title("t-SNE visualization of Custom Classification dataset")
plt.xlabel("First t-SNE")
plt.ylabel("Second t-SNE")
plt.colorbar(scatter)
plt.show()

# Load and preprocess Customer Churn dataset

# Reading the dataset
df = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\Churn.csv")
print(df.columns)

y = df['churn']
X = df.drop(['churn', 'phone number'], axis=1)

# One-hot encode categorical features (e.g., 'state')
X_encoded = pd.get_dummies(X, columns=['international plan', 'voice mail plan'], drop_first=True)

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_encoded)
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, random_state=13, test_size=0.25, shuffle=True
)

# PCA visualization for Customer Churn dataset
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

plt.figure()
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis')
plt.title("PCA visualization of Customer Churn dataset")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.colorbar(scatter)
plt.show()

# t-SNE divergence plot
perplexity = np.arange(5, 55, 5)
divergence = []

for i in perplexity:
    model = TSNE(n_components=2, init="pca", perplexity=i)
    reduced = model.fit_transform(X_train)
    divergence.append(model.kl_divergence_)

plt.figure()
plt.plot(perplexity, divergence, marker='o', color='red')
plt.title("t-SNE Divergence vs Perplexity")
plt.xlabel("Perplexity Values")
plt.ylabel("Divergence")
plt.grid(True)
plt.show()

# t-SNE visualization for Customer Churn dataset
tsne = TSNE(n_components=2, perplexity=40, random_state=42)
X_train_tsne = tsne.fit_transform(X_train)

plt.figure()
scatter = plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=y_train, cmap='viridis')
plt.title("t-SNE visualization of Customer Churn dataset")
plt.xlabel("First t-SNE")
plt.ylabel("Second t-SNE")
plt.colorbar(scatter)
plt.show()

# Reading the dataset
df = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\Churn.csv")
print(df.columns)

y = df['churn']
X = df.drop(['churn', 'phone number'], axis=1)  # Remove 'churn' and 'phone number' columns

# Label encoding for categorical columns
label_encoder = LabelEncoder()
for column in ['state', 'international plan', 'voice mail plan']:
    X[column] = label_encoder.fit_transform(X[column])

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, random_state=13, test_size=0.25, shuffle=True
)

# PCA visualization for Customer Churn dataset
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
print(pca.score(X_test))

plt.figure()
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis')
plt.title("PCA visualization of Customer Churn dataset")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.legend(['Not Churned', 'Churned'])

plt.colorbar(scatter)
plt.show()

# t-SNE divergence plot
perplexity = np.arange(5, 80, 5)
divergence = []

for i in perplexity:
    model = TSNE(n_components=2, init="pca", perplexity=i)
    reduced = model.fit_transform(X_train)
    divergence.append(model.kl_divergence_)
plt.figure()
plt.plot(perplexity, divergence, marker='o', color='red')
plt.title("t-SNE Divergence vs Perplexity")
plt.xlabel("Perplexity Values")
plt.ylabel("Divergence")
plt.legend(['Not Churned', 'Churned'])
plt.grid(True)
plt.show()

# t-SNE visualization for Customer Churn dataset
tsne = TSNE(n_components=2, perplexity=100, random_state=42)
X_train_tsne = tsne.fit_transform(X_train)
print(tsne.kl_divergence_)

# Reading the dataset
df = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\2022-Fall\DATA\Phyton\Practice\Churn.csv")
print(df.columns)

y = df['churn']
X = df.drop(['churn', 'phone number'], axis=1)  # Remove 'churn' and 'phone number' columns

# Label encoding for categorical columns
label_encoder = LabelEncoder()
for column in ['state', 'international plan', 'voice mail plan']:
    X[column] = label_encoder.fit_transform(X[column])

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, random_state=13, test_size=0.25, shuffle=True
)

# Perform KMeans clustering on the entire dataset X
num_clusters = 4  # You can adjust the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X_norm)

# t-SNE visualization for Customer Churn dataset
tsne = TSNE(n_components=2, perplexity=100, random_state=42)
X_tsne = tsne.fit_transform(X_norm)

# Get unique cluster IDs
unique_clusters = df['cluster'].unique()

# Visualize t-SNE scatter plot with cluster coloring and legend
plt.figure()
scatter_list = []
for cluster_id in unique_clusters:
    cluster_data = X_tsne[df['cluster'] == cluster_id]
    scatter = plt.scatter(cluster_data[:, 0], cluster_data[:, 1], cmap='viridis')
    scatter_list.append(scatter)

plt.legend(scatter_list, unique_clusters, title='Cluster')
plt.title("t-SNE visualization of Customer Churn dataset")
plt.xlabel("First t-SNE")
plt.ylabel("Second t-SNE")
plt.colorbar()
plt.show()

# Calculate average values for each cluster
cluster_averages = df.groupby('cluster').mean()

# Print cluster averages
print(cluster_averages)

# Create a Random Forest classifier
random_forest = RandomForestClassifier(random_state=42)

# Initialize an empty DataFrame to store feature importances for each cluster
feature_importance_by_cluster = pd.DataFrame(columns=['Cluster'] + X.columns.tolist())

# Iterate through unique cluster IDs
for cluster_id in unique_clusters:
    # Get data points belonging to the current cluster
    cluster_data = X_norm[df['cluster'] == cluster_id]

    # Get corresponding cluster labels
    cluster_labels = df[df['cluster'] == cluster_id]['churn']

    # Train the Random Forest model on the cluster-specific data
    random_forest.fit(cluster_data, cluster_labels)

    # Get feature importances for the current cluster
    cluster_feature_importances = random_forest.feature_importances_

    # Add cluster ID and feature importances to the DataFrame
    feature_importance_by_cluster.loc[len(feature_importance_by_cluster)] = [
                                                                                cluster_id] + cluster_feature_importances.tolist()


# Print the DataFrame with feature importances for each cluster
print(feature_importance_by_cluster)

# Plot feature importances for each cluster
for cluster_id in unique_clusters:
    cluster_importances = feature_importance_by_cluster[feature_importance_by_cluster['Cluster'] == cluster_id].drop('Cluster', axis=1)
    plt.figure(figsize=(10, 6))
    plt.bar(cluster_importances.columns, cluster_importances.values.flatten())
    plt.title(f"Feature Importance for Cluster {cluster_id}")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()