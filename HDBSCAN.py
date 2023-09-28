# https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html
# https://www.youtube.com/watch?v=dGsxd67IFiU

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.datasets as data

# Set up plotting parameters
sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha': 0.5, 's': 80, 'linewidths': 0}

# Generate some sample data for clustering
moons, _ = data.make_moons(n_samples=50, noise=0.05)
blobs, _ = data.make_blobs(n_samples=50, centers=[(-0.75, 2.25), (1.0, 2.0)], cluster_std=0.25)
test_data = np.vstack([moons, blobs])

# Visualize the generated data
plt.scatter(test_data.T[0], test_data.T[1], color='b', **plot_kwds)
plt.title("Sample Data")
plt.show()

# Import HDBSCAN and perform clustering
import hdbscan

# Create an HDBSCAN clusterer with specified parameters
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)

# Fit the clusterer to the test data
clusterer.fit(test_data)

# Plot the minimum spanning tree of the clusterer
clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                      edge_alpha=0.6,
                                      node_size=80,
                                      edge_linewidth=2)
plt.title("Minimum Spanning Tree")
plt.show()

# Plot the cluster hierarchy
clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
plt.title("Cluster Hierarchy")
plt.show()

# Plot the condensed cluster tree
clusterer.condensed_tree_.plot()
plt.title("Condensed Cluster Tree")
plt.show()

# Select clusters based on persistence
clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
plt.title("Selected Clusters Based on Persistence")
plt.show()

# Visualize the clusters with membership strength
palette = sns.color_palette()
cluster_colors = [sns.desaturate(palette[col], sat)
                  if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
                  zip(clusterer.labels_, clusterer.probabilities_)]
plt.scatter(test_data.T[0], test_data.T[1], c=cluster_colors, **plot_kwds)
plt.title("Cluster Visualization with Membership Strength")
plt.show()

# Print cluster labels including noise
for label in np.unique(clusterer.labels_):
    if label == -1:
        print(f"Noise points: {np.sum(clusterer.labels_ == label)}")
    else:
        print(f"Cluster {label}: {np.sum(clusterer.labels_ == label)} points")


# Create a DataFrame to store data points and cluster labels
df = pd.DataFrame({'X': test_data[:, 0], 'Y': test_data[:, 1], 'Cluster_Label': clusterer.labels_})

# Print the DataFrame
print(df)
