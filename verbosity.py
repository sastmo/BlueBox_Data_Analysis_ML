from sklearn.manifold import MDS
import numpy as np

# Create a random dissimilarity matrix for demonstration
np.random.seed(42)
n_samples = 5
dissimilarity_matrix = np.random.rand(n_samples, n_samples)

# Initialize MDS with verbose set to different values
verbose_levels = [0, 1, 2]

for verbose_level in verbose_levels:
    print(f"Running MDS with verbose level {verbose_level}")

    mds = MDS(n_components=2, verbose=verbose_level)
    embedded_data = mds.fit_transform(dissimilarity_matrix)

    print("Embedded Data:")
    print(embedded_data)
    print("-" * 30)
