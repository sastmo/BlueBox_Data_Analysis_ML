import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

# create DataFrane
df = pd.DataFrame({'player': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'],
                   'points': [4, 4, 6, 7, 8, 14, 16, 19, 25, 25, 28],
                   'assists': [3, 2, 2, 5, 4, 8, 7, 6, 8, 10, 11],
                   'blocks': [7, 3, 6, 7, 5, 8, 8, 4, 2, 2, 1],
                   'rebounds': [4, 5, 5, 6, 5, 8, 10, 4, 3, 2, 2]})

# set player column as index column
df = df.set_index('player')

# view Dataframe
# print(df)

# perform multi-dimensional scaling
mds = MDS(random_state=0)
scaled_df = mds.fit_transform(df)

# view results of multi-dimensional scaling
# print(scaled_df)

# create scatterplot
plt.figure(figsize=(10, 6))
plt.scatter(scaled_df[:, 0], scaled_df[:, 1])

# add axis labels
plt.xlabel('PoC 1')
plt.ylabel('PoC 2')

# add lables to each point
for i, txt in enumerate(df.index):
    plt.annotate(txt, (scaled_df[:, 0][i] + .3, scaled_df[:, 1][i]))

# display scatterplot
plt.show()

# select rows with index labels 'F' and 'G'
print(df.loc[['F', 'G']])

# select rows with index labels 'B' and 'K'
print(df.loc[['B', 'K']])

'''# Distance matrix
dist_matrix = np.array([[0.0, 0.2, 0.5],
                        [0.2, 0.0, 0.4],
                        [0.5, 0.4, 0.0]])

# Convert distance matrix to dissimilarity matrix
dissimilarity_matrix = 1 - dist_matrix

# Perform PCoA using MDS
mds = MDS(n_components=2, dissimilarity='precomputed')
coordinates = mds.fit_transform(dissimilarity_matrix)

# Plot the results
plt.scatter(coordinates[:, 0], coordinates[:, 1])
for i, label in enumerate(['A', 'B', 'C']):
    plt.text(coordinates[i, 0], coordinates[i, 1], label)
plt.xlabel('PCoA 1')
plt.ylabel('PCoA 2')
plt.title('PCoA of Genetic Distances')
plt.show()
'''
