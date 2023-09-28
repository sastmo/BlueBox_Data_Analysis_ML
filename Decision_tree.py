from idlelib import tree

import graphviz
import pandas as pd  # to load and manipulate data and for One-Hot Encoding
import numpy as np  # to calculate the mean and standard deviation
import matplotlib.pyplot as plt  # to draw graphs
from sklearn.tree import DecisionTreeClassifier  # to build a classification tree
from sklearn.tree import plot_tree  # to draw a classification tree
from sklearn.model_selection import train_test_split  # to split data into \training and testing sets
from sklearn.model_selection import cross_val_score  # for cross validation
from sklearn.metrics import ConfusionMatrixDisplay  # creates and draws confusion matrix
# from sklearn.metrics import plot_confusion_matrix # to draw a confusion matrix

import pandas as pd

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
column_names = ['age', 'sex', 'cp', 'restbp', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
                'thal', 'hd']

df = pd.read_csv(url, header=None)
df.columns = column_names
X_df = df.iloc[:, :-1]

print(X_df.columns)


class DataCleaner:
    """A class to clean a pandas DataFrame.

       Attributes:
           df: A pandas DataFrame to be cleaned.
           missing_values: A pandas DataFrame to record the missing values in each row.
       """

    def __init__(self, df):
        self.df = df
        self.missing_values = pd.DataFrame(columns=["RowID", "MissingValues"])

    def clean_age(self):
        self.df["age"] = self.df["age"].apply(lambda x: max(0.0, min(150.0, x)) if pd.notnull(x) else x).astype(float)

    def clean_sex(self):
        self.df["sex"] = self.df["sex"].apply(lambda x: x == 1.0 if pd.notnull(x) else x).astype(bool)

    def clean_column(self, column_name, valid_values=None, lower_bound=None, upper_bound=None):
        if valid_values is not None:
            self.df[column_name] = self.df[column_name].apply(lambda x: x if x in valid_values else None)
        elif lower_bound is not None and upper_bound is not None:
            self.df[column_name] = self.df[column_name].apply(lambda x: x if lower_bound <= x <= upper_bound else None)

    def clean_data(self):
        columns_to_clean = [
            ("age", None, (0.0, 150.0)),
            ("sex", None, None),
            ("cp", [1.0, 2.0, 3.0, 4.0], None),
            ("restbp", None, (50.0, 250.0)),
            ("chol", None, (100.0, 400.0)),
            ("fbs", [0.0, 1.0], None),
            ("restecg", [1.0, 2.0, 3.0], None),
            ("thalach", None, (70.0, 220.0)),
            ("exang", [0.0, 1.0], None),
            ("oldpeak", None, (0.0, 8.0)),
            ("slope", [1.0, 2.0, 3.0], None),
            ("ca", [0.0, 1.0, 2.0, 3.0], None),
            ("thal", [3.0, 6.0, 7.0], None)
        ]

        for idx, row in self.df.iterrows():
            missing_values = []

            for column, valid_values, bounds in columns_to_clean:
                value = row[column]

                if pd.isnull(value):
                    missing_values.append(f"Missing value in {column}")
                elif valid_values is not None and value not in valid_values:
                    missing_values.append(f"Invalid value in {column}")

                if bounds is not None and (value < bounds[0] or value > bounds[1]):
                    missing_values.append(f"Value out of bounds in {column}")

            if missing_values:
                self.record_missing_values(idx, " | ".join(missing_values))

    def record_missing_values(self, row_id, missing_values):
        self.missing_values = self.missing_values.append({"RowID": row_id, "MissingValues": missing_values},
                                                         ignore_index=True)


'''
# Sample DataFrame creation (replace this with your actual dataset)
data = {
    "age": [45, 55, 65, None, 180],
    "sex": [0, 1, 2, 1, 0],
    "cp": [1, 2, 3, 5, None],
    "restbp": [130, 120, 140, 90, None],
    "chol": [200, 220, None, 50, 400],
    "fbs": [0, 1, 1, 0, 1],
    "restecg": [1, 2, 3, 1, 2],
    "thalach": [150, 160, 170, 180, None],
    "exang": [0, 1, 0, 1, 0],
    "oldpeak": [1.2, 2.5, 3.1, None, 0.8], # Add "oldpeak" column
    "slope": [1, 2 ,3 ,2 ,1],
    "ca": [0 ,1 ,2 ,None ,3],
    "thal": [3 ,6 ,7 ,7 ,3]
}

df = pd.DataFrame(data)
'''
X_df.index = range(1, len(X_df) + 1)

print(X_df)
# Create the DataCleaner instance and clean the data
cleaner = DataCleaner(X_df)
cleaner.clean_data()

# Filter out rows present in cleaner.missing_values
rows_with_missing_values = cleaner.missing_values["RowID"]
filtered_df = cleaner.df[~cleaner.df.index.isin(rows_with_missing_values)]

print(filtered_df)
print(cleaner.missing_values)


# Decision Tree plot:
import numpy as np
import graphviz
from sklearn import tree


def print_my_tree(classifier, iteration_name, X_train, y_train):
    # Print the iteration name as a header
    print(f"--- {iteration_name} ---")

    # Generate the DOT data for visualization
    dot_data = tree.export_graphviz(classifier, out_file=None,
                                    feature_names=X_train.columns,
                                    class_names=np.unique(y_train),
                                    filled=True)

    # Create a graph from the DOT data
    graph = graphviz.Source(dot_data, format="png")

    # Print the text representation of the decision tree
    tree_text = tree.export_text(classifier, feature_names=X_train.columns)
    print(tree_text)

    # Render and save the decision tree visualization
    output_filename = f"output2/{iteration_name}"
    graph.render(output_filename, view=False)

    # Print a separator line
    print("-- --")

# Example usage:
# print_my_tree(clfInput, "Iteration1", X_train, y_train)

