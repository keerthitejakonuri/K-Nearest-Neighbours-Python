# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('zip_codes_states_1.csv')
X = dataset.iloc[:, [1, 2]].values        
X1 = dataset.iloc[:,[0,1,2,3,4,5]].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test = train_test_split(X, test_size = 0.25, random_state = 0)

#Finding the Nearest Neighbours
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X_train)
#distances, indices = nbrs.kneighbors(X)
Y = nbrs.kneighbors(X_test,2,return_distance=False)
for x in Y:
    print(X1[x[0],:])
    print(X1[x[1],:])
    print("\n")
    #print(x)