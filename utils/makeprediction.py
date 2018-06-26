from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

import pdb

def predict(inputFeatures):
    iris = datasets.load_iris()

    knn = KNeighborsClassifier()
    
    knn.fit(iris.data, iris.target)

    inputs = np.array(inputFeatures)

    predictInt = knn.predict(inputs.reshape(1,-1))

    if predictInt[0] == 0:
        predictString = 'setosa'
    elif predictInt[0] == 1:
        predictString = 'versicolor'
    elif predictInt[0] == 2:
        predictString = 'virginica'
    else:
        predictString = 'null'

    return predictString