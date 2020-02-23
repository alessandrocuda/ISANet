import numpy as np
import matplotlib.pyplot as plt
import os.path 
from importlib import resources

def load_iris():
    """Load and return the iris dataset (Classification Task).
    The iris dataset is a classic and very easy multi-class classification
    dataset.


    Note
    ----

    The one hot encoder is applied to the target variable::

            {
                "Iris-setosa": [1, 0, 0],
                "Iris-versicolor": [0, 1, 0],
                "Iris-virginica": [0, 0, 1]
            }

    =======================   ==============
    Classes                                3
    Samples per class                     50
    Samples total                        150
    Data X Dimensionality                  4
    Data X Type                         real 
    Data Y Dimensionality                  3
    Data Y Type                         bool
    =======================   ==============
    """
    with resources.open_text("isanet.datasets.data", "iris.data") as fid:
        data = [i.strip('\n').split(',') for i in fid]

    res = [[data_row_value for data_row_value in data_row] for data_row in data ]
    set = np.array(res)

    np.random.shuffle(set)
    X = set[:,:-1]
    Y = set[:,-1]

    Y_ohe = np.zeros((Y.shape[0],3))
    target_ohe = {"Iris-setosa": [1, 0, 0],
                  "Iris-versicolor": [0, 1, 0],
                  "Iris-virginica": [0, 0, 1]}

    for i in range(0, Y.shape[0]):
        Y_ohe[i,:] = target_ohe[Y[i]]

    X = X.astype(float)
    Y = Y_ohe.astype(int)

    return X, Y