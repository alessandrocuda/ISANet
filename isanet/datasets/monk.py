import numpy as np
import matplotlib.pyplot as plt
import os.path 
from importlib import resources

def load_monk(dataset = "1", type = "train"):
    """Load and return the one of the Monk dataset (Classification Task).
    The monk datasets are a classic and very easy class classification
    dataset.

    Parameters
    ----------

    dataset : string ("1", "2", "3"), default = "1"
        define which monks dataset load

    type : string ("train", "test"), default = "train
        define which type of dataset you want to load: train or test

    Note
    ----

    The one hot encoder is applied to the data variable:
    each row is a feature of X Data of a monk dataset and 
    and its related encoding.::

            {
                0: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                1: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                2: [[1,0],[0,1]],
                3: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                4: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                5: [[1,0],[0,1]]
            }

    Monk 1
    
    ========================   =============  =========
                                Train              Test
    Classes                          2                2
    Samples total                  124              432
    Data X Dimensionality           17               17
    Data X Type                   bool             bool
    Data Y Dimensionality            1                1
    Data Y Type                   bool             bool
    ========================   =============  =========
    
    Monk 2
    
    ========================   =============  =========
                                Train              Test
    Classes                          2                2
    Samples total                  169              432
    Data X Dimensionality           17               17
    Data X Type                   bool             bool
    Data Y Dimensionality            1                1
    Data Y Type                   bool             bool
    ========================   =============  =========

    Monk 3
    
    ========================   =============  =========
                                Train              Test
    Classes                          2                2
    Samples total                  122              432
    Data X Dimensionality           17               17
    Data X Type                   bool             bool
    Data Y Dimensionality            1                1
    Data Y Type                   bool             bool
    ========================   =============  =========

    """

    with resources.open_text("isanet.datasets.data", "monks-"+str(dataset)+"."+str(type)) as fid:
        data = [i.strip('\n').split(' ')[1:-1] for i in fid]
    res = [[int(data_row_value) for data_row_value in data_row] for data_row in data ]
    set = np.array(res)
    Y = set[:,[0]]
    X = set[:,1:]


    features = {0: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                1: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                2: [[1,0],[0,1]],
                3: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                4: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                5: [[1,0],[0,1]]}

    X_ohe = []


    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            X_ohe = np.append(X_ohe,features[j][X[i,j]-1])
    X = X_ohe.reshape(-1, 17)

    return X, Y