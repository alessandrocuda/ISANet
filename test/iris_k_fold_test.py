import sys
from os import path

sys.path.insert(0, "../")
sys.path.insert(0, "./")

from isanet.model import Mlp
from isanet.optimizer import SGD
#from isanet.datasets.iris import load_iris
from isanet.utils.model_utils import printMSE, printAcc, plotHistory
import numpy as np
import time

from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder


from isanet.utils.k_fold import k_fold

iris = load_iris()
onehot_encoder = OneHotEncoder(sparse=False)
Y = onehot_encoder.fit_transform(iris.target.reshape(-1, 1))
X = iris.data
K = 5
kfold = k_fold(X, K)

kacc = []


for train_index_fold, val_index_fold in zip(kfold["train"], kfold["val"]):
    X_train = X[train_index_fold]
    Y_train = Y[train_index_fold]
    
    X_val = X[val_index_fold]
    Y_val = Y[val_index_fold]

    model = Mlp()
    model.add(6, input= 4, kernel_initializer = 1/np.sqrt(4), kernel_regularizer = 0.003)
    model.add(3, kernel_initializer = 1/np.sqrt(6), kernel_regularizer = 0.003)

    model.set_optimizer(
        SGD(
            lr = 0.71,
            momentum = 0.8,
            nesterov = True
        ))
    # Batch
    model.fit(X_train,
                Y_train, 
                epochs=1000, 
                #batch_size=31,
                validation_data = [X_val, Y_val],
                verbose=0) 

    outputNet = model.predict(X_val)

    delta = Y_val - outputNet

    error = np.sum(np.square(delta))/X_val.shape[0]
    acc = np.mean(((outputNet > .5) == Y_val).all(1))
    printMSE(outputNet, Y_val, type = "test")
    printAcc(outputNet, Y_val, type = "test")
    kacc.append(acc)

print("k acc: {}".format(np.mean(kacc)))
print("k std acc: {}".format(np.std(kacc)))