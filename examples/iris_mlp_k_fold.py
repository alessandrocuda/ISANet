import sys
from os import path

sys.path.insert(0, "../")
sys.path.insert(0, "./")


from isanet.model_selection import Kfold, GridSearchCV
from isanet.neural_network import MLPClassifier
from isanet.datasets.iris import load_iris
from isanet.metrics import mse, mee, accuracy_binary
import numpy as np

X, Y = load_iris()


X_train = X[:112,:]
Y_train = Y[:112,:]

X_test = X[112:,:]
Y_test = Y[112:,:]


kf = Kfold(n_splits=5, shuffle=True, random_state=42)
kfold = kf.split(X_train,Y_train)

mse_estimation_train = []
accuracy_estimation_train = []
mse_estimation_val = []
accuracy_estimation_val = []


hyper_param = {"n_layer_units": [6, 3],
               "learning_rate": 0.79,
               "max_epoch": 1500,
               "momentum": 0.8,
               "nesterov": True,
               "sigma": None,
               "kernel_regularizer": 0.006,
               "activation": "sigmoid",
               "early_stop": None}

for train_index_fold, val_index_fold in zip(kfold["train"], kfold["val"]):
    X_train = X[train_index_fold]
    Y_train = Y[train_index_fold]
    
    X_val = X[val_index_fold]
    Y_val = Y[val_index_fold]

    mlp_c = MLPClassifier(X_train.shape[1], Y_train.shape[1],  **hyper_param)           

    mlp_c.fit(X_train, Y_train, X_val, Y_val)

    Y_train_predicte = mlp_c.predict(X_train)
    Y_val_predicte   = mlp_c.predict(X_val)

    mse_estimation_train.append(mse(Y_train, Y_train_predicte))
    accuracy_estimation_train.append(accuracy_binary(Y_train, Y_train_predicte))
    mse_estimation_val.append(mse(Y_val, Y_val_predicte))
    accuracy_estimation_val.append(accuracy_binary(Y_val, Y_val_predicte))

print("Results Train:")
print("MSE: {0} +/- {1}".format(np.mean(mse_estimation_train), np.std(mse_estimation_train)))
print("Accuracy: {0} +/- {1}".format(np.mean(accuracy_estimation_train), np.std(mse_estimation_train)))
print()
print("Results Val:")
print("MSE: {0} +/- {1}".format(np.mean(mse_estimation_val), np.std(mse_estimation_val)))
print("Accuracy: {0} +/- {1}".format(np.mean(accuracy_estimation_val), np.std(mse_estimation_val)))
