import sys
from os import path

sys.path.insert(0, "../../")
sys.path.insert(0, "./")

from isanet.model import Mlp
from isanet.optimizer import SGD
from isanet.datasets.iris import load_iris
from isanet.utils.model_utils import printMSE, printAcc, plotHistory
import numpy as np
import time

X, Y = load_iris()

X_train = X[:75,:]
Y_train = Y[:75,:]

X_val = X[75:112,:]
Y_val = Y[75:112,:]

X_test = X[112:,:]
Y_test = Y[112:,:]

model = Mlp()
model.add(6, input= 4, kernel_initializer = 1/np.sqrt(4), kernel_regularizer = 0.006)
model.add(3, kernel_initializer = 1/np.sqrt(6), kernel_regularizer = 0.006)

model.set_optimizer(
    SGD(
        lr = 0.39,
        momentum = 0.8,
        nesterov = True
    ))

start_time = time.time()
model.fit(X_train, Y_train, validation_data = [X_val, Y_val],
            epochs=1500, 
            #batch_size=31,
            verbose=1) 
print("--- %s seconds ---" % (time.time() - start_time))

outputNet = model.predict(X_test)

printMSE(outputNet, Y_test, type = "test")
printAcc(outputNet, Y_test, type = "test")
plotHistory(model.history )