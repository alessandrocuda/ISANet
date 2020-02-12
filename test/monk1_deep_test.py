import sys
from os import path
sys.path.insert(0, "../")
sys.path.insert(0, "./")

from isanet.model import Mlp
from isanet.optimizer import SGD
from isanet.datasets.monk import load_monk
from isanet.utils.model_utils import printMSE, printAcc, plotHistory
import numpy as np
import time

print("Load Monk DataSet")
X_train, Y_train = load_monk("1", "train")
X_test, Y_test = load_monk("1", "test")

print("Build the model")

tk_reg = 0.00001
w_start = 0.79
model = Mlp()
model.add(6, input= 17, kernel_initializer = w_start, kernel_regularizer= tk_reg)
model.add(6, kernel_initializer = w_start, kernel_regularizer= tk_reg)
model.add(2, kernel_initializer = w_start, kernel_regularizer= tk_reg)
model.add(1, kernel_initializer = w_start, kernel_regularizer= tk_reg)

model.set_optimizer(
    SGD(
        lr = 0.85,
        momentum = 0.9,
        nesterov = True
    ))
# Batch
start_time = time.time()
model.fit(X_train,
            Y_train, 
            epochs=2000, 
            #batch_size=31,
            validation_data = [X_test, Y_test],
            verbose=0) 
print("--- %s seconds ---" % (time.time() - start_time))

outputNet = model.predict(X_test)

printMSE(outputNet, Y_test, type = "test")
printAcc(outputNet, Y_test, type = "test")
plotHistory(model.history )
