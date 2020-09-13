import sys
from os import path
sys.path.insert(0, "../../")
sys.path.insert(0, "./")

from isanet.model import Mlp
from isanet.optimizer import SGD, NCG, LBFGS
from isanet.datasets.monk import load_monk
from isanet.utils.model_utils import printMSE, printAcc, plotHistory
import numpy as np

np.random.seed(seed=875)

print("Load Monk DataSet")
X_train, Y_train = load_monk("3", "train")
X_test, Y_test = load_monk("3", "test")

print("Build the model")
model = Mlp()
model.add(4, input= 17, kernel_initializer = 0.003, kernel_regularizer = 0.001)
model.add(1, kernel_initializer = 0.003, kernel_regularizer = 0.001)

# model.set_optimizer(
#     SGD(
#         lr = 0.8,
#         momentum = 0.6,
#         nesterov = True
#     ))

# model.set_optimizer(
#     NCG(tol=1e-20)
# )

model.set_optimizer(
    LBFGS(m=3, c1= 1e-4, c2=0.4, tol=1e-20)
)

# Batch
model.fit(X_train,
            Y_train, 
            epochs=30, 
            #batch_size=31,
            validation_data = [X_test, Y_test],
            verbose=1) 

outputNet = model.predict(X_test)

printMSE(outputNet, Y_test, type = "test")
printAcc(outputNet, Y_test, type = "test")
plotHistory(model.history )