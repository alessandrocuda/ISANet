import sys
from os import path
sys.path.insert(0, "../")
sys.path.insert(0, "./")

from isanet.neural_network import MLPRegressor
from isanet.optimizer import EarlyStopping
from isanet.utils.model_utils import printMSE, printAcc, plotHistory
from isanet.model import Mlp
from isanet.optimizer import SGD

import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(42)
TS = np.genfromtxt('CUP/ML-CUP19-TR_tr_vl.csv',delimiter=',')

X_train = TS[:1000,:-2]
Y_train = TS[:1000,-2:]
    
X_val = TS[1000:,:-2]
Y_val = TS[1000:,-2:]

print(X_train.shape)
print(X_val.shape)

print("Build the model")

# es = EarlyStopping(0.00009, 20)
# mlp_r = MLPRegressor(X_train.shape[1], Y_train.shape[1], n_units=[52],activation="sigmoid", kernel_regularizer=0,
#                      max_epoch=150, learning_rate=0.018, momentum=0.8, nesterov=True, early_stop=es, verbose=1)


# mlp_r.fit(X_train, Y_train, X_val, Y_val)

# outputNet = mlp_r.predict(X_val)





model = Mlp()
model.add(38, activation="sigmoid", input= 20, kernel_initializer = np.sqrt(6)/np.sqrt(20 +38), kernel_regularizer = 0.0001)
model.add(2, activation="linear", kernel_initializer = np.sqrt(6)/np.sqrt(38 + 2), kernel_regularizer = 0.0001)

model.set_optimizer(
    SGD(
        lr = 0.018,
        momentum = 0.8,
        nesterov = True,
        #gain = 0.8
    ))
es = EarlyStopping(0.0009, 20, verbose = True)

start_time = time.time()
model.fit(X_train,
            Y_train, 
            epochs=30, 
            validation_data = [X_val, Y_val],
            es = es,
            verbose=1) 

print("--- %s seconds ---" % (time.time() - start_time))
outputNet = model.predict(X_val)

# plt.plot(outputNet[:,-2], outputNet[:,-1], 'ro', markersize=0.3)
# plt.ylabel('y2')
# plt.xlabel('y1')
# plt.tight_layout()
# plt.show()

printMSE(outputNet, Y_val, type = "test")
plt.plot(model.history["loss_mse"])
plt.plot(model.history["val_loss_mse"])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.tight_layout()
plt.show()
