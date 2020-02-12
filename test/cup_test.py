import sys
from os import path

sys.path.insert(0, "../")
sys.path.insert(0, "./")

from isanet.model import Mlp
from isanet.optimizer import SGD
from isanet.utils.model_utils import printMSE, printAcc, plotMse
import numpy as np
import time
import matplotlib.pyplot as plt


from isanet.utils.k_fold import k_fold, load_kfold

TS = np.genfromtxt('../CUP/ML-CUP19-TR_tr_vl.csv',delimiter=',')

kfold = load_kfold("../CUP/4folds.index")

# plt.plot(TS[:,-2], TS[:,-1], 'ro', markersize=0.3)
# plt.ylabel('y2')
# plt.xlabel('y1')
# plt.tight_layout()
# plt.show()

for train_index_fold, val_index_fold in zip(kfold["train"], kfold["val"]):
    X_train = TS[train_index_fold,:-2]
    Y_train = TS[train_index_fold,-2:]

    X_val = TS[val_index_fold,:-2]
    Y_val = TS[val_index_fold,-2:]

    model = Mlp()
    model.add(40, activation="sigmoid", input= 20, kernel_initializer = 1/np.sqrt(20), kernel_regularizer = 0.0002)
    model.add(30, activation="sigmoid", kernel_initializer = 1/np.sqrt(40), kernel_regularizer = 0.0002)
    model.add(2, activation="linear", kernel_initializer = 1/np.sqrt(30), kernel_regularizer = 0.0002)


    model.set_optimizer(
        SGD(
            lr = 0.0198,
            momentum = 0.8,
            nesterov = False
        ))
    # Batch
    start_time = time.time()
    model.fit(X_train,
                Y_train, 
                epochs=10000, 
                #batch_size=31,
                validation_data = [X_val, Y_val],
                verbose=1)
    print("--- %s seconds ---" % (time.time() - start_time))

    print("--- %s seconds ---" % (time.time() - start_time))
    outputNet = model.predict(X_val)

    # plt.plot(outputNet[:,-2], outputNet[:,-1], 'ro', markersize=0.3)
    # plt.ylabel('y2')
    # plt.xlabel('y1')
    # plt.tight_layout()
    # plt.show()

    printMSE(outputNet, Y_val, type = "test")
    plotMse(model.history )
