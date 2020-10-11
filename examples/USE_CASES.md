An example with the **low level api (keras-like)**:

```python
# ...
from isanet.model import Mlp
from isanet.optimizer import SGD, EarlyStopping
from isanet.datasets.monk import load_monk
import numpy as np

X_train, Y_train = load_monk("1", "train")
X_test, Y_test = load_monk("1", "test")

#create the model
model = Mlp()
# Specify the range for the weights and lambda for regularization
# Of course can be different for each layer
kernel_initializer = 0.003 
kernel_regularizer = 0.001

# Add many layers with different number of units
model.add(4, input= 17, kernel_initializer, kernel_regularizer)
model.add(1, kernel_initializer, kernel_regularizer)

es = EarlyStopping(0.00009, 20) # eps_GL and s_UP

#fix which optimizer you want to use in the learning phase
model.setOptimizer(
    SGD(lr = 0.83,          # learning rate
        momentum = 0.9,     # alpha for the momentum
        nesterov = True,    # Specify if you want to use Nesterov
        sigma = None        # sigma for the Acc. Nesterov
    ))

#start the learning phase
model.fit(X_train,
          Y_train, 
          epochs=600, 
          #batch_size=31,
          validation_data = [X_test, Y_test],
          es = es,
          verbose=0) 
            
# after trained the model the prediction operation can be
# perform with the predict method
outputNet = model.predict(X_test)
```

An example with the **low level api (keras-like)** with NCG or LBFGS optimizer:
```python
from isanet.model import Mlp
from isanet.optimizer import NCG, LBFGS
from isanet.optimizer.utils import l_norm
from isanet.datasets.monk import load_monk
from isanet.utils.model_utils import printMSE, printAcc, plotHistory
import isanet.metrics as metrics
import numpy as np

X_train, Y_train = load_monk("1", "train")
X_test, Y_test = load_monk("1", "test")

#create the model
model = Mlp()
# Specify the range for the weights and lambda for regularization
# Of course can be different for each layer
kernel_initializer = 0.003 
kernel_regularizer = 0.001

# Add many layers with different number of units
model.add(4, input= 17, kernel_initializer, kernel_regularizer)
model.add(1, kernel_initializer, kernel_regularizer)

es = EarlyStopping(0.00009, 20) # eps_GL and s_UP

optimizer = NCG(beta_method="hs+", c1=1e-4, c2=0.1, restart=None, ln_maxiter = 100, norm_g_eps = 1e-9, l_eps = 1e-9)
# or you can choose the LBFGS optimizer
#optimizer = LBFGS(m = 30, c1=1e-4, c2=0.9, ln_maxiter = 100, norm_g_eps = 1e-9, l_eps = 1e-9)

#start the learning phase
# no batch with NCG or LBFGS optimizer
model.fit(X_train,
          Y_train, 
          epochs=600,  
          validation_data = [X_test, Y_test],
          es = es,
          verbose=0) 
            
# after trained the model the prediction operation can be
# perform with the predict method
outputNet = model.predict(X_test)
```


An example with the **high level API (sklearn like)**:

```python
from isanet.neural_network import MLPClassifier
from isanet.datasets.monk import load_monk

X_train, Y_train = load_monk("1", "train")
X_test, Y_test = load_monk("1", "test")

mlp_c = MLPClassifier(X_train.shape[1],             # input dim
                      Y_train.shape[1],             # out dim
                      n_layer_units=[4],            # topology
                      activation="sigmoid",         # activation hidden layer
                      kernel_regularizer=0.001,     # l2 regularization term
                      max_epoch=600,                # Max number of Epoch
                      learning_rate=0.83,           # learning rate
                      momentum=0.9,                 # momentum term
                      nesterov=True,                # if Nesterov
                      sigma=None,                   # sigma Acc. Nesterov
                      early_stop=None,              # define the early stop
                      verbose=0)                    # verbosity
mlp_c.fit(X_train, Y_train, X_test, Y_test)
outputNet = mlp_c.predict(X_test)
```

An example with the **Model Selection Module**:
```python
from isanet.model_selection import Kfold, GridSearchCV
from isanet.neural_network import MLPRegressor

dataset = np.genfromtxt('CUP/ML-CUP19-TR_tr_vl.csv',delimiter=',')
X_train, Y_train = dataset[:,:-2], dataset[:,-2:]

grid = {"n_layer_units": [[38], [20, 32]], #[20, 32] means two hidden layer
         "learning_rate": [0.014, 0.017],
         "max_epoch": [13000, 1000],
         "momentum": [0.8, 0.6],
         "nesterov": [True, False],
         "sigma": [None, 0.8, 0.6, 2, 4]
         "kernel_regularizer": [0.0001],
         "activation": ["sigmoid"],
         "early_stop": [EarlyStopping(0.00009, 20), EarlyStopping(0.09, 200)]}

mlp_r = MLPRegressor(X_train.shape[1], Y_train.shape[1])
kf = Kfold(n_splits=5, shuffle=True, random_state=1)
gs = GridSearchCV(estimator=mlp_r, param_grid = grid, cv = kf, verbose=2)
result = gs.fit(X, Y) # dict with keys as column headers and values as columns
```