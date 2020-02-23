import numpy as np

from isanet.model import Mlp
from isanet.optimizer import SGD
from isanet.optimizer import EarlyStopping
from isanet.metrics import mse, mee, accuracy_binary

class __BaseMLP():
    """Base class for MLP classification and regression."""

    def __init__(self, task, input_dim, out_dim, n_layer_units = [100], activation = "relu",
                 kernel_regularizer = 0.0001, batch_size = None, max_epoch = 1000,
                 learning_rate = 0.1, momentum = 0.9, nesterov = False, sigma = None, early_stop = False,
                 random_state = None, verbose = 0):
        
        if random_state:
            np.random.seed(random_state)

        self.__input_dim = input_dim
        self.__out_dim = out_dim
        self.__task = task
        self.verbose = verbose
        self.set_params(input_dim, out_dim, n_layer_units, activation,
                        kernel_regularizer, batch_size, max_epoch,
                        learning_rate, momentum, nesterov, sigma, early_stop)

    def fit(self, X_train, Y_train, X_val, Y_val):
        """ Fit the model to data matrix X_train and target(s) Y_train and 
        evaluates it on the validation set (X_val, Y_val).
        """
        self._model.fit(X_train,
                        Y_train, 
                        epochs=self._params["max_epoch"],
                        batch_size=self._params["batch_size"],
                        validation_data = [X_val, Y_val],
                        es = self._params["early_stop"],
                        verbose = self.verbose)

    def predict(self, X):
        """Predict using the multi-layer perceptron classifier."""

        return  self._model.predict(X)

    def get_params(self):
        """Returns the parameters of the multi-layer perceptron."""

        return self._params

    def get_history(self):
        """Returns the history of the multi-layer perceptron."""

        return self._model.history

    def get_weights(self):
        """Returns the weights of the multi-layer perceptron."""

        return self._model.weights

    def set_params(self, input_dim, out_dim, n_layer_units = [100], activation = "relu",
                      kernel_regularizer = 0.0001, batch_size = None, max_epoch = 1000,
                      learning_rate = 0.1, momentum = 0.9, nesterov = False, sigma = None, early_stop = False):
        """Set the multi-layer perceptron model."""

        self._params = {
            "n_layer_units": n_layer_units, 
            "activation": activation,
            "kernel_regularizer": kernel_regularizer,
            "batch_size": batch_size,
            "max_epoch": max_epoch,
            "learning_rate": learning_rate,
            "momentum": momentum,
            "nesterov": nesterov,
            "sigma": sigma,
            "early_stop": early_stop
        }

        self.__input_dim = input_dim
        self.__out_dim = out_dim   

        self._model = self.__set_model(self.__task, n_layer_units, activation,
                                         kernel_regularizer, learning_rate, 
                                         momentum, nesterov, sigma)  


    def __set_model(self, task, n_layer_units, activation, kernel_regularizer, 
                      learning_rate, momentum, nesterov, sigma):
        model = Mlp()

        output_activation = "linear"
        if task == 'c':
            output_activation = "sigmoid"

        model.add(n_layer_units[0], 
                  activation=activation, 
                  input=self.__input_dim, 
                  kernel_initializer = np.sqrt(6)/np.sqrt(self.__input_dim + n_layer_units[0]), 
                  kernel_regularizer = kernel_regularizer)

        for layer in range(1, len(n_layer_units)):
            model.add(n_layer_units[layer], 
                      activation=activation, 
                      kernel_initializer = np.sqrt(6)/np.sqrt(n_layer_units[layer-1] + n_layer_units[layer]), 
                      kernel_regularizer = kernel_regularizer)
        
        model.add(self.__out_dim, 
                  activation = output_activation, 
                  kernel_initializer = np.sqrt(6)/np.sqrt(self.__out_dim + n_layer_units[-1]), 
                  kernel_regularizer = kernel_regularizer)

        model.set_optimizer(
        SGD(
            lr = learning_rate,
            momentum = momentum,
            nesterov = nesterov,
            sigma = sigma,
        ))
        return model



class MLPClassifier(__BaseMLP):                     
    """Multi-layer Perceptron classifier.
    
    This model optimizes the MSE function using the stochastic gradient descent.

    Parameters
    ----------
    input_dim : int, no default value mandatory parameter
        allows you to specify the number of inputs on the input layer

    out_dim : int, no default value mandatory parameter
        allows you to specify the number of outputs on the output layer

    hidden_layer_sizes : list, default=[100]
        The ith element of the list represents the number of neurons in the ith
        hidden layer::
        
                E.g. [20, 40, 60] means 3 hidden layers with 20, 40 and 60 neurons respectively.

    activation : {'identity', 'sigmoid', 'tanh', 'relu'}, default='relu'
        Activation function available for the hidden layer.

        - 'identity', no-op activation, useful to implement linear bottleneck,
          returns f(x) = x
        - 'sigmoid', the logistic sigmoid function,
          returns f(x) = 1 / (1 + exp(-x)).
        - 'tanh', the hyperbolic tan function,
          returns f(x) = tanh(x).
        - 'relu', the rectified linear unit function,
          returns f(x) = max(0, x)

    kernel_regularizer : float, default=0.0001
        Tikhonov regularization term, L2 penalty parameter.

    batch_size : int, default='None'
        Size of minibatches for the SGD optimizer.
        When set to "none", the SGD will performe a full batch.

    learning_rate : float, default=0.1
         The constant value that will be used by the SGD optimizer as learning rate

    momentum : float, default=0.9
        Momentum for gradient descent update. Should be between 0 and 1.

    nesterovs_momentum : boolean, default=True
        Whether to use Nesterov's momentum. If the momentum == 0 this parameter
        is useless.

    sigma : float, default=None
        Parameter of the Super Accelerated Nesterov's momentum.
        If 'nesterov' is True and 'sigma' equals to 'momentum', then we have the
        simple Nesterov momentum. Instead, if 'sigma' is different from 
        'momentum', we have the super accelerated Nesterov.

    max_epoch : int, default=1000
        It will set the Maximum number of Epoch for the SGD optimizer.
        The solver iterates until convergence (determined by 'tol') or this number of iterations. 

    early_stopping : bool or isanet.callbacks.EarlyStopping, default=False
        When set to False it will only use the ``max_epoch`` to finish training.
        Otherwise, an EarlyStopping type object has been passed and will stop 
        training if the model goes overfitting after a number of consecutive iterations.
        See docs in optimizier module.

    random_state : int, RandomState instance or None, default=None
        If int, random_state is the seed used by numpy random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : bool, default=False
        Whether to print progress messages to stdout.

    Notes
    -----
    MLPClassifier provides a high-level interface capable of biting a neural network 
    using the parameters passed to the class as hyper parameters.
    
    It can also have a regularization term added to prevent overfitting.
    
    Numpy arrays of floating point values are used to store all the data under the hood. 
    this This matrix implementation allowed us to speed up the computation compared to an 
    object-oriented structure, this was possible thanks to Numpy that is able to perform 
    matrix operation in an efficient way by parallelizes each operation. 
    Numpy use optimized math routines, written in C or Fortran, for linear algebra operation 
    as: Blas, OpenBlas or Intel Math Kernel Library (MKL).

    Methods
    -------

    fit(self, X_train, Y_train, X_val, Y_val)
        Fit the model to data matrix X_train and target(s) Y_train and 
        evaluates it on the validation set (X_val, Y_val).
       
    predict(self, X)
        Predict using the multi-layer perceptron classifier.

    get_params(self)
        Returns the parameters of the multi-layer perceptron.

    get_history(self)
        Returns the history of the multi-layer perceptron.

    get_weights(self)
        Returns the weights of the multi-layer perceptron.
    """
    def __init__(self, input_dim, out_dim, n_layer_units = [100], activation = "relu",
                      kernel_regularizer = 0.0001, batch_size = None, max_epoch = 1000,
                      learning_rate = 0.1, momentum = 0.9, nesterov = False, sigma = None, early_stop = False,
                      random_state = None,verbose = 0):
                      
        super().__init__('c', input_dim, out_dim, n_layer_units, activation,
                         kernel_regularizer, batch_size, max_epoch,
                         learning_rate, momentum, nesterov, sigma, early_stop, 
                         random_state, verbose)

class MLPRegressor(__BaseMLP):
    """Multi-layer Perceptron regressor.
    
    This model optimizes the MSE function using the stochastic gradient descent.

    Parameters
    ----------
    input_dim : int, no default value mandatory parameter
        allows you to specify the number of inputs on the input layer

    out_dim : int, no default value mandatory parameter
        allows you to specify the number of outputs on the output layer

    hidden_layer_sizes : list, default=[100]
        The ith element of the list represents the number of neurons in the ith
        hidden layer.

            E.g. [20, 40, 60] means 3 hidden layers with 20, 40 and 60 neurons respectively

    activation : {'identity', 'sigmoid', 'tanh', 'relu'}, default='relu'
        Activation function available for the hidden layer.

        - 'identity', no-op activation, useful to implement linear bottleneck,
          returns f(x) = x
        - 'sigmoid', the logistic sigmoid function,
          returns f(x) = 1 / (1 + exp(-x)).
        - 'tanh', the hyperbolic tan function,
          returns f(x) = tanh(x).
        - 'relu', the rectified linear unit function,
          returns f(x) = max(0, x)

    kernel_regularizer : float, default=0.0001
        Tikhonov regularization term, L2 penalty parameter.

    batch_size : int, default='None'
        Size of minibatches for the SGD optimizer.
        When set to "none", the SGD will performe a full batch.

    learning_rate : float, default=0.1
         The constant value that will be used by the SGD optimizer as learning rate

    momentum : float, default=0.9
        Momentum for gradient descent update. Should be between 0 and 1.

    nesterovs_momentum : boolean, default=True
        Whether to use Nesterov's momentum. If the momentum == 0 this parameter
        is useless.

    sigma : float, default=None
        Parameter of the Super Accelerated Nesterov's momentum.
        If 'nesterov' is True and 'sigma' equals to 'momentum', then we have the
        simple Nesterov momentum. Instead, if 'sigma' is different from 
        'momentum', we have the super accelerated Nesterov.

    max_epoch : int, default=1000
        It will set the Maximum number of Epoch for the SGD optimizer.
        The solver iterates until convergence (determined by 'tol') or this number of iterations. 

    early_stopping : bool or isanet.callbacks.EarlyStopping, default=False
        When set to False it will only use the ``max_epoch`` to finish training.
        Otherwise, an EarlyStopping type object has been passed and will stop 
        training if the model goes overfitting after a number of consecutive iterations.
        See docs in optimizier module. 

    random_state : int, RandomState instance or None, default=None
        If int, random_state is the seed used by numpy random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : bool, default=False
        Whether to print progress messages to stdout.
    
    Notes
    -----
    MLPRegressor provides a high-level interface capable of biting a neural network 
    using the parameters passed to the class as hyper parameters.
    
    It can also have a regularization term added to prevent overfitting.
    
    Numpy arrays of floating point values are used to store all the data under the hood. 
    this This matrix implementation allowed us to speed up the computation compared to an 
    object-oriented structure, this was possible thanks to Numpy that is able to perform 
    matrix operation in an efficient way by parallelizes each operation. 
    Numpy use optimized math routines, written in C or Fortran, for linear algebra operation 
    as: Blas, OpenBlas or Intel Math Kernel Library (MKL).
    
    Methods
    -------

    fit(self, X_train, Y_train, X_val, Y_val)
        Fit the model to data matrix X_train and target(s) Y_train and 
        evaluates it on the validation set (X_val, Y_val).
       
    predict(self, X)
        Predict using the multi-layer perceptron classifier.

    get_params(self)
        Returns the parameters of the multi-layer perceptron.

    get_history(self)
        Returns the history of the multi-layer perceptron.

    get_weights(self)
        Returns the weights of the multi-layer perceptron.
    """
    def __init__(self, input_dim, out_dim, n_layer_units = [100], activation = "relu",
                      kernel_regularizer = 0.0001, batch_size = None, max_epoch = 1000,
                      learning_rate = 0.1, momentum = 0.9, nesterov = False, sigma = None, early_stop = False,
                      random_state = None,verbose = 0):
                      
        super().__init__('r', input_dim, out_dim, n_layer_units, activation,
                         kernel_regularizer, batch_size, max_epoch,
                         learning_rate, momentum, nesterov, sigma, early_stop, 
                         random_state, verbose)