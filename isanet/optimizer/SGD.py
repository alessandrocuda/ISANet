""" Optimizer Module.
"""
import numpy as np
import time
import copy
import isanet.metrics as metrics
from isanet.optimizer import Optimizer
from isanet.optimizer.utils import l_norm
from isanet.optimizer.utils import make_vector, restore_w_to_model

class SGD(Optimizer):
    """Stochastic Gradient Descent (SGD)

    Parameters
    ----------
    lr : float, default=0.1
        Learning rate schedule for weight updates (delta rule).

    momentum : float, default=0
        Momentum for gradient descent update.

    nesterov : boolean, default=False
        Whether to use Nesterov’s momentum.

    sigma : float, default=None
        Parameter of the Super Accelerated Nesterov's momentum.
        If 'nesterov' is True and 'sigma' equals to 'momentum', then we have the
        simple Nesterov momentum. Instead, if 'sigma' is different from 
        'momentum', we have the super accelerated Nesterov.

    tol : float, default=None
        Tolerance for the optimization. When the loss on training is
        not improving by at least tol for 'n_iter_no_change' consecutive 
        iterations convergence is considered to be reached and training stops.

    n_iter_no_change : integer, default=None
        Maximum number of epochs with no improvements > tol. 

    norm_g_eps : float, optional      
        Threshold that is used to decide whether to stop the 
        fitting of the model (it stops if the norm of the gradient reaches 
        'norm_g_eps').

    l_eps : float, optional       
        Threshold that is used to decide whether to stop the 
        fitting of the model (it stops if the loss function reaches 
        'l_eps').

    debug : boolean, default=False
        If True, allows you to perform iterations one at a time, pressing the Enter key.

    Attributes
    ----------
    history : dict
        Save for each iteration some interesting values.

        Dictionary's keys:
            ``norm_g``
                Gradient norm. 
    """
    def __init__(self, lr=0.1, momentum=0, nesterov=False, sigma = None, 
                 tol = None, n_iter_no_change = None, norm_g_eps = None, 
                 l_eps = None, debug = False):
        super().__init__(tol = tol, n_iter_no_change = n_iter_no_change, norm_g_eps = norm_g_eps, l_eps = l_eps, debug = debug)
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.sigma = sigma
        if sigma is None and nesterov is True:
            self.sigma = momentum

        self.history = {"norm_g":     []}

        
    def optimize(self, model, epochs, X_train, Y_train, validation_data=None, batch_size=None, es=None, verbose=0):
        """
        Parameters
        ----------
        model : isanet.model.MLP
            Specify the Multilayer Perceptron object to optimize.

        epochs : integer
            Maximum number of epochs.

        X_train : array-like of shape (n_samples, n_features)
            The input data.

        Y_train : array-like of shape (n_samples, n_output)
            The target values.

        validation_data : list of arrays-like, [X_val, Y_val], optional
            Validation set.

        batch_size : integer, optional
            Size of minibatches for the optimizer.
            When set to "none", the optimizer will performe a full batch.

        es : isanet.callbacks.EarlyStopping, optional
            When set to None it will only use the ``epochs`` to finish training.
            Otherwise, an EarlyStopping type object has been passed and will stop 
            training if the model goes overfitting after a number of consecutive iterations.
            See docs in optimizier module for the EarlyStopping Class.

        verbose : integer, default=0
            Controls the verbosity: the higher, the more messages.

        Returns
        -------
        integer

        """
        if ~model.is_fitted:
            self.delta_w = [0]*len(model.weights)
        super().optimize(model, epochs, X_train, Y_train, validation_data=validation_data, batch_size=batch_size, es=es, verbose=verbose)


    def step(self, model, X, Y, verbose):
        """Implements the SGD step update method.

        Parameters
        ----------
        model : isanet.model.MLP
            Specify the Multilayer Perceptron object to optimize

         X : array-like of shape (n_samples, n_features)
            The input data.

        Y : array-like of shape (n_samples, n_output)
            The target values.

        verbose : integer, default=0
            Controls the verbosity: the higher, the more messages.

        Returns
        -------
            float
                The gradient norm.

        """

        current_batch_size = X.shape[0]
        lr = self.lr/current_batch_size

        weights = model.weights
        g  = self.backpropagation(model, weights, X, Y)
        norm_g = l_norm(g)

        if self.nesterov == True:
            weights = copy.deepcopy(model.weights)
            for i in range(0, len(model.weights)): 
                weights[i] += self.sigma*self.delta_w[i] 
            g  = self.backpropagation(model, weights, X, Y)

        #      Delta Rule Update
        #  w_i = w_i + eta*nabla_W_i
        for i in range(0, len(model.weights)):
            self.delta_w[i] = -lr*g[i] + self.momentum*self.delta_w[i]   
            regularizer = model.kernel_regularizer[i]*current_batch_size/self.tot_n_patterns
            weights_decay = regularizer*model.weights[i]
            weights_decay[0,:] = 0
            model.weights[i] += (self.delta_w[i] - weights_decay)

        self.history["norm_g"].append(norm_g)
        if verbose >= 2:
            print("opt debug - | norm_g: {:4.4f} |".format(norm_g)) 
        return norm_g