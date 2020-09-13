""" Optimizer Module.
"""
import numpy as np
import time
import copy
import isanet.metrics as metrics
from isanet.optimizer import Optimizer
# from isanet.optimizer.linesearch import armijo_wolfe_ls
from isanet.optimizer.utils import make_vector, restore_w_to_model

class SGD(Optimizer):
    """
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
    """
    def __init__(self, lr=0.1, momentum=0, nesterov=False, sigma = None, tol = None, n_iter_no_change = None):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        if n_iter_no_change is None:
            self.n_iter_no_change = 1

        self.sigma = sigma
        if sigma is None and nesterov is True:
            self.sigma = momentum

        
    def optimize(self, model, epochs, X_train, Y_train, validation_data=None, batch_size=None, es=None, verbose=0):
        if ~model.is_fitted:
            self.delta_w = [0]*len(model.weights)
        super().optimize(model, epochs, X_train, Y_train, validation_data=validation_data, batch_size=batch_size, es=es, verbose=verbose)


    def step(self, model, X, Y):

        current_batch_size = X.shape[0]
        lr = self.lr/current_batch_size

        weights = model.weights
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
