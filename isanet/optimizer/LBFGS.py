""" Optimizer Module.
"""
import numpy as np
import time
import copy
import isanet.metrics as metrics
from isanet.optimizer import Optimizer
from isanet.optimizer.linesearch import line_search_wolfe, line_search_wolfe_f, phi_function
from isanet.optimizer.utils import make_vector, restore_w_to_model

class LBFGS(Optimizer):
    """Limited-memory BFGS (L-BFGS)
    
    Parameters
    ----------
    m : integer, default=3
        The Hessian approximation will keep the curvature information from the 'm' 
        most recent iterations.

    c1 : float, default=1e-4
        Parameter for the Armijo-Wolfe line search.

    c2 : float, default=0.9
        Parameter for the Armijo-Wolfe line search.

    ln_maxiter : integer, default=10
        Maximum number of iterations of the Line Search.

    tol : float, optional
        Tolerance for the optimization. When the loss on training is
        not improving by at least tol for 'n_iter_no_change' consecutive 
        iterations convergence is considered to be reached and training stops.

    n_iter_no_change : integer, optional
        Maximum number of iterations with no improvements > tol.

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
            ``alpha``
                Step size chosen by the line search.
            ``norm_g``
                Gradient norm.
            ``ls_conv``
                Specifies whether the line search was able to find an alpha.
            ``ls_it``
                Number of iterations of the line search.
            ``ls_time``
                Computational time of the line search 
                (includes the computational time of the zoom method, if used).
            ``zoom_used``
                Specifies whether the zoom method has been used.
            ``zoom_conv``
                Specifies whether the zoom method was able to find an alpha.
            ``zoom_it``
                Number of iterations of the zoom method.
    """
    def __init__(self, m = 3, c1=1e-4, c2=.9, ln_maxiter = 10, tol = None, 
                 n_iter_no_change = None, norm_g_eps = None, l_eps = None, 
                 debug = False):
        super().__init__(loss="loss_mse_reg", tol = tol, n_iter_no_change = n_iter_no_change, norm_g_eps = norm_g_eps, l_eps = l_eps, debug = debug)
        self.c1 = c1
        self.c2 = c2
        self.m = m
        self.restart = 0
        self.ln_maxiter = ln_maxiter
        self.history = {"alpha":        [],
                        "norm_g":       [],
                        "ls_conv":      [],
                        "ls_it":        [],
                        "ls_time":      [],
                        "zoom_used":    [],
                        "zoom_conv":    [],
                        "zoom_it":      []} 

        self.__old_phi0 = None
        self.__s = []
        self.__y = []


    def backpropagation(self, model, weights, X, Y):
        """Computes the derivative of 1/n sum_n (y_i -y_i')^2 + lamda*||weights||^2.

        Parameters
        ----------
        model : isanet.model.MLP
            Specify the Multilayer Perceptron object to optimize

        weights : list
            List of arrays, the ith array represents all the 
            weights of each neuron in the ith layer.

        X : array-like of shape (n_samples, n_features)
            The input data.

        Y : array-like of shape (n_samples, n_output)
            The target values.

        Returns
        -------
        list
            contains the gradients norm for each layer to be used in the delta rule. 
            Each index in the list represents the ith layer. (from the first
            hidden layer to the output layer).::

                E.g. 0 -> first hidden layer, ..., n+1 -> output layer
                where n is the number of hidden layer in the net.
        """
        g = super().backpropagation(model, weights, X, Y)
        for i in range(len(g)):
            g[i]  = (2/X.shape[0])*g[i] + (2*model.kernel_regularizer[0])*weights[i]
        return g

    def step(self, model, X, Y, verbose):
        """Implements the LBFGS step update method.

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
        ------
            float
                The gradient norm.

        """

        current_batch_size = X.shape[0]

        w = make_vector(model.weights)
        g = make_vector(self.backpropagation(model, model.weights, X, Y))
        norm_g = np.linalg.norm(g)
        phi0 = metrics.mse_reg(Y, model.predict(X), model, model.weights)

        if ~model.is_fitted and self.epoch == 0:
            d = - g
        else:
            self.__y[-1] = g - self.__y[-1]
            gamma = np.dot(self.__s[-1].T, self.__y[-1])/np.dot(self.__y[-1].T, self.__y[-1])
            H0 = gamma
            d = -self.__compute_search_dir(g, H0, self.__s, self.__y)
            curvature_condition = np.dot(self.__s[-1].T, self.__y[-1])
            if curvature_condition <= 1e-8:
                print("curvature condition: {}".format(curvature_condition))
                raise Exception("Curvature condition is negative")

        phi = phi_function(model, self, w, X, Y, d)
        ls_verbose = False
        if verbose >=3:
            ls_verbose = True
        alpha, ls_log = line_search_wolfe(phi = phi.phi, derphi= phi.derphi, 
                                          phi0 = phi0, old_phi0 = self.__old_phi0, 
                                          c1=self.c1, c2=self.c2, verbose = ls_verbose)

        self.__old_phi0 = phi0
        delta = alpha*d
        w += delta
        model.weights = restore_w_to_model(model, w)

        # l_w1 = restore_w_to_model(model, w1)
        # for i in range(0, len(model.weights)):
        #     regularizer = model.kernel_regularizer[i]*current_batch_size/self.tot_n_patterns
        #     weights_decay = 2*regularizer*model.weights[i]
        #     # weights_decay[0,:] = 0 # In ML the bias should not be regularized
        #     model.weights[i] = l_w1[i] - weights_decay
        
        if( len(self.__s) == self.m and len(self.__y) == self.m):
            self.__s.pop(0)
            self.__y.pop(0)
        # w_new - w_old = w_old + alpha*d - w_old = alpha*d = delta
        self.__s.append(delta) # delta = w_new - w_old
        self.__y.append(g)
        if verbose >= 2:
            print("| alpha: {} | ng: {} | ls conv: {}, it: {}, time: {:4.4f} | zoom used: {}, conv: {}, it: {}|".format(
                    alpha, norm_g, ls_log["ls_conv"], ls_log["ls_it"], ls_log["ls_time"],
                    ls_log["zoom_used"], ls_log["zoom_conv"], ls_log["zoom_it"])) 
        self.__append_history(alpha, norm_g, ls_log)
        return norm_g


    def __compute_search_dir(self, g, H0, s, y):
        q = copy.deepcopy(g)
        a = []
        for s_i, y_i in zip(reversed(s), reversed(y)):
            p = 1/(np.dot(y_i.T, s_i))
            alpha = p*np.dot(s_i.T,q)
            a.append(alpha)
            q -= alpha*y_i
    
        r = H0*q
        for s_i, y_i, a_i in zip(s, y, reversed(a)):
            p = 1/(np.dot(y_i.T, s_i))
            b = p*np.dot(y_i.T,r)
            r += s_i*(a_i -b)
        return r


    def __append_history(self, alpha, norm_g, ls_log):
        self.history["alpha"].append(alpha)
        self.history["norm_g"].append(norm_g)
        self.history["ls_conv"].append(ls_log["ls_conv"])
        self.history["ls_it"].append(ls_log["ls_it"])
        self.history["ls_time"].append(ls_log["ls_time"])
        self.history["zoom_used"].append(ls_log["zoom_used"])
        self.history["zoom_conv"].append(ls_log["zoom_conv"])
        self.history["zoom_it"].append(ls_log["zoom_it"])