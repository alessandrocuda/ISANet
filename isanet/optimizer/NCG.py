""" Optimizer Module.
"""
import numpy as np
import time
import copy
import isanet.metrics as metrics
from isanet.optimizer import Optimizer
from isanet.optimizer.linesearch import line_search_wolfe, line_search_wolfe_f, phi_function
from isanet.optimizer.utils import make_vector, restore_w_to_model

class NCG(Optimizer):
    """Nonlinear Conjugate Gradient (NCG).

    Parameters
    ----------
    beta_method : string, default="hs+"
        Beta formulas available for the NCG.

        - 'fr', Fletcher-Reeves formula.
        - 'pr', Polak-Ribière formula.
        - 'hs', Hestenes-Stiefel formula.
        - 'pr+', modified Polak-Ribière formula.
        - 'hs+', modified Hestenes-Stiefel formula.

    c1 : float, default=1e-4
        Parameter for the Armijo-Wolfe line search.

    c2 : float, default=0.9
        Parameter for the Armijo-Wolfe line search.

    restart : integer, optional
        Every 'restart' iterations Beta is set to 0.

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

    history : dict
        Save for each iteration some interesting values.

        Dictionary's keys:
            ``beta``
                Beta value. 
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

    Methods
    -------
    optimize(self, model, epochs, X_train, Y_train, validation_data, batch_size, es, verbose)

    forward(self, weights, X)
        Uses the weights passed to the function to make the Feed-Forward step.

    backpropagation(self, model, weights, X, Y)
        Computes the derivative of 1/n sum_n (y_i -y_i') + L2 regularization.
       
    step(self, model, X, Y, verbose)
        NCG algorithm.

    """

    def __init__(self, beta_method = "hs+", c1=1e-4, c2=.9, restart = None,
                 ln_maxiter = 10, tol = None, n_iter_no_change = None, 
                 norm_g_eps = None, l_eps = None, debug = False):
        super().__init__(tol = tol, n_iter_no_change = n_iter_no_change, norm_g_eps = norm_g_eps, l_eps = l_eps, debug = debug)
        self.c1 = c1
        self.c2 = c2
        self.old_phi0 = None
        self.past_g = 0
        self.past_d = 0
        self.past_ng = 0
        self.w = 0
        self.fbeta = self.__get_beta_function(beta_method)
        self.restart = 0
        self.max_restart = restart
        self.ln_maxiter = ln_maxiter

        self.history = {"beta":         [],
                        "alpha":        [],
                        "norm_g":      [],
                        "ls_conv":      [],
                        "ls_it":        [],
                        "ls_time":      [],
                        "zoom_used":    [],
                        "zoom_conv":    [],
                        "zoom_it":      []} 

    def backpropagation(self, model, weights, X, Y):
        g = super().backpropagation(model, weights, X, Y)
        for i in range(len(g)):
            g[i]  = (2/X.shape[0])*g[i] + (2*model.kernel_regularizer[0])*weights[i]
        return g

    def step(self, model, X, Y, verbose):
        current_batch_size = X.shape[0]
        w = make_vector(model.weights)
        g = make_vector(self.backpropagation(model, model.weights, X, Y))
        norm_g = np.linalg.norm(g)

        if ~model.is_fitted and self.epoch == 0:
            beta = 0
            d = -g
            phi0 = metrics.mse_reg(Y, model.predict(X), model, model.weights)
        else:
            # calcolo del beta
            beta = self.fbeta(g, self.past_g, self.past_ng, self.past_d)
            self.restart +=1
            if self.max_restart is not None and (self.restart == self.max_restart):
                self.restart = 0
                beta = 0
            if beta != 0:
                d = - g + beta*self.past_d 
            else:
                d = - g
            phi0 = model.history["loss_mse_reg"][-1]

        self.past_ng = norm_g
        self.past_g = g
        self.past_d = d
        derphi0 = np.asscalar(np.dot(g.T, d))

        phi = phi_function(model, self, w, X, Y, d)
        ls_verbose = False
        if verbose >=3:
            ls_verbose = True
        alpha, ls_log = line_search_wolfe(phi = phi.phi, derphi= phi.derphi, 
                                  phi0 = phi0, old_phi0 = self.old_phi0, derphi0 = derphi0,
                                  c1=self.c1, c2=self.c2, maxiter=self.ln_maxiter, verbose = ls_verbose)

        self.old_phi0 = phi0

        w += alpha*d
        model.weights = restore_w_to_model(model, w)
        
        if verbose >= 2:
            print("| beta: {} | alpha: {} | ng: {} | ls conv: {}, it: {}, time: {:4.4f} | zoom used: {}, conv: {}, it: {}|".format(
                    beta, alpha, norm_g, ls_log["ls_conv"], ls_log["ls_it"], ls_log["ls_time"],
                    ls_log["zoom_used"], ls_log["zoom_conv"], ls_log["zoom_it"])) 
        self.__append_history(beta, alpha, norm_g, ls_log)
        return norm_g


    def __get_beta_function(self, beta_method):
        if beta_method == "fr":
            return self.__beta_fr
        if beta_method == "pr":
            return self.__beta_pr
        if beta_method == "hs":
            return self.__beta_hs
        if beta_method == "pr+":
            return self.__beta_pr_plus
        if beta_method == "hs+":
            return self.__beta_hs_plus

    def __beta_fr(self, g, past_g, past_norm_g, past_d):
        A = np.dot(g.T,g)
        B = np.dot(past_g.T,past_g)
        beta = np.asscalar(A/B)
        return beta

    def __beta_pr(self, g, past_g, past_norm_g, past_d):
        A = g.T
        B = g-past_g
        C = np.square(past_norm_g)
        beta = np.asscalar(np.dot(A,B)/C)
        return beta

    def __beta_hs(self, g, past_g, past_norm_g, past_d):
        A = g.T
        B = g-past_g
        beta = np.asscalar(np.dot(A,B)/(np.dot(B.T, past_d)))
        return beta 

    def __beta_pr_plus(self, g, past_g, past_norm_g, past_d):
        beta = self.__beta_pr(g, past_g, past_norm_g, past_d)
        return max(0, beta)
    
    def __beta_hs_plus(self, g, past_g, past_norm_g, past_d):
        beta = self.__beta_hs(g, past_g, past_norm_g, past_d)
        return max(0, beta)

    def __append_history(self, beta, alpha, norm_g, ls_log):
        self.history["beta"].append(beta)
        self.history["alpha"].append(alpha)
        self.history["norm_g"].append(norm_g)
        self.history["ls_conv"].append(ls_log["ls_conv"])
        self.history["ls_it"].append(ls_log["ls_it"])
        self.history["ls_time"].append(ls_log["ls_time"])
        self.history["zoom_used"].append(ls_log["zoom_used"])
        self.history["zoom_conv"].append(ls_log["zoom_conv"])
        self.history["zoom_it"].append(ls_log["zoom_it"])