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

    def __init__(self, beta_method = "hs", c1=1e-4, c2=.9, restart = None, sfgrd = 0.01, ln_maxiter = 10, tol = None, n_iter_no_change = None):
        super().__init__()
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        if n_iter_no_change is None:
            self.n_iter_no_change = 1
        self.c1 = c1
        self.c2 = c2
        self.sfgrd = sfgrd
        self.old_phi0 = None
        self.past_g = 0
        self.past_d = 0
        self.past_ng = 0
        self.w = 0
        self.fbeta = self.get_beta_function(beta_method)
        self.restart = 0
        self.max_restart = restart
        self.ln_maxiter = ln_maxiter

    def optimize(self, model, epochs, X_train, Y_train, validation_data = None, batch_size = None, es = None, verbose = 0):
        self.model = model
        super().optimize(model, epochs, X_train, Y_train, validation_data=validation_data, batch_size=batch_size, es=es, verbose=verbose)

    def step(self, model, X, Y):
        #input()
        print()
        w = make_vector(model.weights)
        g = make_vector(self.backpropagation(model, model.weights, X, Y))
    
        if ~model.is_fitted and self.epoch == 0:
            d = -g
            phi0 = metrics.mse(Y, model.predict(X))
        else:
            # calcolo del beta
            beta = self.fbeta(g, self.past_g, self.past_ng, self.past_d)
            self.restart +=1
            if self.max_restart is not None and (self.restart == self.max_restart):
                self.restart = 0
                beta = 0
            print("Beta: {}".format(beta), end=" -> compute alpha: ")
            if beta != 0:
                d = - g + beta*self.past_d 
            else:
                d = - g
            phi0 = model.history["loss_mse"][-1]

        self.past_ng = np.linalg.norm(g)       
        self.past_g = g
        self.past_d = d

        phi = phi_function(model, self, w, X, Y, d)
        alpha = line_search_wolfe(phi = phi.phi, derphi= phi.derphi, 
                                  phi0 = phi0, old_phi0 = self.old_phi0, 
                                  c1=self.c1, c2=self.c2, maxiter=self.ln_maxiter)
        #alpha = line_search_wolfe_f(phi = phi.phi, derphi= phi.derphi, phi0 = phi0, c1=self.c1, c2=self.c2)

        print("Alpha: {}".format(alpha), end=" - ")
        print("norm_g: {}".format(self.past_ng))
        self.old_phi0 = phi0
        w += alpha*d
        model.weights = restore_w_to_model(model, w)


    def get_beta_function(self, beta_method):
        if beta_method == "fr":
            return self.beta_fr
        if beta_method == "pr":
            return self.beta_pr
        if beta_method == "hs":
            return self.beta_hs
        if beta_method == "pr+":
            return self.beta_pr_plus
        if beta_method == "hs+":
            return self.beta_hs_plus

    def beta_fr(self, g, past_g, past_norm_g, past_d):
        A = np.dot(g.T,g)
        B = np.dot(past_g.T,past_g)
        beta = np.asscalar(A/B)
        return beta

    def beta_pr(self, g, past_g, past_norm_g, past_d):
        A = g.T
        B = g-past_g
        C = np.square(past_norm_g)
        beta = np.asscalar(np.dot(A,B)/C)
        return beta

    def beta_hs(self, g, past_g, past_norm_g, past_d):
        A = g.T
        B = g-past_g
        beta = np.asscalar(np.dot(A,B)/(np.dot(B.T, past_d)))
        return beta 

    def beta_pr_plus(self, g, past_g, past_norm_g, past_d):
        beta = self.beta_pr(g, past_g, past_norm_g, past_d)
        return max(0, beta)
    
    def beta_hs_plus(self, g, past_g, past_norm_g, past_d):
        beta = self.beta_hs(g, past_g, past_norm_g, past_d)
        return max(0, beta)