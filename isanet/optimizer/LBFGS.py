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

    def __init__(self, m = 3, c1=1e-4, c2=.9, ln_maxiter = 10, tol = None, 
                 n_iter_no_change = None, norm_g_eps = None, l_eps = None, 
                 debug = False):
        super().__init__(tol = tol, n_iter_no_change = n_iter_no_change, norm_g_eps = norm_g_eps, l_eps = l_eps, debug = debug)
        self.c1 = c1
        self.c2 = c2
        self.old_phi0 = None
        self.n_vars = 0
        self.past_g = 0
        self.past_d = 0
        self.past_ng = 0
        self.w = 0
        self.restart = 0
        self.s = []
        self.y = []
        self.m = m
        self.ln_maxiter = ln_maxiter

        self.history = {"alpha":        [],
                        "norm_g":       [],
                        "ls_conv":      [],
                        "ls_it":        [],
                        "ls_time":      [],
                        "zoom_used":    [],
                        "zoom_conv":    [],
                        "zoom_it":      []} 

    def optimize(self, model, epochs, X_train, Y_train, validation_data = None, batch_size = None, es = None, verbose = 0):
        self.model = model
        for i in range(len(model.weights)):
            self.n_vars += model.weights[i].shape[0]*model.weights[i].shape[1]
        self.H0 = np.eye(self.n_vars)
        super().optimize(model, epochs, X_train, Y_train, validation_data=validation_data, batch_size=batch_size, es=es, verbose=verbose)

    def step(self, model, X, Y, verbose):
        w0 = make_vector(model.weights)
        g = make_vector(self.backpropagation(model, model.weights, X, Y))
        norm_g = np.linalg.norm(g)
        if ~model.is_fitted and self.epoch == 0:
            d = - g
            phi0 = metrics.mse(Y, model.predict(X))
        else:
            self.y[-1] = g - self.y[-1]
            gamma = np.dot(self.s[-1].T, self.y[-1])/np.dot(self.y[-1].T, self.y[-1])
            H0 = gamma*np.eye(self.n_vars)
            d = -self.compute_search_dir(g, H0, self.s, self.y)
            curvature_condition = np.dot(self.s[-1].T, self.y[-1])
            if(curvature_condition < 0):
                print("curvature condition: {}".format(curvature_condition))
                raise Exception("Curvature condition is negative")
            phi0 = model.history["loss_mse"][-1]

        phi = phi_function(model, self, w0, X, Y, d)
        alpha, ls_log = line_search_wolfe(phi = phi.phi, derphi= phi.derphi, phi0 = phi0, old_phi0 = self.old_phi0, c1=self.c1, c2=self.c2)
        #alpha = line_search_wolfe_f(phi = phi.phi, derphi= phi.derphi, phi0 = phi0, c1=self.c1, c2=self.c2)

        self.old_phi0 = phi0
        w1 = w0 + alpha*d
        model.weights = restore_w_to_model(model, w1)
        if( len(self.s) == self.m and len(self.y) == self.m):
            self.s.pop(0)
            self.y.pop(0)
        self.s.append(w1 - w0)
        self.y.append(g)
        if verbose >= 2:
            print("| alpha: {} | ng: {} | ls conv: {}, it: {}, time: {:4.4f} | zoom used: {}, conv: {}, it: {}|".format(
                    alpha, norm_g, ls_log["ls_conv"], ls_log["ls_it"], ls_log["ls_time"],
                    ls_log["zoom_used"], ls_log["zoom_conv"], ls_log["zoom_it"])) 
        append_history(self, alpha, norm_g, ls_log)
        return 


    def compute_search_dir(self, g, H0, s, y):
        q = copy.deepcopy(g)
        a = []
        for s_i, y_i in zip(reversed(s), reversed(y)):
            p = 1/(np.dot(y_i.T, s_i))
            alpha = p*np.dot(s_i.T,q)
            a.append(alpha)
            q -= alpha*y_i
    
        r = np.dot(H0, q)
        for s_i, y_i, a_i in zip(s, y, reversed(a)):
            p = 1/(np.dot(y_i.T, s_i))
            b = p*np.dot(y_i.T,r)
            r += s_i*(a_i -b)
        return r


def append_history(self, alpha, norm_g, ls_log):
    self.history["alpha"].append(alpha)
    self.history["norm_g"].append(norm_g)
    self.history["ls_conv"].append(ls_log["ls_conv"])
    self.history["ls_it"].append(ls_log["ls_it"])
    self.history["ls_time"].append(ls_log["ls_time"])
    self.history["zoom_used"].append(ls_log["zoom_used"])
    self.history["zoom_conv"].append(ls_log["zoom_conv"])
    self.history["zoom_it"].append(ls_log["zoom_it"])