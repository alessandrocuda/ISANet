import numpy as np
import time
import copy
import isanet.metrics as metrics
from isanet.optimizer.utils import make_vector, restore_w_to_model

def line_search_wolfe(f, myfprime, xk, pk, gfk=None, old_fval=None,
                       old_old_fval=None, args=(), c1=1e-4, c2=0.4, amax=None, maxiter=10):
    
    raise NotImplementedError



def armijo_wolfe_ls(o, model, w, X, Y, phi0, old_phi0, g, d, c1, c2, max_iter = 10):

    derphi0 = np.asscalar(np.dot(g.T,d))

    alpha0 = 0.
    amax = 100000000000.
    alpha1 = 0.1
    # if old_phi0 is not None and derphi0 != 0:
    #     alpha1 = min(1.0, 1.01*2*(phi0 - old_phi0)/derphi0)
    # else:
    #     alpha1 = 1.0

    # if alpha1 < 0.000000000000001:
    #     alpha1 = 1.0

    # if amax is not None:
    #     alpha1 = min(alpha1, amax)

    phi_a0 = phi0

    iter = 1

    while iter <= 1000000:
        #print(iter)
        # Evaluate phi(alpha)
        if alpha1 == 0 or (alpha0 == amax):
            print("errro? why? frangio help us")
        
        w1 = restore_w_to_model(model, w+alpha1*d)
        phi_a1 = metrics.mse(Y, o.forward( w1, X))
        if (phi_a1 > phi0 + c1 * alpha1 * derphi0) or ((phi_a1 >= phi_a0) and (iter > 1)):
            print("zoom1: {0}, {1}".format(alpha0, alpha1), end=", ")
            return zoom(o, w, alpha0, alpha1, phi0, derphi0, c1, c2, model, X, Y, d)

        g_a1 = make_vector(o.backpropagation(model, w1, X, Y))
        derphi_a1 = np.asscalar(np.dot(g_a1.T, d))

        if (np.abs(derphi_a1) <= (-c2*derphi0)):
            print("f-w-verified", end=", ")
            return alpha1

        if (derphi_a1 >= 0):
            print("dev > 0, zoom2: {0}, {1}".format(alpha1, alpha0), end=", ")
            return zoom(o, w, alpha1, alpha0, phi0, derphi0, c1, c2, model, X, Y, d)
        
        alpha2 = 2 * alpha1  
        if amax is not None:
            alpha2 = min(alpha2, amax)
        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1

        iter += 1
    print("Max iter reached", end=", ")
    return alpha1

def zoom(o, w, a_lo, a_hi, phi0, derphi0, c1, c2, model, X, Y, d, maxiter=10):
    delta1 = 0.2  # cubic interpolant check
    delta2 = 0.1  # quadratic interpolant check
    phi_rec = phi0
    a_rec = 0


    w_lo = restore_w_to_model(model, w + a_lo * d)
    w_hi = restore_w_to_model(model, w + a_hi * d)
    
    phi_lo = metrics.mse(Y, o.forward(w_lo, X))
    phi_hi = metrics.mse(Y, o.forward( w_hi, X))

    g_lo = make_vector(o.backpropagation(model, w_lo, X, Y))
    derphi_lo = np.asscalar(np.dot(g_lo.T, d))

    i = 0
    alpha_j = 0
    
    while i < maxiter:
        # quadratic interpolation
        dalpha = a_hi - a_lo
        if dalpha < 0:
            a, b = a_hi, a_lo
        else:
            a, b = a_lo, a_hi

        if (i > 0):
            cchk = delta1 * dalpha
            a_j = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi,
                            a_rec, phi_rec)
        if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
            qchk = delta2 * dalpha
            a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
            if (a_j is None) or (a_j > b-qchk) or (a_j < a+qchk):
                a_j = a_lo + 0.5*dalpha
        
        # safeguarded
        #a = max( [ am + ( as - am ) * sfgrd, min( [ as - ( as - am ) * sfgrd,  a ] ) ])

        # alpha_j = np.max( [ alpha_lo + (alpha_hi - alpha_lo)*self.sfgrd, 
        #                 np.min([ alpha_hi - (alpha_hi - alpha_lo)*self.sfgrd, alpha_j])])
                                        
        w_j = restore_w_to_model(model, w + a_j * d)
        phi_aj = metrics.mse(Y, o.forward(w_j, X))

        if (phi_aj > phi0 + c1*a_j*derphi0) or (phi_aj >= phi_lo):
            phi_rec = phi_hi
            a_rec = a_hi
            a_hi = a_j
            phi_hi = phi_aj
        else:
            g_aj = make_vector(o.backpropagation(model, w_j, X, Y))
            derphi_aj = np.asscalar(np.dot(g_aj.T,d))
            if abs(derphi_aj) <= -c2*derphi0:
                return a_j
            if derphi_aj*(a_hi - a_lo) >= 0:
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_lo
                phi_hi = phi_lo
            else:
                phi_rec = phi_lo
                a_rec = a_lo
            a_lo = a_j
            phi_lo = phi_aj
            derphi_lo = derphi_aj                 
        i += 1

    return alpha_j

def _cubicmin(a, fa, fpa, b, fb, c, fc):
    """
    Finds the minimizer for a cubic polynomial that goes through the
    points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.
    If no minimizer can be found return None
    """
    # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D

    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            C = fpa
            db = b - a
            dc = c - a
            denom = (db * dc) ** 2 * (db - dc)
            d1 = np.empty((2, 2))
            d1[0, 0] = dc ** 2
            d1[0, 1] = -db ** 2
            d1[1, 0] = -dc ** 3
            d1[1, 1] = db ** 3
            [A, B] = np.dot(d1, np.asarray([fb - fa - C * db,
                                            fc - fa - C * dc]).flatten())
            A /= denom
            B /= denom
            radical = B * B - 3 * A * C
            xmin = a + (-B + np.sqrt(radical)) / (3 * A)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin

def _quadmin(a, fa, fpa, b, fb):
    """
    Finds the minimizer for a quadratic polynomial that goes through
    the points (a,fa), (b,fb) with derivative at a of fpa,
    """
    # f(x) = B*(x-a)^2 + C*(x-a) + D
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            D = fa
            C = fpa
            db = b - a * 1.0
            B = (fb - D - C * db) / (db * db)
            xmin = a - C / (2.0 * B)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin
