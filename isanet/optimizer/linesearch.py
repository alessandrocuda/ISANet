import numpy as np
import time
import copy
import isanet.metrics as metrics
from isanet.optimizer.utils import make_vector, restore_w_to_model

def line_search_wolfe_f(phi, derphi, phi0=None, c1=1e-4, c2=0.9):
    """ a fast strong Wolfe line search 
    """
    phip0 = derphi(0)
    a_s = 1
    feval = 0
    mina = 1e-16
    sfgrd = 0.01
    while feval <= 1000:
        feval+=1
        phia  = phi(a_s)
        phips = derphi(a_s)

        if ( phia <= phi0 + c1 * a_s * phip0) & (np.abs( phips ) <= - c2 * phip0):
            return a_s
         
        if phips >= 0:
            print("derivative is positive", end=" - ")
            break
        a_s = a_s / 0.9
    
    am = 0
    a = a_s
    phipm = phip0
    while ( feval <= 1000 ) & ( ( a_s - am )  > mina) & ( phips > 1e-12 ):
        
        # compute the new value by safeguarded quadratic interpolation
        a = ( am * phips - a_s * phipm ) / ( phips - phipm )
        a = max( [ am + ( a_s - am ) * sfgrd,
                 min( [ a_s - ( a_s - am ) * sfgrd,  a ] ) ] )
    
        # compute phi( a )
        phia = phi(a)
        phip = derphi(a)
        feval+=1
        if ( phia <= phi0 + c1 * a * phip0 ) & ( np.abs( phip ) <= - c2 * phip0 ):
            # Armijo + strong Wolfe satisfied, we are done
            return a
    
       # restrict the interval based on sign of the derivative in a
        if phip < 0:
           am = a
           phipm = phip
        else:
           a_s = a
           if a_s <= mina:
              break
           phips = phip

    if a <= mina:
        print("error")
        exit
    else:
        return a

class phi_function(object):
    def __init__(self, model, optimizer, w, X, Y, d):
        self.optimizer = optimizer
        self.model = model
        self.w = w
        self.X = X
        self.Y = Y
        self.d = d

    def phi(self, a):
        w_a = restore_w_to_model(self.model, self.w+a*self.d)
        phia = metrics.mse(self.Y, self.optimizer.forward(w_a, self.X))
        return phia
        
    def derphi(self, a):
        w_a = restore_w_to_model(self.model, self.w+a*self.d)
        g_a = make_vector(self.optimizer.backpropagation(self.model, w_a, self.X, self.Y))
        phips = np.asscalar(np.dot(g_a.T, self.d))
        return phips

class LineSearch(object):

    def __init__(self, phi = None):
        if phi is None:
            raise Exception("A Phi object must be provided")
        self.phi = phi

    def set_phi(self):
        pass

    def strong_wolfe(self, phi0=None,
                         old_phi0=None, derphi0=None,
                         c1=1e-4, c2=0.9, amax=None, maxiter=10):
        pass


def line_search_wolfe(phi, derphi, phi0=None,
                         old_phi0=None, derphi0=None,
                         c1=1e-4, c2=0.9, amax=None, maxiter=10):
    """Return alpha > 0 that satisfies strong Wolfe conditions in order
    to get a descent direction or the last alpha found if the line search 
    algorithm did not converge.
    For major details on the implementation refer to Wright and Nocedal,
    'Numerical Optimization', 1999, pp. 59-61.

    Parameters
    ----------
    phi : callable phi(alpha)
        Objective scalar function.
    derphi : callable phi'(alpha)
        Objective function derivative. Returns a scalar.
    phi0 : float, optional
        Value of phi at 0.
    old_phi0 : float, optional
        Value of phi at previous point.
    derphi0 : float, optional
        Value of derphi at 0
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax : float, optional
        Maximum step size.
    maxiter : int, optional
        Maximum number of iterations to perform.
    Returns
    -------
    alpha_star : float
        Best alpha, or last alpha if the line search algorithm did not converge.
    """
    
    # data struct used to log the behavior of the line search
    ls_log = {"ls_conv": "y",
              "ls_it": 0,
              "ls_time": 0,
              "zoom_used": "n",
              "zoom_conv": "-",
              "zoom_it": 0 } 


    if phi0 is None:
        phi0 = phi(0.)

    if derphi0 is None:
        derphi0 = derphi(0.)

    alpha0 = 0

    alpha1 = 1.0

    if alpha1 < 0:
        alpha1 = 1.0

    if amax is not None:
        alpha1 = min(alpha1, amax)

    phi_a1 = phi(alpha1)

    phi_a0 = phi0
    derphi_a0 = derphi0


    # for i in range(maxiter):
    start_time = time.time()

    i = 0
    while i < maxiter:

        if (phi_a1 > phi0 + c1 * alpha1 * derphi0) or ((phi_a1 >= phi_a0) and (i > 1)):
            alpha_star, zoom_log = _zoom(alpha0, alpha1, phi_a0,
                                         phi_a1, derphi_a0, phi, derphi,
                                         phi0, derphi0, c1, c2)
            ls_log["zoom_used"] = "y"
            ls_log["zoom_conv"] = zoom_log["zoom_conv"]
            ls_log["zoom_it"] = zoom_log["zoom_it"]
            break

        derphi_a1 = derphi(alpha1)
        if (abs(derphi_a1) <= -c2*derphi0):
            alpha_star = alpha1
            break

        if (derphi_a1 >= 0):
            alpha_star, zoom_log = _zoom(alpha1, alpha0, phi_a1,
                                         phi_a0, derphi_a1, phi, derphi,
                                         phi0, derphi0, c1, c2)
            ls_log["zoom_used"] = "y"
            ls_log["zoom_conv"] = zoom_log["zoom_conv"]
            ls_log["zoom_it"] = zoom_log["zoom_it"]
            break

        alpha0 = alpha1
        alpha1 = 2 * alpha1  # increase by factor of two on each iteration
        if amax is not None:
            alpha1 = min(alpha1, amax)
        phi_a0 = phi_a1
        phi_a1 = phi(alpha1)
        derphi_a0 = derphi_a1
        
        i += 1

    else:
        # stopping test maxiter reached
        alpha_star = alpha1
        ls_log["ls_conv"] = "n"
    if ls_log["zoom_conv"] is "n":
        ls_log["ls_conv"] = "n"
    ls_log["ls_it"] = i
    ls_log["ls_time"] = (time.time() - start_time)
    return alpha_star, ls_log


def _cubicmin(a, fa, fpa, b, fb, c, fc):
    """
    Finds the minimizer for a cubic polynomial that goes through the
    points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.
    If no minimizer can be found, return None.
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
    the points (a,fa), (b,fb) with derivative at a of fpa.
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


def _zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo,
          phi, derphi, phi0, derphi0, c1, c2):
    """Zoom function of linesearch satisfying strong Wolfe conditions.
    For major details on the implementation refer to Wright and Nocedal,
    'Numerical Optimization', 1999, pp. 59-61. For the interpolation step
    refer to scipy.
    """

    zoom_log = {}

    maxiter = 10
    i = 0
    delta1 = 0.2  # cubic interpolant check
    delta2 = 0.1  # quadratic interpolant check
    phi_rec = phi0
    a_rec = 0
    while i < maxiter:
        
        # interpolate to find a trial step length between a_lo and
        # a_hi Need to choose interpolation here. Use cubic
        # interpolation and then if the result is within delta *
        # dalpha or outside of the interval bounded by a_lo or a_hi
        # then use quadratic interpolation, if the result is still too
        # close, then use bisection

        dalpha = a_hi - a_lo
        if dalpha < 0:
            a, b = a_hi, a_lo
        else:
            a, b = a_lo, a_hi

        # minimizer of cubic interpolant
        # (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
        #
        # if the result is too close to the end points (or out of the
        # interval), then use quadratic interpolation with phi_lo,
        # derphi_lo and phi_hi if the result is still too close to the
        # end points (or out of the interval) then use bisection

        if (i > 0):
            cchk = delta1 * dalpha
            a_j = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi,
                            a_rec, phi_rec)
        if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
            qchk = delta2 * dalpha
            a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
            if (a_j is None) or (a_j > b-qchk) or (a_j < a+qchk):
                a_j = a_lo + 0.5*dalpha

        # Check new value of a_j

        phi_aj = phi(a_j)
        if (phi_aj > phi0 + c1*a_j*derphi0) or (phi_aj >= phi_lo):
            phi_rec = phi_hi
            a_rec = a_hi
            a_hi = a_j
            phi_hi = phi_aj
        else:
            derphi_aj = derphi(a_j)
            if abs(derphi_aj) <= -c2*derphi0:
                zoom_log["zoom_conv"] = "y"
                zoom_log["zoom_it"] = i
                return a_j, zoom_log
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
    # Failed to find a conforming step size
    # return last a_j
    zoom_log["zoom_conv"] = "n"
    zoom_log["zoom_it"] = i
    return a_j, zoom_log