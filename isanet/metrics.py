"""Metrics Module.
"""

import numpy as np
from isanet.optimizer.utils import l_norm


def mse_reg(y_true, y_pred, model, weights):
    """MSE + L2 regularization

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
        
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    model : isanet.model.MLP

    weights : list
        List of arrays, the ith array represents all the 
        weights of each neuron in the ith layer.

    Returns
    -------
    loss : float
        A non-negative floating point value (the best value is 0.0)
    """
    return np.mean(np.square(y_true - y_pred)) \
        + model.kernel_regularizer[0]*np.square(l_norm(weights))

def mse(y_true, y_pred):
    """Mean squared error regression loss

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
        
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    Returns
    -------
    loss : float
        A non-negative floating point value (the best value is 0.0)
    """
    delta = y_true - y_pred
    return np.mean(np.square(delta))#/2

def mee(y_true, y_pred):
    """Mean Euclidean error regression loss

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    Returns
    -------
    loss : float
        A non-negative floating point value (the best value is 0.0)
    """
    delta = y_true - y_pred
    return  np.mean(np.sqrt(np.sum(np.square(delta),1)))

def accuracy_binary(y_true, y_pred):
    """Accuracy classification score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    Returns
    -------
    loss : float
        A non-negative floating point value (the best value is 100.0)
    """
    return np.mean(((y_pred > .5) == y_true).all(1))