import numpy as np

def l_norm(l_v):
    """Computes the norm of a list of numpy array.

    Parameters
    ----------
    l_v : array-like

    Returns
    -------
    float
        The norm of l_v.
    """
    return np.sqrt(np.sum([np.sum(np.square(l_v[i])) for i in range(0, len(l_v))]))

def l_scalar_product(l_v, l_w):
    """Computes the scalar product between two list of numpy array.

    Parameters
    ----------
    l_v : array-like
    l_w : array-like

    Returns
    -------
    float
        The scalar product between l_v and l_w.

    """

    return np.sum([np.sum(np.multiply(l_v[i], l_w[i])) for i in range(0, len(l_v))])

def make_vector(l_v):
    """Takes a list of numpy array and returns a column vector.

    Parameters
    ----------
    l_v : array-like

    Returns
    -------
    array
        Array with dimensions (n, 1).
    """

    row_vector = [l_v[l].flatten() for l in range(len(l_v))]
    return np.concatenate(row_vector).reshape(-1, 1)

def restore_w_to_model(model, w):
    """Takes an array of weights and transforms it into a list 
       of matrices with dimensions taken from the model passed.

    Parameters
    ----------
    model : isanet.model.MLP
            The Multilayer Perceptron object.
    w : array

    Returns
    -------
        array-like
    """
    start = 0
    weights = [0]*model.n_layers
    for i in range(model.n_layers):
        n_rows = model.weights[i].shape[0]
        n_cols = model.weights[i].shape[1]
        end = n_rows*n_cols
        weights[i] = w[start:start + end].reshape(n_rows,n_cols)
        start = start + end  
    return weights
