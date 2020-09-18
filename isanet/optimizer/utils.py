import numpy as np

def l_norm(l_v):
    # compute the norm of a list of numpy array
    return np.sqrt(np.sum([np.sum(np.square(l_v[i])) for i in range(0, len(l_v))]))

def l_scalar_product(l_v, l_w):
    # compute the scalar product between two list of numpy array
    return np.sum([np.sum(np.multiply(l_v[i], l_w[i])) for i in range(0, len(l_v))])

def make_vector(l_v):
    #take a list of numpy array and return a collum vector
    row_vector = [l_v[l].flatten() for l in range(len(l_v))]
    return np.concatenate(row_vector).reshape(-1, 1)

def restore_w_to_model(model, w):
    start = 0
    weights = [0]*model.n_layers
    for i in range(model.n_layers):
        n_rows = model.weights[i].shape[0]
        n_cols = model.weights[i].shape[1]
        end = n_rows*n_cols
        weights[i] = w[start:start + end].reshape(n_rows,n_cols)
        start = start = start + end  
    return weights
