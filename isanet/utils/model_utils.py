import numpy as np
"""Utils Moduel.
"""
import matplotlib.pyplot as plt
import pickle

def plotHistory(history, loss = 'MSE', orientation = "horizontal"):
    """Plots the learning curve for the training and the validation
    for for a specific loss and accuracy.

    Parameters
    ----------
    history : dict
        It contains for each epoch the values of mse, mee and accuracy for 
        training and validation and the time.
    
    loss : string, accepted value == ("MSE", "MEE")
        Specifies what type of loss to use.

    orientation : None, "horizontal", "vertical"
        Indicates the orientation of the two plots.
    """
    pos_train = (0,0)
    if orientation == "horizontal":
        figsize = (12, 4)
        figdims = (1, 2)
        pos_val = (0, 1)
    elif orientation == "vertical":
        figsize = (7, 7)
        figdims = (2, 1)
        pos_val = (1, 0)
    else:
        raise Exception('Wrong value for orientation par.')

    fig = plt.figure(figsize=figsize) 
    fig_dims = figdims
    plt.subplot2grid(fig_dims, pos_train)
    plt.plot(history["loss_mse"])
    plt.plot(history["val_loss_mse"], linestyle='--')
    plt.title('MSE')
    plt.ylabel(loss)
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['Training', 'Test'], loc='upper right', fontsize='large')

    plt.subplot2grid(fig_dims, pos_val)
    plt.plot(history['acc'])
    plt.plot(history['val_acc'], linestyle='--')
    plt.title('Accuracy %')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['Training', 'Test'], loc='lower right', fontsize='large')
    plt.tight_layout()
    plt.show()

def plotMse(history):
    """Plots the learning curve for the training and the validation
    for for MSE.

    Parameters
    ----------
    history : dict
        It contains for each epoch the values of mse, mee and accuracy for 
        training and validation and the time.
    """
    plt.plot(history["loss_mse"])
    plt.plot(history["val_loss_mse"], linestyle='--')
    plt.title('MSE')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['Training', 'Test'], loc='upper right', fontsize='large')
    plt.show()

def printMSE(y_pred, y_true, type = ""):
    """Print to the stout the MSE value.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    
    type : string
        Useful to specify what kind of mse you are going to print. 
        E.g. Train, Test or Val.
    """
    delta = y_true - y_pred
    error = np.mean(np.square(delta))#/2
    print("MSE {}: {} ".format(type, error))

def printMEE(y_pred, y_true, type = ""):
    """Print to the stout the MEE value.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    
    type : string
        Useful to specify what kind of mse you are going to print. 
        E.g. Train, Test or Val.
    """
    delta = y_true - y_pred
    error = np.mean(np.sqrt(np.sum(np.square(delta),1)))
    print("MEE {}: {} ".format(type, error))

def printAcc(y_pred, y_true, type = ""):
    """Print to the stout the MSE value.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    type : string
        Useful to specify what kind of mse you are going to print. 
        E.g. Train, Test or Val.
    """
    acc = np.mean(((y_pred > .5) == y_true).all(1))
    print("Accuracy {}: {} ".format(type, acc))

def save_data(data, filename):
    """Serialize and save the past object on file.

    Parameters
    ----------
    data : object
        Data to serialize.

    filename : string
        Specifies the name of the file to save to.
    """   
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_data(filename):
    """Deserialize and load an object from a specific path.

    Parameters
    ----------
    filename : string
        Specifies the name of the file to load to.

    Returns
    -------
    objct
        the deserialized object returns
    """   
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data