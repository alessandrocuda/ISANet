import numpy as np
from . import optimizer 
from . import activation
from . import metrics

def random_weights_layer(size, std = 0.06):
    """Initializes neural net weights with a uniformly 
    distributed over the half-open interval [-std, std)

    Parameters
    ----------
    size : tuple 
        This tuple represents the matrix size of a given layer of the neural network.
        E.g. dimension = (dim_input_layer, n_units), where 'dim_input_layer' is the number
        of input at that level and 'n_units' is the number of neurons in that level.

    std : float, default = 0.06
        Defines the standard deviation value for the distribution.
    
    Returns
    -------
    Array-like
        Where each column represents the input weights to each neuron.
    """
    return np.random.uniform(-std, std, size)

class Mlp():
    """This class implements the Multi-layer Perceptron.

    Attributes
    ----------
    weights : list
        List of arrays, the ith array represents all the 
        weights of each neuron in the ith layer.

    activations : list
        List of Activation class, the ith Activation represents
        the activation function for that layer.
    
    kernel_regularizer : list
        List of float value where ith element represents the 
        l2 norm regularitation term for the ith layer.
    
    n_layers : int
        Represents the total number of layers - 1 
        (hidden layers + output layer)
    
    history : dict
        Save for each epoch the values of mse, mee and accuracy for 
        training and validation and the time taken to compute that epoch.

        Dictionary's keys:
            ``loss_mse``
                The mean square error on training for each epochs. 
            ``loss_mee``
                The mean euclidean error on training for each epochs. 
            ``val_loss_mse``
                The mean square error on validation for each epochs. 
            ``val_loss_mee``
                The mean euclidean error on validation for each epochs.  
            ``acc``
                The accuracy on training for each epochs.
            ``val_acc``
                The accuracy on validation for each epochs.
            ``epoch_time``
                The time elapsed for compute for each epochs.
                
        A dict with keys as column headers and values as columns, that can be imported into a pandas DataFrame.                   
        For instance the below given table::

                +-------------------------+---------------+----------------+---------------+---+-----------------+
                |         loss_mse        |   loss_mee    |  val_loss_mse  | val_loss_mee  |...|   epoch_time    |
                +=========================+===============+================+===============+===+=================+
                |          0.985          |     0.992     |      1.00      |      1.00     |...|     0.0030      |   
                +-------------------------+---------------+----------------+---------------+---+-----------------+
                |          0.984          |     0.991     |      1.00      |      1.00     |...|     0.0029      |
                +-------------------------+---------------+----------------+---------------+---+-----------------+

    Notes
    -----
    Mlp trains iteratively since at each time step the partial derivatives of the loss function with 
    respect to the model parameters are computed to update the parameters. It can also have a regularization 
    term added to the loss function that shrinks model parameters to prevent overfitting.
    """
    
    def __init__(self):
        self.weights = []
        self.activations = []
        self.kernel_regularizer = []
        self.n_layers = 0  #hidden + out
        self.history = {"loss_mse":     [],
                        "loss_mee":     [], 
                        "val_loss_mse": [], 
                        "val_loss_mee": [], 
                        "acc":          [], 
                        "val_acc":      [], 
                        "epoch_time":   []}

        self.__optimizer = optimizer.SGD()
        self.__is_input_layer_set = False

    def fit(self, X_train, Y_train, epochs = 0, batch_size = None , validation_data = None, es = False, verbose = 0):
        """Fit the model to data matrix X_train and target(s) Y_train and evaluates it on the validation set (if there is one).

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            The input data.

        Y_train : array-like of shape (n_samples,)
            The target values.

        epochs : integer, default=0
            Maximum number of epochs.

        batch_size : integer, optional
            Size of minibatches for the optimizers.
            When set to "none", the optimizer will performe a full batch.

        validation_data : list of arrays-like, [X_val, Y_val], optional
            Validation set.

        es : bool or isanet.callbacks.EarlyStopping, default=False
            When set to False it will only use the ``epochs`` to finish training.
            Otherwise, an EarlyStopping type object has been passed and will stop 
            training if the model goes overfitting after a number of consecutive iterations.
            See docs in optimizier module.

        verbose : integer, default=0
            Controls the verbosity: the higher, the more messages.

        Raises
        ------
        Error when checking input
            If the number of columns in the input matrix differs from the number of rows 
            in the first weight matrix (weights[0]).

        Error when checking output
            If the number of columns in the target matrix differs from the number of rows 
            in the last weight matrix (weights[-1]).
        """
        #####################################
        # Check data dimension with NN dims 
        #####################################
        if X_train.shape[1] != (self.weights[0].shape[0] - 1):
            raise Exception("Error when checking input: expected input shape ({},) but got first layer shape ({},)".format(self.weights[0].shape[0] - 1, X_train.shape[1]))
        if Y_train.shape[1] != (self.weights[-1].shape[1]):
            raise Exception("Error when checking output: expected ouput shape (,{}) but got output layer shape (,{})".format(Y_train.shape[1], self.weights[-1].shape[1]))
               
        ####################################
        #          Start learning          
        ####################################
        self.__optimizer.optimize(model=self, epochs=epochs, batch_size=batch_size, verbose=verbose,
                                X_train=X_train, 
                                Y_train=Y_train, 
                                validation_data=validation_data,
                                es=es)
                        
    def __forward(self, layer, input):
        """Does the dot product between the input and the weight matrix of the specified layer"""

        z = np.dot(np.insert(input, 0, 1, 1) ,self.weights[layer])
        return self.activations[layer].f(z)

    def append_history(self, history, is_validation_set):
        """Adds results on the training and validation sets to the history.

        Parameters
        ----------
        history : dict
            Save for each epoch the values of mse, mee and accuracy for 
            training and validation and the time taken to compute that epoch.

        validation : boolean
            True to append validation history, False do not.
        """
        self.history["loss_mse"].append(history["mse_train"])
        self.history["loss_mee"].append(history["mee_train"])
        self.history["acc"].append(history["acc_train"])
        if is_validation_set:
            self.history["val_loss_mse"].append(history["mse_val"])
            self.history["val_loss_mee"].append(history["mee_val"])
            self.history["val_acc"].append(history["acc_val"])
        self.history["epoch_time"].append(history["time"])
    
    
    def predict(self, inputs):
        """Predict using the multi-layer perceptron classifier.

        Parameters
        ----------
        inputs : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        a : ndarray, shape (n_samples,)
            The predicted classes.
        """
        a = inputs.copy()
        for layer in range(0, self.n_layers):
            a = self.__forward(layer, a)
        return a

    def set_optimizer(self, optimizer):
        """Set the optimizer of the 'Mlp' used in the learning phase.

        Parameters
        ----------
        optimizer : 'isanet.optimizer.SGD' object
        """
        self.__optimizer = optimizer

    def add(self, n_units, input = None, activation = "sigmoid", kernel_initializer = 0.6, kernel_regularizer = 0):
        """Adds a hidden layer to the Multi-layer Perceptron.

        Parameters
        ----------
        n_units : integer
            Number of neurons in the hidden layer.

        input : integer , optional
            Number of inputs. 

        activation : string, default="sigmoid"
            Activation function of the hidden layer.

            - 'identity', no-op activation, useful to implement linear bottleneck, returns f(x) = x
            - 'sigmoid', the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
            - 'tanh', the hyperbolic tan function, returns f(x) = tanh(x).
            - 'relu', the rectified linear unit function, returns f(x) = max(0, x)

        kernel_initializer : float, default=0.6
            Defines the standard deviation value for the uniform distributed 
            distribution used to generate the weights of that layer.

        kernel_regularizer : float, default=0
            Tikhonov regularization term, L2 penalty parameter.

        Raises
        ------
        Warning: this is your first hidden layer
            If you are not inserting the first hidden layer but you are still specifying the number of inputs.

        Warning: First hidden layer already added
            If you are inserting the first hidden layer without specifying the number of inputs.
        """

        dim_input_layer = 0
        actf = None

        if not self.__is_input_layer_set and input is None:
            raise Exception('Warning: this is your first hidden layer, specify the number of inputs')
        else:
            if not self.__is_input_layer_set:
                self.__is_input_layer_set = True
                dim_input_layer = input + 1 
                actf = self.__get_activation_f(activation)
            elif input is None:
                dim_input_layer = self.weights[-1].shape[1] + 1 
                actf = self.__get_activation_f(activation)
            else:
                raise Exception('Warning: First hidden layer already added, avoid input parameter from now')
        
        self.weights.append(random_weights_layer(size = (dim_input_layer, n_units), std = kernel_initializer))
        self.activations.append(actf)
        self.kernel_regularizer.append(kernel_regularizer)
        self.n_layers += 1

    def __get_activation_f(self, act):
        """Returns the activation function."""

        if act == "sigmoid":
            return activation.Sigmoid()
        if act == "tanh":
            return activation.Tanh()
        if act == "linear":
            return activation.Identity()
        if act == "relu":
            return activation.Relu()
        else:
            raise Exception('Warning: activation function unknown')
