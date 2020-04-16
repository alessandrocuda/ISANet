""" Optimizer Module.
"""
import numpy as np
import time
import copy
import isanet.metrics as metrics
from isanet.optimizer.linesearch import armijo_wolfe_ls
from isanet.optimizer.utils import make_vector, restore_w_to_model

class EarlyStopping():
    """Stop training when a the MSE (Mean squared error) on validation
    (generalization error) is increasing and exceeds a certain threshold
    for a finite number of epochs.

    Notes
    -----
    The class define the generalization loss at epoch t to be the relative
    increase of the validation error over the minimum-so-far (in percent)::
    
            GL(t) = 100*(mse_val(t)/min_mse_val(t)-1)

    Then it will stop the optimization when the generatilization loss exceeds
    a certain thresold for a finite number of epochs::

            G(t) > eps

    Parameters
    ----------
    eps : float       
        Threshold that is used to decide whether to stop the 
        fitting of the model (it stops if this is true and after
        a number of epoch > 'patience').

    patience : integer 
        Number of epochs that mse should be worse after which training
        will be stopped.

    verbose : boolean, default=False
        Whether to print progress messages to stdout.
    """
    def __init__(self, eps, patience, verbose = False):
        self.eps = eps
        self.patience = patience
        self.verbose = verbose

        self.__min_val = None
        self.__weights_backup = None
        self.__es_count = None
    
    def __repr__(self):
        return ("eps: {0}, patience: {1}").format(self.eps, self.patience)

    def __str__(self):
        return ("eps: {0}, patience: {1}").format(self.eps, self.patience)

    def check_early_stop(self, model, epoch, history):
        """Check if Early Stopping criteria has occurred.

        Parameters
        ----------
        model : isanet.model.MLP

        epoch : integer
            Current number of epoch (epoch == 0 is the first epoch).

        history : dict
            Contains, for the current epoch, the values of mse, mee and accuracy 
            for training and validation and the time taken to compute that epoch.
            This parameter is used to check the value of current MSE (Mean squared 
            error) on validation (generalization error). 
        
        Returns
        -------
        boolean
            True if the Early stopping has occurred, else False
        """
        if epoch == 0 or history["mse_val"] <= self.__min_val:
            self.__min_val = history["mse_val"]
            self.__weights_backup = copy.deepcopy(model.weights)
            self.__es_count = 0
            return False
        else:
            self.__es_count += 1
            gl = 100*(history["mse_val"]/self.__min_val  - 1)
            if self.verbose: 
                print(" - ES Info: going in overfitting, gl_value: {} count: {}".format(gl, self.__es_count))
            if gl > self.eps and self.__es_count >=  self.patience:
                if self.verbose:
                    print(" - ES Info: Stop For Overfitting")
                    print(" - best mse on validation: {}".format(self.__min_val))
                model.weights = self.__weights_backup
                return True      

    def get_weights_backup(self):
        """Returns the weights's backup if a minimum of the generalization 
        has been reached after overfitting.
        
        Returns
        -------
        list of arrays-like
            Weights's backup.
        """
        return self.__weights_backup

    def get_min_val(self):
        """Returns the min mse on validation if a minimum of the generalization 
        has been reached after overfitting.
        
        Returns
        -------
        float
           Min of the mse on validation.
        """
        return self.__min_val
        

class Optimizer(object):

    def __init__(self):
        self.epoch = 0
        self.model = None
        self.X_part = None
        self.Y_part = None
        self.d = None

    def optimize(self, model, epochs, X_train, Y_train, validation_data = None, batch_size = None, es = None, verbose = 0):
        """
        Parameters
        ----------
        model : isanet.model.MLP
            Specify the Multilayer Perceptron object to optimize

        epochs : integer
            Maximum number of epochs.

        X_train : array-like of shape (n_samples, n_features)
            The input data.

        Y_train : array-like of shape (n_samples, n_output)
            The target values.

        validation_data : list of arrays-like, [X_val, Y_val], optional
            Validation set.

        batch_size : integer, optional
            Size of minibatches for the optimizer.
            When set to "none", the optimizer will performe a full batch.

        es : isanet.callbacks.EarlyStopping, optional
            When set to None it will only use the ``epochs`` to finish training.
            Otherwise, an EarlyStopping type object has been passed and will stop 
            training if the model goes overfitting after a number of consecutive iterations.
            See docs in optimizier module for the EarlyStopping Class.

        verbose : integer, default=0
            Controls the verbosity: the higher, the more messages.

        Returns
        -------
        integer

        """
        self.tot_n_patterns = X_train.shape[0]
        self.__batch_size = batch_size

        if batch_size is None:
            self.__batch_size = self.tot_n_patterns
        is_validation_set = True
        if validation_data is None:
            is_validation_set = False


        #for epoch in range(epochs):
        while self.epoch < epochs:
            start_time = time.time()
            batchs = self.get_batch(X_train, Y_train, self.__batch_size)

            for current_batch in batchs:
                X = batchs[current_batch]["batch_x_train"]
                Y = batchs[current_batch]["batch_y_train"]
                
                # Performs a single optimization step (parameter update).
                self.step(model, X, Y)

            end_time = (time.time() - start_time)
            
            # Update history with MSE, MEE, Accuracy, Time after each Epoach 
            history = self.get_epoch_history(model, X_train, Y_train, validation_data, end_time)
            model.append_history(history, is_validation_set)

            if verbose and (self.epoch + 1) % verbose == 0:
                print("Epoch: {} - time: {:4.4f} - loss_train: {} - loss_val: {}".format(self.epoch + 1, history["time"], history["mse_train"], history["mse_val"]))

            # Check Early Stopping 1: avoid overfitting
            if  is_validation_set and es and es.check_early_stop(model, self.epoch, history):
                return 0
            
            # Check Early Stopping 2: no change on training error
            if self.tol and self.__no_change_in_training(model, epoch, is_validation_set, es, verbose):
                return 0

            self.epoch+=1

    def phi(self, w):
        weights = restore_w_to_model(model, w)
        y_pred = self.forward(weights, self.X_part)
        return metrics.mse(self.Y_part, y_pred)

    def derphi(self, w):
        weights = restore_w_to_model(model, w)
        g = make_vector(self.backpropagation(model, weights, self.X_part, self.Y_part))
        derphi_a1 = np.asscalar(np.dot(g.T, self.d))

    def forward(self, weights, X):
        a = X.copy()
        for layer in range(self.model.n_layers):
            z = np.dot(np.insert(a, 0, 1, 1), weights[layer])
            a = self.model.activations[layer].f(z)
        return a

    def backpropagation(self, model, weights, X, Y):
        """
        Parameters
        ----------
        model : isanet.model.MLP
            Specify the Multilayer Perceptron object to optimize

        X : array-like of shape (n_samples, n_features)
            The input data.

        Y : array-like of shape (n_samples, n_output)
            The target values.

        Returns
        -------
        dict
            contains the gradients for each layer to be used in the delta rule. 
            Each key in the dictionary represents the ith layer. (from the first
            hidden layer to the output layer).::

                E.g. 0 -> first hidden layer, ..., n+1 -> output layer
                where n is the number of hidden layer in the net.
        """
        A = [0]*(model.n_layers+1)   # outputs after the activation functions of all layers ( input to output)
        Z = [0]*(model.n_layers)                       # outputs before the activation functions of all layers ( hidden layers to output)
        g = [0]*model.n_layers                 # dictionary of gradient for each layer (hidden to output)

        #####################
        # Feed Forward Phase
        ###################
        A[0] = np.insert(X, 0, 1, 1)
        for layer in range(0, model.n_layers):
            z = np.dot(A[layer], weights[layer])
            Z[layer] = z
            output = model.activations[layer].f(z)
            A[layer+1] = np.insert(output, 0, 1, 1)
        Y_pred = A[-1][:,1:]
        Y_z_pred = Z[-1]

        #######################
        # Output layer K 
        #####################
        loss_delta = Y - Y_pred  
        derivates = model.activations[-1].derivative(Y_z_pred)
        d_node_k = -loss_delta*derivates
        g[model.n_layers-1] = np.dot(A[-2].T, d_node_k)

        ########################
        # Hidden layers H
        ######################
        d_to_prop = d_node_k
        for l in range(2, model.n_layers+1):
            d = np.dot(d_to_prop, weights[-l+1].T)[:,1:]
            derivates_h = model.activations[-l].derivative(Z[-l])
            d_node_h = d*derivates_h
            g[model.n_layers-l] = np.dot(A[-l-1].T, d_node_h)
            d_to_prop = d_node_h
        
        return g 

    def get_batch(self, X_train, Y_train, batch_size):
        """ 
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            The input data.

        Y_train : array-like of shape (n_samples, n_output)
            The target values.

        batch_size : integer
            Size of minibatches for the optimizer.

        Returns
        -------
        dict of dict
            Each key of the dictionary is a integer value from 0 to
            number_of_batch -1 and define a batch. Each element is a
            dictionary and has two key: 'batch_x_train' and 'batch_y_train'
            and refer to the portion of data and target respectively used 
            for the training.
        """
        if batch_size >= X_train.shape[0]:
            return {0: {"batch_x_train": X_train, "batch_y_train": Y_train}}
        else:
            trainXY = np.append(X_train, Y_train, axis=1)
            np.random.shuffle(trainXY)
            trainX = trainXY[:,:X_train.shape[1]]
            trainY = trainXY[:,X_train.shape[1]:] 
            batchs = {}
            p = trainX.shape[0]
            mb = int(p/batch_size)
            if (p % batch_size) == 0:
                batchsX = np.split(trainX, mb)
                batchsY = np.split(trainY, mb)
                for i in range(0, mb):
                    batchs[i] = {"batch_x_train": batchsX[i], "batch_y_train": batchsY[i]}
            else:
                vsplit_idx = mb*batch_size
                xbl = trainX[:vsplit_idx,:]
                xbr = trainX[vsplit_idx:,:]
                ybl = trainY[:vsplit_idx,:]
                ybr = trainY[vsplit_idx:,:]
                batchsX = np.split(xbl, mb)
                batchsY = np.split(ybl, mb)
                for i in range(0, mb):
                    batchs[i] = {"batch_x_train": batchsX[i], "batch_y_train": batchsY[i]}
                batchs[mb] = {"batch_x_train": xbr, "batch_y_train": ybr}
            return batchs

    def get_epoch_history(self, model, X_train, Y_train, validation_data, time):
        """Given the model, training data, validation data and time returns a dictionary
        that contains: 

                {"mse_train": mse_train,
                 "mee_train": mee_train, 
                 "acc_train": acc_train,
                 "mse_val": mse_val,
                 "mee_val": mee_val,
                 "acc_val": acc_val,
                 "time": time
                }
        """
        mse_val = mee_val = acc_val = 0
        if validation_data is not None:
            out = model.predict(validation_data[0])
            mse_val =  metrics.mse(validation_data[1], out)
            mee_val =  metrics.mee(validation_data[1], out)
            acc_val =  metrics.accuracy_binary(validation_data[1], out)
        out = model.predict(X_train)
        mse_train =  metrics.mse(Y_train, out)
        mee_train =  metrics.mee(Y_train, out)
        acc_train =  metrics.accuracy_binary(Y_train, out)
        
        return {"mse_train": mse_train,
                "mee_train": mee_train, 
                "acc_train": acc_train,
                "mse_val": mse_val,
                "mee_val": mee_val,
                "acc_val": acc_val,
                "time": time
                }

    def __no_change_in_training(self, model, epoch, is_validation_set, es, verbose):
        """Check if there are no improvements in optimizations for a given 
        epsilon 'self.eps' and for a given number of consecutive epochs 
        'self.patience'.
        
        Parameters
        ----------
        model : isanet.model.MLP
            Specify the Multilayer Perceptron object to optimize

        epoch : integer
            Current number of epoch (epoch == 0 is the first epoch).

        es : None or isanet.optimizer.EalryStopping
            Specify if a EalryStopping for the overfitting has been set.

        verbose : boolean
            Whether to print progress messages to stdout.

        Returns
        -------
        boolean
            True if there hasn't been any improvement, else False
        """
        if epoch > 1 and model.history["loss_mse"][-2] > model.history["loss_mse"][-1]:
            delta_improv_train = model.history["loss_mse"][-2] - model.history["loss_mse"][-1]
            if delta_improv_train < self.tol:
                self.__no_change_count += 1
                if verbose: 
                    print(" - Improv Info - delta_improv_train: {}, epoch count {}".format(delta_improv_train, self.__no_change_count))
                if self.__no_change_count >= self.n_iter_no_change:
                    if is_validation_set and es and model.history["val_loss_mse"][-1] > es.get_min_val(): # controllare che sia stato passato il validation
                        model.weights = es.get_weights_backup()
                        print(" - Model gone overfitting: restore weights")
                    print(" - Stop: no improvement in training error")
                    print(" - Delta error for training: {} after {} epoch of no change".format(delta_improv_train, self.__no_change_count))
                    return True
                return False
            else:
                self.__no_change_count = 0
                return False


class SGD(Optimizer):
    """
    Parameters
    ----------
    lr : float, default=0.1
        Learning rate schedule for weight updates (delta rule).

    momentum : float, default=0
        Momentum for gradient descent update.

    nesterov : boolean, default=False
        Whether to use Nesterov’s momentum.

    sigma : float, default=None
        Parameter of the Super Accelerated Nesterov's momentum.
        If 'nesterov' is True and 'sigma' equals to 'momentum', then we have the
        simple Nesterov momentum. Instead, if 'sigma' is different from 
        'momentum', we have the super accelerated Nesterov.

    tol : float, default=None
        Tolerance for the optimization. When the loss on training is
        not improving by at least tol for 'n_iter_no_change' consecutive 
        iterations convergence is considered to be reached and training stops.

    n_iter_no_change : integer, default=None
        Maximum number of epochs with no improvements > tol. 
    """
    def __init__(self, lr=0.1, momentum=0, nesterov=False, sigma = None, tol = None, n_iter_no_change = None):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        if n_iter_no_change is None:
            self.n_iter_no_change = 1

        self.sigma = sigma
        if sigma is None and nesterov is True:
            self.sigma = momentum

        self.__batch_size = None
        self.__es_count = 0
        self.__no_change_count = 0

        
    def optimize(self, model, epochs, X_train, Y_train, validation_data=None, batch_size=None, es=None, verbose=0):
        if ~model.is_fitted:
            self.delta_w = [0]*len(model.weights)
        super().optimize(model, epochs, X_train, Y_train, validation_data=validation_data, batch_size=batch_size, es=es, verbose=verbose)


    def step(self, model, X, Y):

        current_batch_size = X.shape[0]
        lr = self.lr/current_batch_size

        weights = model.weights
        if self.nesterov == True:
            weights = copy.deepcopy(model.weights)
            for i in range(0, len(model.weights)): 
                weights[i] += self.sigma*self.delta_w[i] 

        g  = self.backpropagation(model, weights, X, Y)

        #      Delta Rule Update
        #  w_i = w_i + eta*nabla_W_i
        for i in range(0, len(model.weights)):
            self.delta_w[i] = -lr*g[i] + self.momentum*self.delta_w[i]   
            old_w = copy.deepcopy(model.weights[i])
            old_w[0,:] = 0
            regularizer = model.kernel_regularizer[i]*current_batch_size/self.tot_n_patterns
            model.weights[i] += (self.delta_w[i] - regularizer*old_w)


class NCG(Optimizer):

    def __init__(self, beta_method = "pr", d_method='standard', c1=1e-4, c2=.9, sfgrd = 0.01, tol = None, n_iter_no_change = None):
        super().__init__()
        self.tol = tol
        self.c1 = c1
        self.c2 = c2
        self.sfgrd = sfgrd
        self.old_phi0 = None
        self.past_g = 0
        self.past_d = 0
        self.past_ng = 0
        self.w = 0
        self.fbeta = self.get_beta_function(beta_method)

    def optimize(self, model, epochs, X_train, Y_train, validation_data = None, batch_size = None, es = None, verbose = 0):
        self.model = model
        super().optimize(model, epochs, X_train, Y_train, validation_data=validation_data, batch_size=batch_size, es=es, verbose=verbose)

    def step(self, model, X, Y):
        print()
        w = make_vector(model.weights)
        g = make_vector(self.backpropagation(model, model.weights, X, Y))
    
        if ~model.is_fitted and self.epoch == 0:
            d = -g
            phi0 = metrics.mse(Y, model.predict(X))
        else:
            # calcolo del beta
            beta = self.fbeta(g, self.past_g, self.past_ng)
            print("Beta: {}".format(beta), end=" -> compute alpha: ")
            if beta != 0:
                d = - g + beta*self.past_d 
            else:
                d = - g
            phi0 = model.history["loss_mse"][-1]

        self.past_ng = np.linalg.norm(g)        
        self.past_g = g
        self.past_d = d

        alpha = armijo_wolfe_ls(self, model, w, X, Y, phi0, self.old_phi0, g, d, self.c1, self.c2)

        print("Alpha: {}".format(alpha))
        self.old_phi0 = phi0
        w += alpha*d
        model.weights = restore_w_to_model(model, w)



    def get_beta_function(self, beta_method):
        if beta_method == "pr":
            return self.beta_pr
    
    def beta_pr(self, g, past_g, past_norm_g):
        A = g.T
        B = g-past_g
        C = np.square(past_norm_g)
        beta = np.asscalar(np.dot(A,B)/C)
        return max(0, beta)