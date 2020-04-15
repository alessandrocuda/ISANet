""" Optimizer Module.
"""
import numpy as np
import time
import copy
from . import metrics


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

    def update_weights(self, weights, alpha, d):
        for i in range(0, len(weights)):
            weights[i] += alpha*d[i]
        return weights

    def forward(self, model, weights, X):
        a = X.copy()
        for layer in range(0, model.n_layers):
            z = np.dot(np.insert(a, 0, 1, 1), weights[layer])
            a = model.activations[layer].f(z)
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

    def norm(self, g):
        return np.sqrt(np.sum([np.sum(np.square(g[i])) for i in range(0, len(g))]))

    def scalar_product(self, g, d):
        return np.sum([np.sum(np.multiply(-g[i], d[i])) for i in range(0, len(g))])

    def armijo_wolfe_ls(self, model, X, Y, g, d, c_1, c_2, max_iter = 10):
        alpha_prev, alpha_i, alpha_max = 0., 0.1, 1

        error_zero = metrics.mse(Y, model.predict(X))
        d_error_zero = self.scalar_product(g, d)
        error_prev = 0
        iter = 0

        while iter < max_iter:
            # # Evaluate phi(alpha)
            if alpha_i == 0 or (alpha_max is not None and alpha_prev == alpha_max):
                print("errore")
            
            weight_i = self.update_weights(copy.deepcopy(model.weights), alpha_i, d)
            error_i = metrics.mse(Y, self.forward(model, weight_i, X))

            if error_i > error_zero + c_1*alpha_i*d_error_zero or (error_i >= error_prev and iter > 0):
                print("faccio zoom: {0}, {1}".format(alpha_prev, alpha_i))
                if alpha_prev == alpha_i:
                     raise Exception("sono uguali")
                return self.zoom(alpha_prev, alpha_i, error_zero, d_error_zero, c_1, c_2, model, X, Y, d)

            g_alpha_i = self.backpropagation(model, weight_i, X, Y)
            d_error_i = self.scalar_product(g_alpha_i, d)

            if np.abs(d_error_i) <= -c_2*d_error_zero:
                return alpha_i

            if d_error_i >= 0:
                print("faccio zoom")
                return self.zoom(alpha_i, alpha_prev, error_zero, d_error_zero, c_1, c_2, model, X, Y, d)
            
            alpha_prev = alpha_i
            error_prev = error_i
            alpha_i = min(alpha_i*2, alpha_max)

            iter += 1
        print("iteration line search: {}".format(iter))
        return alpha_i

    def zoom(self, alpha_lo, alpha_hi, error_zero, d_error_zero, c_1, c_2, model, X, Y, d, max_iter=10):
        delta2 = 0.1  # quadratic interpolant check

        weights_lo = self.update_weights(copy.deepcopy(model.weights), alpha_lo, d)
        weights_hi = self.update_weights(copy.deepcopy(model.weights), alpha_hi, d)
        
        error_lo = metrics.mse(Y, self.forward(model, weights_lo, X))
        error_hi = metrics.mse(Y, self.forward(model, weights_hi, X))

        g_lo = self.backpropagation(model, weights_lo, X, Y)
        d_error_lo = self.scalar_product(g_lo, d)

        i = 0
        alpha_j = 0
        
        while i < max_iter:
            # quadratic interpolation
            dalpha = alpha_hi - alpha_lo
            if dalpha < 0:
                a, b = alpha_hi, alpha_lo
            else:
                a, b = alpha_lo, alpha_hi

            qchk = delta2 * dalpha

            alpha_j = self._quadmin(alpha_lo, error_lo, d_error_lo, alpha_hi, error_hi)
            if (alpha_j is None) or (alpha_j > b-qchk) or (alpha_j < a+qchk):
                alpha_j = alpha_lo + 0.5*dalpha
            
            # safeguarded
            #a = max( [ am + ( as - am ) * sfgrd, min( [ as - ( as - am ) * sfgrd,  a ] ) ])

            # alpha_j = np.max( [ alpha_lo + (alpha_hi - alpha_lo)*self.sfgrd, 
            #                 np.min([ alpha_hi - (alpha_hi - alpha_lo)*self.sfgrd, alpha_j])])
                                            
            weights_j = self.update_weights(copy.deepcopy(model.weights), alpha_j, d)
            error_j = metrics.mse(Y, self.forward(model, weights_j, X))

            if error_j > error_zero + c_1*alpha_j*d_error_zero or error_j >= error_lo:
                alpha_hi = alpha_j
                error_hi = error_j
            else:
                g_alpha_j = self.backpropagation(model, weights_j, X, Y)
                d_error_j = self.scalar_product(g_alpha_j, d)
                if np.abs(d_error_j) <= -c_2*d_error_zero:
                    return alpha_j
                if d_error_j*(alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo
                    error_hi = error_lo

                alpha_lo = alpha_j
                error_lo = error_j
                d_error_lo = d_error_j                   
            i += 1

        return alpha_j

    def _quadmin(self, a, fa, fpa, b, fb):
        D = fa
        C = fpa
        db = b - a * 1.0
        B = (fb - D - C * db) / (db * db)
        xmin = a - C / (2.0 * B)
        return xmin


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

    def __init__(self, beta_method = "pr", d_method='standard', c_1=1e-4, c_2=.1, sfgrd = 0.01, tol = None, n_iter_no_change = None):
        super().__init__()
        self.tol = tol
        self.c_1 = c_1
        self.c_2 = c_2
        self.sfgrd = sfgrd
        self.past_g = 0
        self.past_d = 0
        self.past_norm_g = 0
        self.model_backup = 0
        self.beta_function = self.get_beta_function(beta_method)

    def optimize(self, model, epochs, X_train, Y_train, validation_data = None, batch_size = None, es = None, verbose = 0):
        super().optimize(model, epochs, X_train, Y_train, validation_data=validation_data, batch_size=batch_size, es=es, verbose=verbose)

    def step(self, model, X, Y):
        d = {}
        g = self.backpropagation(model, model.weights, X, Y)
    
        if ~model.is_fitted and self.epoch == 0:
            d = g
        else:
            # calcolo del beta
            beta = self.beta_function(g, self.past_g, self.past_norm_g)
            print("Beta: {}".format(beta))
            if beta != 0:
                for i in range(0, len(model.weights)):
                    d[i] = g[i] + beta * self.past_d[i]
            else:
                d = g
        
        self.past_g = g
        self.past_d = d
        self.past_norm_g = self.norm(self.past_g)

        alpha = self.armijo_wolfe_ls(model, X, Y, g, d, self.c_1, self.c_2)

        print("Alpha: {}".format(alpha))
        for i in range(0, len(model.weights)):
            model.weights[i] += alpha*d[i]


    def get_beta_function(self, beta_method):
        if beta_method == "pr":
            return self.beta_pr
    
    def beta_pr(self, g, past_g, past_norm_g):
        diff = {}
        for i in range(0,len(past_g)):
            diff[i] =  -g[i] + past_g[i]
        beta = self.scalar_product(g, diff)/np.square(past_norm_g)
        return max(0, beta)