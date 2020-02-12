from itertools import product, chain
import numpy as np
import numbers
import pickle
import time
import datetime
import copy


from isanet.model import Mlp
from isanet.optimizer import SGD
from isanet.optimizer import EarlyStopping
from isanet.metrics import mse, mee, accuracy_binary

class Kfold():
    """K-Folds cross-validator

    Provides train/validation indices to split data into train/validation sets. Split
    dataset into k consecutive folds (without shuffling by default).
    Each fold is then used once as validation while the k - 1 remaining folds form 
    the training set.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.

    shuffle : boolean, optional
        Whether to shuffle the data before splitting into folds.

    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Only used when ``shuffle`` is True. This should be left
        to None if ``shuffle`` is False.

    Notes
    -----
    The first ``n_splits - 1`` folds have size
    ``n_samples // n_splits``, the other fold has size
    ``n_samples // n_splits + n_samples % n_splits``, where ``n_samples`` is the number of samples.
    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting ``random_state``
    to an integer.
    """
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self):
        """Returns the number of folds of the KFold object."""
        return self.n_splits

    def split(self, X, y=None): 
        """Generate indices to split data into training and validation set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,), optional
            The target variable for supervised learning problems.

        Raises
        ------
        ValueError
            If the number of folds is greater than the number of samples.

        Returns
        -------
        dict
            a dict of two lists, one with the indices of the training and
            the other with the indices of the validation set.
        """
        n_samples = _num_samples(X)
        if self.n_splits > n_samples:
            raise ValueError(
                ("Cannot have number of splits n_splits={0} greater"
                 " than the number of samples: n_samples={1}.")
                .format(self.n_splits, n_samples))
    
        index = []

        elem = np.arange(0, n_samples)
        
        if self.shuffle:
            if self.random_state:
                np.random.seed(self.random_state)
            np.random.shuffle(elem)

        num_elem = int(np.floor(n_samples/self.n_splits))

        for fold in range(0, self.n_splits-1):
            start = fold*num_elem
            end = num_elem*(fold+1)
            index.append(np.array(elem[start:end]))
        start = (self.n_splits - 1)*num_elem
        index.append(np.array(elem[start:]))

        val_index = []
        train_index = []
        for val in range(0, self.n_splits):
            list_temp = []
            for i in range(0, self.n_splits):
                if val == i:
                    val_index.append(index[i].tolist())
                else:
                    list_temp.append(index[i].tolist())
            train_index.append(list(chain.from_iterable(list_temp)))
        
        return {"train": train_index, "val": val_index}


class GridSearchCV():
    """Exhaustive search over specified parameter values for an estimator.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated grid-search over a parameter grid.

    Parameters
    ----------
    estimator : estimator object.
        This represents the type of task to be solved using a neural network, 
        this can be a classifier or a regressor, implemented using the 
        MLPClassifier and MLPRegressor classes provided by the 
        isanet.neural_network module.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

        NOTE
        
        The dictionary keys must have the same name as the parameters of the 
        estimator passed to the class. 

    cv : KFold object or dict, default=Kfold(n_splits=5, shuffle=True)
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        
        - None, to use the default 5-fold cross validation,
        - A dict of two lists, one with the indices of the training and
            the other with the indices of the validation set.

    verbose : integer, default=0
        Controls the verbosity: the higher, the more messages.

        - 0: no messages
        - 1: grid iterations
        - 2: grid iterations + folds results

    Attributes
    ----------
    grid_dim : integer
        The number of hyperparameters.

    folds_dim : integer
        The number of folds of the KFold object.

    tot_train : integer
        The total number of training needed.

    Notes
    -----
    The parameters selected are those that maximize the score of the left out
    data, unless an explicit score is passed in which case it is used instead.
    If `n_jobs` was set to a value higher than one, the data is copied for each
    point in the grid (and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.  
    """
    def __init__(self, estimator, param_grid, cv = Kfold(n_splits=5, shuffle=True), verbose=0):
        self.estimator = estimator
        self.param_grid = self._init_grid(param_grid)
        self.cv = cv 
        self.verbose = verbose

        self.grid_dim = len(self.param_grid)
        
        if isinstance(self.cv, dict):
            self.folds_dim = len(self.cv["train"])
        else:
            self.folds_dim = self.cv.get_n_splits()
        
        self.tot_train = self.grid_dim*self.folds_dim
        

    def get_grid_dim(self):
        """Returns the number of hyperparameters of the GridSearchCV object."""
        return self.grid_dim

    def _init_grid(self, param_grid):
        """Take the hyperparameters and return a list of dict containing
         all the possible hyperparameters combinations.
         
        Parameters
        ----------
        param_grid : dict of lists
            Dictionary with parameters names (string) as keys and lists of
            parameter settings to try as values, or a list of such
            dictionaries, in which case the grids spanned by each dictionary
            in the list are explored. This enables searching over any sequence
            of parameter settings.

        Returns
        -------
        list
            a list of dict. Each dict contains a hyperparameters combination.

        Examples
        --------
        >>> param_grid = {"n_units": [[38], [40,50]], "learning_rate": [0.014, 0.017]}
        >>> param_list = list(product_dict(**param_grid))
        >>> param_list
        [{"n_units": [38], "learning_rate": 0.014}, {"n_units": [38], "learning_rate": 0.017}, 
        {"n_units": [40,50], "learning_rate": 0.014}, {"n_units": [40,50], "learning_rate": 0.017}]
        """
        param_list = list(product_dict(**param_grid))
        return param_list

    
    def fit(self, X, Y):
        """ Run fit with all sets of parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        
        Y : array-like, shape = [n_samples] or [n_samples, n_output]
            Target relative to X for classification or regression;
            None for unsupervised learning.

        Returns
        -------
        dict
            A dict with keys as column headers and values as columns, that can be imported into a pandas DataFrame.

            For instance the below given table::
            
                    +-------------------------+---------------+----------------+---------------+---+------------+
                    | hyper_param             | fold_results  | mean_train_mse | std_train_mse |...| time_train |
                    +=========================+===============+================+===============+===+============+
                    | {'n_layer_units': [38], | {'train_mse': |                |               |   |            |   
                    | 'learning_rate': 0.014, | [2.019141..., |   1.966004     |   0.061030    |...|  1.786321  |   
                    | 'max..                  | 1.95675058... |                |               |   |            |
                    +-------------------------+---------------+----------------+---------------+---+------------+
                    | {'n_layer_units': [38], | {'train_mse': |                |               |   |            |
                    | 'learning_rate': 0.017, | [2.019141..., |   1.853260     |   0.047401    |...|  1.532336  |
                    | 'max..                  | 1.95675058... |                |               |   |            |
                    +-------------------------+---------------+----------------+---------------+---+------------+

            will be represented as follow::

                    {
                     "hyper_param":          [{'n_layer_units': [38], 'learning_rate': 0.014...}, ...],
                     "fold_results":         [{'train_mse': [1.9814435389846516, 1.865889091...}, ...],
                     "mean_train_mse":       [ ... ],
                     "std_train_mse":        [ ... ],
                     "mean_train_mee":       [ ... ],
                     "std_train_mee":        [ ... ],
                     "mean_train_acc":       [ ... ],
                     "std_train_acc":        [ ... ],
                     "mean_val_mse":         [ ... ],
                     "std_val_mse":          [ ... ],
                     "mean_val_mee":         [ ... ],
                     "std_val_mee":          [ ... ],
                     "mean_val_acc":         [ ... ],
                     "std_val_acc":          [ ... ],
                     "time_train":           [ ... ]
                    }

            NOTE
            
            The key 'hyper_param' is used to store a list of parameter settings dicts for all 
            the parameter combination candidates. Each dicts will have the same keys of 'param_grid'
            passed as parameter to the GridSearchCV object.

            The 'fold_results' key is used to store a list of dictionaries that represent the result 
            of cross validation (E.g. Kfolds) and will have a length equal to the number of folds 
            specified in the 'cv' parameter. 
            
            E.g the folds result with K = 4::

                    {
                     "train_mse":    [1.9814435389846516, ..., 2.013302806990128],
                     "train_mee":    [ ... ],
                     "train_acc":    [ ... ],
                     "val_mse":      [ ... ],
                     "val_mee":      [ ... ],
                     "val_acc":      [ ... ],
                     "init_weigth":  [ ... ],
                     "final_weigth": [ ... ],
                     "fold_time":    [ ... ],
                    }      

            where each list has lenght 4





        """
        grid_results = _GridResult()
        history_time = []
        grid_iter = 0
        print("Total Grid Search Iteration: {0}, Total train: {1}".format(self.grid_dim, self.tot_train))

        if isinstance(self.cv, dict):
            kfold = self.cv
        else:
            kfold = self.cv.split(X,Y)

        for hyper_param in self.param_grid:
            fold_results = _FoldResult()
            folds_iter = 0

            grid_iter += 1
            train_folds_time = time.time()

            if self.verbose > 1:
                print("Start Grid Combination n. {}:".format(grid_iter))
            
            for train_index_fold, val_index_fold in zip(kfold["train"], kfold["val"]):
                folds_iter += 1
                X_train = X[train_index_fold]
                Y_train = Y[train_index_fold]

                X_val = X[val_index_fold]
                Y_val = Y[val_index_fold]

                self.estimator.set_params(X.shape[1], Y.shape[1], **hyper_param)


                init_weigth = copy.deepcopy(self.estimator.get_weights())
                
                fold_time = time.time()

                self.estimator.fit(X_train, Y_train, X_val, Y_val)
                
                fold_time = time.time() - fold_time

                final_weigth = copy.deepcopy(self.estimator.get_weights())
                train_out = self.estimator.predict(X_train)
                val_out = self.estimator.predict(X_val)

                fold_results.add_row(mse(Y_train, train_out),
                                     mee(Y_train, train_out),
                                     accuracy_binary(Y_train, train_out),
                                     mse(Y_val, val_out),
                                     mee(Y_val, val_out),
                                     accuracy_binary(Y_val, val_out),
                                     init_weigth,
                                     final_weigth,
                                     fold_time)
                
                if self.verbose > 1:
                    print("fold: {}/{} - TR MSE: {:0.4f} - VL MSE: {:0.4f} - Time: {}".format(
                                folds_iter,
                                self.folds_dim,
                                fold_results.get_value("train_mse")[-1],                                            
                                fold_results.get_value("val_mse")[-1], 
                                datetime.timedelta(seconds=fold_time)))

            train_folds_time = time.time() - train_folds_time

            grid_results.add_row(hyper_param, fold_results.get_dic_results(), 
                                 np.mean(fold_results.get_value("train_mse")),
                                 np.std(fold_results.get_value("train_mse")),
                                 np.mean(fold_results.get_value("train_mee")),
                                 np.std(fold_results.get_value("train_mee")),  
                                 np.mean(fold_results.get_value("train_acc")),
                                 np.std(fold_results.get_value("train_acc")),
                                 np.mean(fold_results.get_value("val_mse")),
                                 np.std(fold_results.get_value("val_mse")),
                                 np.mean(fold_results.get_value("val_mee")),
                                 np.std(fold_results.get_value("val_mee")),  
                                 np.mean(fold_results.get_value("val_acc")),
                                 np.std(fold_results.get_value("val_acc")),
                                 train_folds_time)
            
            remaining_time = self.__get_remaining_time(history_time, train_folds_time, grid_iter)
            if self.verbose > 0:
                print("Result It: {}/{} - TR MSE: {:0.4f}+/-{:0.4f} - VL MSE: {:0.4f}+/-{:0.4f} - Time: {} | Remaining time: {}".format(
                            grid_iter,
                            self.grid_dim,
                            grid_results.get_value("mean_train_mse")[-1],                                            
                            grid_results.get_value("std_train_mse")[-1],
                            grid_results.get_value("mean_val_mse")[-1], 
                            grid_results.get_value("std_val_mse")[-1], 
                            datetime.timedelta(seconds=train_folds_time), 
                            datetime.timedelta(seconds=remaining_time)))

        return grid_results.get_dic_results()

    def __get_remaining_time(self, history_time, train_folds_time, current_iter):
        """Returns how long it is until the end of the grid search."""
        history_time.append(train_folds_time)
        if len(history_time) == 20:
            history_time = history_time[-10:]
        return (np.mean(history_time)*(self.grid_dim-current_iter))


class _Results():
    """A class used to represent a general result, structuring it as a dict.

    Parameters
    ----------
    struct : dict

    Attributes
    ----------
    struct : dict
        define the wrapped dictionary 
    """
    def __init__(self, struct):
        self.struct = struct

    def append_value(self, key, value):
        """Adds the tuple {"key": value} to the dict.
        
        Parameters
        ----------
        key : string

        value : dict or integer or float
        """
        self.struct[key].append(value)

    def get_value(self, key):
        """Given the key, it returns the associated value.
        
        Parameters
        ----------
        key : string
        """
        return self.struct[key]
    
    def get_dic_results(self):
        """Returns the dict."""
        return self.struct
    
class _GridResult(_Results):
    """Child class of Result. Represents the results of the Grid Search.

    Attributes
    ----------
    grid_results : dict
        Scores of the estimator for each run of the grid search.

        A dict of arrays containing the score/time arrays for each scorer is
        returned. The possible keys for this ``dict`` are:

            ``hyper_param``
                The used hyperparameters for a run of the grid search.
            ``fold_results``
                The results of each fold of the K-Fold. A _FoldResult object.
            ``mean_train_mse``
                The mean of the mse (Mean Squared Error) over the training set,
                for a run of the grid search.
            ``std_train_mse``
                The standard deviation of the mse (Mean Squared Error) 
                over the training set, for a run of the grid search.
            ``mean_train_mee``
                The mean of the mee (Mean Euclidian Error) over the training set,
                for a run of the grid search.
            ``std_train_mee``
                The standard deviation of the mee (Mean Euclidian Error) 
                over the training set, for a run of the grid search.
            ``mean_train_acc``
                The mean of the accuracy over the training set,
                for a run of the grid search.
            ``std_train_acc``
                The standard deviation of the accuracy over the training set, 
                for a run of the grid search.
            ``mean_val_mse``
                The mean of the mse (Mean Squared Error) over the validation set,
                for a run of the grid search.
            ``std_val_mse``
                The standard deviation of the mse (Mean Squared Error) over 
                the validation set, for a run of the grid search.
            ``mean_val_mee``
                The mean of the mee (Mean Euclidian Error) over the validation set,
                for a run of the grid search.
            ``std_val_mee``
                The standard deviation of the mse (Mean Squared Error) over 
                the validation set, for a run of the grid search.
            ``mean_val_acc``
                The mean of the accuracy over the validation set,
                for a run of the grid search.
            ``std_val_acc``
                The standard deviation of the accuracy over the validation set,
                for a run of the grid search.
            ``time_train``
                The time for fitting the estimator on the train
                set for a run of the grid search.
    """
    def __init__(self):

        grid_results = {
            "hyper_param":          [],
            "fold_results":         [],
            "mean_train_mse":       [],
            "std_train_mse":        [],
            "mean_train_mee":       [],
            "std_train_mee":        [],
            "mean_train_acc":       [],
            "std_train_acc":        [],
            "mean_val_mse":         [],
            "std_val_mse":          [],
            "mean_val_mee":         [],
            "std_val_mee":          [],
            "mean_val_acc":         [],
            "std_val_acc":          [],
            "time_train":           []
        }
        super().__init__(grid_results)
    
    def add_row(self, hyper_param, fold_results, 
                mean_train_mse, std_train_mse,
                mean_train_mee, std_train_mee, 
                mean_train_acc, std_train_acc, 
                mean_val_mse, std_val_mse, 
                mean_val_mee, std_val_mee, 
                mean_val_acc, std_val_acc, time_train):
        """Adds the results of a run of the grid search.
        
        Parameters
        ----------
        hyper_param : dict
            The used hyperparameters for a run of the grid search.
            
        fold_results : dict
            The results of each fold of the K-Fold. A _FoldResult object.
        
        mean_train_mse : float
            The mean of the mse (Mean Squared Error) over the training set,
            for a run of the grid search.

        std_train_mse : float
            The standard deviation of the mse (Mean Squared Error) 
            over the training set, for a run of the grid search.
        mean_train_mee : float
            The mean of the mee (Mean Euclidian Error) over the training set,
            for a run of the grid search.

        std_train_mee : float
            The standard deviation of the mee (Mean Euclidian Error) 
            over the training set, for a run of the grid search.

        mean_train_acc : float
            The mean of the accuracy over the training set,
            for a run of the grid search.

        std_train_acc : float
            The standard deviation of the accuracy over the training set, 
            for a run of the grid search.

        mean_val_mse : float
            The mean of the mse (Mean Squared Error) over the validation set,
            for a run of the grid search.

        std_val_mse : float
            The standard deviation of the mse (Mean Squared Error) over 
            the validation set, for a run of the grid search.

        mean_val_mee : float
            The mean of the mee (Mean Euclidian Error) over the validation set,
            for a run of the grid search.

        std_val_mee : float
            The standard deviation of the mse (Mean Squared Error) over 
            the validation set, for a run of the grid search.

        mean_val_acc : float
            The mean of the accuracy over the validation set,
            for a run of the grid search.

        std_val_acc : float
            The standard deviation of the accuracy over the validation set,
            for a run of the grid search.

        time_train : float
            The time for fitting the estimator on the train
            set for a run of the grid search.        
        """

        self.append_value("hyper_param", hyper_param)
        self.append_value("fold_results", fold_results)
        self.append_value("mean_train_mse", mean_train_mse)
        self.append_value("std_train_mse", std_train_mse)
        self.append_value("mean_train_mee", mean_train_mee)
        self.append_value("std_train_mee", std_train_mee)
        self.append_value("mean_train_acc", mean_train_acc)
        self.append_value("std_train_acc", std_train_acc)
        self.append_value("mean_val_mse", mean_val_mse)
        self.append_value("std_val_mse", std_val_mse)
        self.append_value("mean_val_mee", mean_val_mee)
        self.append_value("std_val_mee", std_val_mee)
        self.append_value("mean_val_acc", mean_val_acc)
        self.append_value("std_val_acc", std_val_acc)
        self.append_value("time_train", time_train)

class _FoldResult(_Results):
    """Child class of _Result. Represents the results of the K-Fold.
    
    Attributes
    ----------
    fold_results : dict
        Scores of the estimator for each run of the K-Fold.

        A dict of arrays containing the score/time arrays for each scorer is
        returned. The possible keys for this ``dict`` are:

            ``train_mse``
                The mse (Mean Squared Error) over the training set, for each fold.
            ``train_mee``
                The mee (Mean Euclidian Error) over the training set, for each fold.
            ``train_acc``
                The accuracy over the training set, for each fold.
            ``val_mse``
                The mse (Mean Squared Error) over the validation set, for each fold.
            ``val_mee``
                The mee (Mean Euclidian Error) over the validation set, for each fold.
            ``val_acc``
                The accuracy over the validation set, for each fold.
            ``init_weigth``
                Initial weights of the neural network, for each fold.
            ``final_weigth``
                Final weights of the neural network, for each fold.
            ``fold_time``
                The time for fitting the estimator on the train
                set for each fold.
    """

    def __init__(self):
        fold_results = {
            "train_mse":    [],
            "train_mee":    [],
            "train_acc":    [],
            "val_mse":      [],
            "val_mee":      [],
            "val_acc":      [],
            "init_weigth":  [],
            "final_weigth": [],
            "fold_time":    []
        }
        super().__init__(fold_results)
    
    def add_row(self, train_mse, train_mee, train_acc,
                val_mse, val_mee, val_acc,
                init_weigth, final_weigth,
                fold_time):
            """Adds the results of a fold.
                    
            Parameters
            ----------

            train_mse : float
                The mse (Mean Squared Error) over the training set, for each fold.
            train_mee : float
                The mee (Mean Euclidian Error) over the training set, for each fold.
            train_acc : float
                The accuracy over the training set, for each fold.
            val_mse : float
                The mse (Mean Squared Error) over the validation set, for each fold.
            val_mee : float
                The mee (Mean Euclidian Error) over the validation set, for each fold.
            val_acc : float
                The accuracy over the validation set, for each fold.
            init_weigth : float
                Initial weights of the neural network, for each fold.
            final_weigth : float
                Final weights of the neural network, for each fold.
            fold_time : float
                The time for fitting the estimator on the train
                set for each fold.
            """
            self.append_value("train_mse", train_mse)
            self.append_value("train_mee", train_mee)
            self.append_value("train_acc", train_acc)
            self.append_value("val_mse", val_mse)
            self.append_value("val_mee", val_mee)
            self.append_value("val_acc", val_acc)
            self.append_value("init_weigth", init_weigth)
            self.append_value("final_weigth", final_weigth)
            self.append_value("fold_time", fold_time) 


def product_list(hyper_par_list):
    """Does the cartesian product of a a list of lists.

    Parameters
    ----------
    kwargs : list of lists

    Returns
    -------
    list
       a list of lists. Each list is a pair od the cartesian product.
    """
    return np.array(list(product(*hyper_par_list)))

def product_dict(**kwargs):
    """Does the cartesian product of a dict of lists.

    Parameters
    ----------
    kwargs : dict of lists

    Returns
    -------
    list
        a list of dict. Each dict is a pair od the cartesian product.

    Examples
    --------
    >>> param_grid = {"n_units": [[38], [40,50]], "learning_rate": [0.014, 0.017]}
    >>> param_list = list(product_dict(**param_grid))
    >>> param_list
    [{"n_units": [38], "learning_rate": 0.014}, {"n_units": [38], "learning_rate": 0.017}, 
    {"n_units": [40,50], "learning_rate": 0.014}, {"n_units": [40,50], "learning_rate": 0.017}]
    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))

def _num_samples(x):
    """Return number of samples in array-like x."""
    message = 'Expected sequence or array-like, got %s' % type(x)

    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, 'shape') and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError:
        raise TypeError(message)