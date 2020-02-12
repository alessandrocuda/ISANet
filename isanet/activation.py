"""Activation Functions Module.
"""

import numpy as np

class Activation:
    """Base class for the activation function.
    
    Warning: This class should not be used directly.
    Use derived classes instead.
    
    Methods
    -------
    f(x)
        return the value of activation function on x.

        Warning: empty method.
    
    derivative(x)
        return the derivative of the activation function on x.

        Warning: empty method.
    """
    def f(self, x):
        """Compute the activation function on x.

        Warning: Overrides this method in order to 
        implement the activation function.
        
        Parameters
        ----------
        x : array-like, shape = [n_samples, out_layer_dim]
            The output of a layer, usually correspond to:
            x = np.dot(A*W), where A is the input matrix to a layer
            and W is the weight matrix.
        """
        pass

    def derivative(self, x):
        """Compute the derivative of an activation function on x.

        Warning: Overrides this method in order to 
        implement the derivative of an activation function.
        
        Parameters
        ----------
        x : array-like
            It will performe the derivative on x
        """
        pass

class Sigmoid(Activation):
    """This class provide the logistic sigmoid function and its derivative.

    Parameters
    ----------

    a : float,
        a value usede to dilate and shrink the sigmoid:
                1 / (1 + exp(-a*x)).

    Methods
    -------
    f(x)
        return the value of activation function on x:
        f(x) = 1 / (1 + exp(-a*x)).
    
    derivative(x)
        return the derivative of the activation function on x:
        f'(x) = a*f(x)*(1-f(x)).
    """
    def __init__(self, a=1):
        self.a = a
    
    def f(self, x):
        """Compute the activation function on x.
        
        Parameters
        ----------
        x : array-like, shape = [n_samples, out_layer_dim]
            The output of a layer, usually correspond to:
            x = np.dot(A*W), where A is the input matrix to a layer
            and W is the weight matrix.

        Returns
        -------
        The value of activation function on x.
        """
        return 1 / ( 1 + np.exp(-self.a*x))

    def derivative(self, x):
        """Compute the derivative of an activation function on x.

        Parameters
        ----------
        x : array-like
            It will performe the derivative on x

        Returns
        -------
        return the derivative of the activation function on x.
        """
        x = self.f(x)
        return self.a*x * (1 - x) 

class Identity(Activation):
    """This class provide the identity function and its derivative.

    Methods
    -------
    f(x)
        return the value of activation function on x: f(x)=x.
    
    derivative(x)
        return the derivative of the activation function on x: 
        f'(x) = 1.
    """
    def f(self, x):
        """Compute the activation function on x.
        
        Parameters
        ----------
        x : array-like, shape = [n_samples, out_layer_dim]
            The output of a layer, usually correspond to:
            x = np.dot(A*W), where A is the input matrix to a layer
            and W is the weight matrix.

        Returns
        -------
        The value of activation function on x.
        """
        return x

    def derivative(self, x):
        """Compute the derivative of an activation function on x.

        Parameters
        ----------
        x : array-like
            It will performe the derivative on x

        Returns
        -------
        return the derivative of the activation function on x.
        """
        return np.ones(x.shape)

class Tanh(Activation):
    """This class provide the hyperbolic tan function and its derivative.

    Parameters
    ----------

    a : float,
        a value usede to dilate and shrink the tanh: tanh(a*x/2).

    Methods
    -------
    f(x)
        return the value of activation function on x:
        f(x) = tanh(a*x/2).
    
    derivative(x)
        return the derivative of the activation function on x:
        f'(x) = 1 - tanh(a*x/2)^2
    """
    def __init__(self, a=2):
        """Compute the activation function on x.
        
        Parameters
        ----------
        x : array-like, shape = [n_samples, out_layer_dim]
            The output of a layer, usually correspond to:
            x = np.dot(A*W), where A is the input matrix to a layer
            and W is the weight matrix.

        Returns
        -------
        The value of activation function on x.
        """
        self.a = a

    def f(self, x):
        """Compute the activation function on x.
        
        Parameters
        ----------
        x : array-like, shape = [n_samples, out_layer_dim]
            The output of a layer, usually correspond to:
            x = np.dot(A*W), where A is the input matrix to a layer
            and W is the weight matrix.

        Returns
        -------
        The value of activation function on x.
        """
        return np.tanh(self.a*x/2)

    def derivative(self, x):
        """Compute the derivative of an activation function on x.

        Parameters
        ----------
        x : array-like
            It will performe the derivative on x

        Returns
        -------
        return the derivative of the activation function on x.
        """      
        return 1 - np.tanh((self.a*x)/2)**2


class Relu(Activation):
    """This class provide the rectified linear unit function and its derivative.

    Methods
    -------
    f(x)
        return the value of activation function on x:
        f(x) = max(0,x)
    
    derivative(x)
        return the derivative of the activation function on x:
        if x > 0 return 1 else 0
    """
    def f(self, x):
        """Compute the activation function on x.
        
        Parameters
        ----------
        x : array-like, shape = [n_samples, out_layer_dim]
            The output of a layer, usually correspond to:
            x = np.dot(A*W), where A is the input matrix to a layer
            and W is the weight matrix.

        Returns
        -------
        The value of activation function on x.
        """
        return np.maximum(0,x)

    def derivative(self, x):
        """Compute the derivative of an activation function on x.

        Parameters
        ----------
        x : array-like
            It will performe the derivative on x

        Returns
        -------
        return the derivative of the activation function on x.
        """
        return (x > 0).astype(int)

# da sistemare la softmax
class Softmax(Activation):
    """Softmax activation function.

    Warning: this class has not been fully implemented.
    """
    def f(self, x):
        return np.exp(x)/np.sum(np.exp(x), axis=1)

    def derivative(self, x):
        return np.diagflat(self.f(x)) - np.dot(self.f(x), self.f(x).T)

