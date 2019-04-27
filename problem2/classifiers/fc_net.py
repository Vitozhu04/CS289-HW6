from builtins import range
from builtins import object
import numpy as np

from layers import *


class FullyConnectedNet(object):
    """
    A fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of [H, ...], and perform classification over C classes.

    The architecure should be like affine - relu - affine - softmax for a one
    hidden layer network, and affine - relu - affine - relu- affine - softmax for
    a two hidden layer network, etc.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim, hidden_dim=[10, 5], num_classes=8,
                 weight_scale=0.1):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: A list of integer giving the sizes of the hidden layers
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        """
        """
        #question2
        #self.params = {}
        #self.hidden_dim = hidden_dim
        ############################################################################
        # TODO: Initialize the weights and biases of the net. Weights              #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        
        #self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim[0])
        #self.params['b1'] = np.zeros(hidden_dim[0])
        #self.params['W2'] = weight_scale * np.random.randn(hidden_dim[0], num_classes)
        #self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        """
        #question3
        self.params = {}
        self.hidden_dim = hidden_dim
        self.num_layers = 1 + len(hidden_dim)
        
        layer_input_dim = input_dim
        for i, hd in enumerate(hidden_dim):
            self.params['W%d'%(i+1)] = weight_scale * np.random.randn(layer_input_dim, hd)
            self.params['b%d'%(i+1)] = weight_scale * np.zeros(hd)
            layer_input_dim = hd
        self.params['W%d'%(self.num_layers)] = weight_scale * np.random.randn(layer_input_dim, num_classes)
        self.params['b%d'%(self.num_layers)] = weight_scale * np.zeros(num_classes)
        #pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the net, computing the              #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        #problem 2
        """
        #a1, cache_fc = affine_forward(X, self.params['W1'], self.params['b1'])
        #out1, cache_relu = relu_forward(a1)
        #cache1 = (cache_fc, cache_relu)
        
        #out2, cache2 = affine_forward(out1, self.params['W2'], self.params['b2'])
        #scores = out2
        """
        #problem 3
        layer_input = X
        ar_cache = {}
        dp_cache = {}
    
        for lay in range(self.num_layers-1):
            #relu_forward
            layer_input, ar_cache[lay] = affine_relu_forward(layer_input, self.params['W%d'%(lay+1)], self.params['b%d'%(lay+1)])
        
        #affine_forward
        ar_out, ar_cache[self.num_layers] = affine_forward(layer_input, self.params['W%d'%(self.num_layers)], self.params['b%d'%(self.num_layers)])
        scores = ar_out
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the net. Store the loss            #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k].                                                          #
        ############################################################################
        """
        #problem 2
        loss, dscores = softmax_loss(scores, y)
        dx2, dw2, db2 = affine_backward(dscores, cache2)
        grads['W2'] = dw2 
        grads['b2'] = db2
        
        fc_cache, relu_cache = cache1
        da = relu_backward(dx2, relu_cache)
        dx1, dw1, db1 = affine_backward(da, fc_cache)
        
        grads['W1'] = dw1 
        grads['b1'] = db1 
        """
        
        #problem 3
        loss, dscores = softmax_loss(scores, y)
        dhout = dscores
 
        #affine_backward
        dx , dw , db = affine_backward(dhout , ar_cache[self.num_layers])
        grads['W%d'%(self.num_layers)] = dw
        grads['b%d'%(self.num_layers)] = db
        dhout = dx
        for idx in range(self.num_layers-1):
            lay = self.num_layers - 1 - idx - 1
            #relu_backward
            dx, dw, db = affine_relu_backward(dhout, ar_cache[lay])
            grads['W%d'%(lay+1)] = dw
            grads['b%d'%(lay+1)] = db
            dhout = dx
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads