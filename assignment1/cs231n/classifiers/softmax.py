from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]

    for i in range(num_train):
      scores = X[i].dot(W)
      dS_dW = np.ones_like(W) * X[i].T[:,np.newaxis]
      scores -= scores.max() # shifting for calculation stability
      exp_scores = np.exp(scores)
      total_score = np.sum(exp_scores)
      soft = exp_scores/total_score
      loss -= np.log(soft[y[i]])

      # dloss/dsoft => -(1/soft[y[i]]) * (soft)'
      # case 1 y 
      # -(1/soft[y[i]]) * (exp(s_y)*sum(exp(s)) - exp(s_y)*exp(s_y))/(sum(exp(s))*sum(exp(s))) 
      # = -(1/soft[y[i]]) * exp(s_y) * (sum(exp(s)) - exp(s_y)/(sum(exp(s))*sum(exp(s)))
      # = -(1/soft[y[i]]) *  exp(s_y)/sum(exp(s)) * (1 - exp(s_y)/sum(exp(s)))
      # = -(1/soft[y[i]]) *  soft[y[i]] * (1 - soft[y[i]])
      # = soft[y[i]] - 1

      # case 2 other(j) : 
      # -(1/soft[y[i]]) * (- exp(s_y)*exp(s_j))/(sum(exp(s))*sum(exp(s))) 
      # = (1/soft[y[i]]) * exp(s_y)/(sum(exp(s)) * exp(s_j)/(sum(exp(s))
      # = (1/soft[y[i]]) *  soft[y[i]] * soft[j]
      # = soft[j]

      dL_dS = np.ones_like(soft) * soft 
      dL_dS[y[i]] -=  1
      
      dW += dL_dS * dS_dW
      
    loss = loss/num_train + reg* np.sum(W*W)
    dW = dW/num_train + 2*reg*W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]

    scores = X.dot(W)
    dS_dW = X.T
    scores -= scores.max(axis=1)[:,np.newaxis]
    exp_scores = np.exp(scores)
    total_score = np.sum(exp_scores,axis = 1)[:,np.newaxis]
    soft = exp_scores/total_score
    loss_ = -np.log(soft[range(num_train),y])
    loss = np.mean(loss_) + reg* np.sum(W*W)
    
    # dL_dS = np.ones_like(soft) * soft 
    dL_dS = soft 
    dL_dS[range(num_train),y] -=  1
    
    dW = np.dot(dS_dW,dL_dS)/num_train + 2*reg*W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
