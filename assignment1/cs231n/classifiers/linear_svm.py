from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin

                # 추가 - margin w에 대해 편미분
                # scores[j] = X[i]W[j], correct_class_score = X[i]W[y[i]]
                # dW[:,j] += X[i], dW[:,y[i]] -= X[i], 즉 틀린 값이 커질 수록, 정답 값이 맞을 수록 loss가 커짐
                dW[:,j] = np.add(dW[:,j],X[i])
                dW[:,y[i]] = np.subtract(dW[:,y[i]],X[i])
                # 추가 끝

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # dW 역시 num_train으로 나누기
    dW = dW/num_train
    # reg의 loss만큼 편미분 값을 더함 => 2W *reg
    dW = np.add(dW,2*reg*W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    scores = np.dot(X,W) # 500, 10
    score_y = scores[range(num_train),y][:,np.newaxis] # 500, => 500,1

    margins = np.maximum(0, scores - score_y + 1) # 500,10 - 500,1 => 500,10
    margins[range(num_train),y] = 0

    temp_loss = np.sum(margins) /num_train
    reg_loss = reg*np.sum(W*W)
    loss = np.add(temp_loss,reg_loss)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Regulation Loss
    dReg = 2*reg*W 

    # cause_loss(i,j) == 1 => dW[:,j] += X[i,:], dW[:,y[i]] -= X[i,:]
    # loss를 유발한 경우/train <=> margin > 0 
    cause_loss = np.array(margins > 0,dtype = np.int32) # count the number of +s_j using margin above
    cause_loss[range(num_train),y] = -np.sum(cause_loss,axis=1) # count the number of -s_y for each y

    # X.T(3072, 500) dot cause_loss(500,10) => dW
    dW = np.dot(X.T,cause_loss)/ num_train +dReg
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
