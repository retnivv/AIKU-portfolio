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

    N = X.shape[0]
    C = W.shape[1]
    for i in range(N):
        scores = np.dot(X[i],W)
        scores = np.exp(scores)
        s_sum = np.sum(scores)
        loss -= np.log(scores[y[i]]/s_sum)
        for j in range(C):
            t = 0
            if (j == y[i]):
                t = 1
            ds = (scores[j] / s_sum) - t
            dW[:,j] += ds * X[i].T 
            
    
    loss = loss / N
    loss += reg * np.sum(W * W)
    dW = dW / N
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    
    N = X.shape[0]
    C = W.shape[1]
    scores = np.dot(X,W)  # N x C
    scores = np.exp(scores)
    s_sum = np.sum(scores, axis = 1)    # N x 1
    loss -= np.sum(np.log(scores[np.arange(N), y] / s_sum))

    ds = scores / s_sum.reshape(-1,1)
    ds[np.arange(N),y] -= 1
    dW = np.dot(X.T, ds)

    loss = loss / N
    loss += reg * np.sum(W * W)
    dW = dW / N
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
