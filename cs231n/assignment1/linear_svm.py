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
                dW[:,j:j+1] = dW[:,j:j+1] + np.reshape(X[i], (-1,1))
                dW[:,y[i]:y[i]+1] = dW[:,y[i]:y[i]+1] - np.reshape(X[i],(-1,1))

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

    dW /= num_train
    dW += 2 * reg * W

    # for i in range(num_train):
    #     scores = X[i].dot(W)
    #     correct_class_score = scores[y[i]]
    #     for j in range(num_classes):
    #         if j == y[i]:
    #             continue
    #         margin = scores[j] - correct_class_score + 1  # note delta = 1
    #         if margin > 0:
    #             loss += margin

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

    num_classes = W.shape[1]
    num_train = X.shape[0]
    S = np.dot(X,W) #score matrix
    M = S - np.reshape(S[np.arange(num_train), y], (-1, 1)) + 1 # margin matrix
    M[np.arange(num_train), y] = 0
    loss = loss + np.sum(M[M > 0])
    loss /= num_train
    loss += reg * np.sum(W * W)
    

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
    
    M_pos = np.zeros_like(M) 
    M_pos[M>0] = 1             #M_pos 는 M이 양수면 1, 아니면 0.
    M_count = np.sum(M_pos, axis = 1)   # 행별로 M이 양수인 개수를 셈
    M_count = M_count.reshape(-1,1)
    M_label = np.zeros_like(M)
    M_label[np.arange(num_train), y] = 1    # 정답에 해당하는 칸은 1, 아니면 0
    M_labelcount = M_count * M_label    # 각 행별로 정답의 위치와 양수의 개수가 들어있음
    dW = dW + np.dot(X.T, M_pos - M_labelcount)
    # dW = dW + np.dot(X.T, M_pos) - np.dot(X.T, M_labelcount) 를 합친 것.
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
