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

  # compute the loss and the gradient
  num_classes = W.shape[1] # C
  num_train = X.shape[0] # D
  
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class = y[i]
    loss += -np.log( (np.exp(scores[correct_class]) / np.sum(np.exp(scores)) ) )
    for j in xrange(num_classes):
      if j == y[i]:
        dW[:, y[i]] += (np.exp(scores[j]) / np.sum(np.exp(scores)) - 1) * X[i, :]
      else:
        dW[:, j] += (np.exp(scores[j]) / np.sum(np.exp(scores))) * X[i, :]
        
  
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * np.sum(W * W)

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]

  scores = np.dot(X, W)

  loss_i = np.exp(scores)/np.sum(np.exp(scores), axis=1, keepdims=True)
  loss = np.sum(-np.log(loss_i[range(num_train), y]))

  der_scores = loss_i
  der_scores[range(num_train), y] -= 1
  dW = np.dot(X.T, der_scores)
  
  loss /= num_train
  dW /= num_train

  loss += reg * np.sum(W * W)
  dW += reg * W

  return loss, dW

