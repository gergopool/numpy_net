import numpy as np

from .layers import Sigmoid, Softmax

__all__ = ["BinaryCrossEntropy", "MSE", "CrossEntropy"]

sigmoid = Sigmoid()
softmax = Softmax()


class Loss:
    """Loss
    General asbtract loss class
    """

    def __init__(self, eps=1e-6):
        self.eps = eps

    def __call__(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class BinaryCrossEntropy(Loss):

    def __call__(self, y_hat, y):
        batch_size = y.shape[0]
        y_hat = sigmoid(y_hat)
        y_hat = np.clip(y_hat, self.eps, 1 - self.eps)
        pos_terms = y * np.log(y_hat)
        neg_terms = (1 - y) * np.log(1 - y_hat)
        return np.sum((neg_terms - pos_terms) / batch_size)

    def backward(self, y_hat, y):
        batch_size = y.shape[0]
        return (y_hat - y) / batch_size


class MSE(Loss):

    def __call__(self, y_hat, y):
        batch_size = y.shape[0]
        return (y_hat - y)**2 * 0.5 / batch_size

    def backward(self, y_hat, y):
        batch_size = y.shape[0]
        return np.sum((y_hat - y) / batch_size)


class CrossEntropy(Loss):

    def __call__(self, y_hat, y):
        batch_size = y.shape[0]
        y_hat = softmax(y_hat, axis=-1)
        y_hat = np.clip(y_hat, self.eps, 1 - self.eps)
        return np.sum(-y * np.log(y_hat + self.eps) / batch_size)

    def backward(self, y_hat, y):
        batch_size = y.shape[0]
        grad = softmax(y_hat, axis=-1)
        grad -= y
        return grad / batch_size
