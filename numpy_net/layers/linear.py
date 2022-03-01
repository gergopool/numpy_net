import numpy as np
from .layer import Layer


class Linear(Layer):
    """Linear

    Simple dense linear layer. Input dimension must be 2 as:
        batch_size x num_dimension

    Parameters
    ----------
    in_dim : int
        Number of input neurons
    out_dim : int
        Number of output neurons
    """

    def __init__(self, in_dim: int, out_dim: int):
        super(Linear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Since linear layer has weights, it is a trainable layer
        self.is_trainable = True

        # Initialize weights
        sigma = np.sqrt(2 / (in_dim + out_dim))
        self.W = np.random.normal(0., sigma, size=(out_dim, in_dim))
        self.b = np.zeros(out_dim)

        # Placeholder for differentials
        self._dW = None
        self._db = None

    def forward(self, prev_Z: np.ndarray) -> np.ndarray:
        return prev_Z @ self.W.T + self.b

    def backward(self, dZ: np.ndarray) -> np.ndarray:  # batch_size x ... x out_dim
        self._dW = dZ.T @ self._prev_Z / dZ.shape[0]
        self._db = dZ.mean(axis=0)
        return dZ @ self.W