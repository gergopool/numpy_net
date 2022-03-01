import numpy as np
from .layer import Layer


class Sigmoid(Layer):
    """Sigmoid
    Sigmoid activation
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def backward(self, delta: np.ndarray) -> np.ndarray:
        return delta * self._out * (1 - self._out)