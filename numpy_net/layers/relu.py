import numpy as np
from .layer import Layer


class ReLU(Layer):
    """ReLU 
    ReLU activation.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def backward(self, delta: np.ndarray) -> np.ndarray:
        return delta * np.where(self._prev_Z < 0, 0, 1)