import numpy as np
from .layer import Layer


class Softmax(Layer):
    """Softmax
    Softmax activation
    """

    def forward(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        # Safe softmax
        max_values = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - max_values)
        return e_x / e_x.sum(axis=axis, keepdims=True)

    def backward(self, delta: np.ndarray) -> np.ndarray:
        # TODO
        # I have got lost in this one, but actually it doesn't
        # matter that much because in cross entropy loss it all
        # works out into softmax(y_hat)-y
        NotImplementedError