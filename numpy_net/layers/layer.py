import numpy as np


class Layer:

    def __init__(self):
        self._prev_Z = None  # Input activations to this layer, saved for brackprop
        self.is_trainable = False  # Flag if this layer has trainable parameters
        self._dW = None
        self._db = None
        self._last_dW_change = 0
        self._last_db_change = 0

    def __call__(self, x: np.ndarray, *args, **kwargs):
        # Save input
        self._prev_Z = x
        return self.forward(x, *args, **kwargs)

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, delta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update(self, *args, **kwargs):
        pass

    def update(self, lr: float, momentum: float = 0.9):
        W_was = self.W.copy()
        b_was = self.b.copy()
        self.W -= lr * self._dW + momentum * self._last_dW_change
        self.b -= lr * self._db + momentum * self._last_db_change

        self._last_dW_change = self.W - W_was
        self._last_db_change = self.b - b_was

        # Make sure we won't apply gradients twice
        self._dW = None
        self._db = None