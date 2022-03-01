import numpy as np


class Layer:

    def __init__(self):
        self._prev_Z = None  # Input activations to this layer, saved for brackprop
        self.is_trainable = False  # Flag if this layer has trainable parameters

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