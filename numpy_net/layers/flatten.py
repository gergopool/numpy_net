import numpy as np
from .layer import Layer


class Flatten(Layer):
    """Flatten _summary_

    Takes a tensor in any shapes and reshapes it to 2 dimensions as:
        batch_size x (shape1 * shape2 * shape3 * ...)
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.shape_was = x.shape
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

    def backward(self, delta: np.ndarray) -> np.ndarray:
        return delta.reshape(self.shape_was)
