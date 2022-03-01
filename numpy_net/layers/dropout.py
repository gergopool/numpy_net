import numpy as np
from .layer import Layer


class Dropout(Layer):
    """Dropout

    Drops out neurons randomly with a given chance.

    Parameters
    ----------
    p : float, optional
        Keep chance, by default 0.5
    """

    def __init__(self, p: float = 0.5):
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, x: np.ndarray) -> np.ndarray:
        mask = np.random.rand(*x.shape)
        self.mask = np.where(mask < self.p, 1, 0)
        return x * self.mask

    def backward(self, delta: np.ndarray) -> np.ndarray:
        return delta * self.mask