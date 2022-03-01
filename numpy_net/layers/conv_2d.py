import numpy as np
from .layer import Layer
from .utils import as_stride

from typing import Union, Tuple


class Conv2D(Layer):
    """Conv

    Convolution layer. Accepts tensors in shape:
        batch_size x height x width x channels

    Parameters
    ----------
    in_dim : int
        Number of input channels
    out_dim : int
        Number of output channels
    k : Union[int, Tuple[int, int]], optional
        Kernel window size, by default 3
    """

    def __init__(self, in_dim: int, out_dim: int, k: Union[int, Tuple[int, int]] = 3):
        super(Conv2D, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k_size = k

        # check on kernel size
        if isinstance(self.k_size, int):
            self.k_size = (self.k_size, self.k_size)
        assert isinstance(self.k_size, list) or isinstance(self.k_size, tuple)
        assert len(self.k_size) == 2

        # TODO set strides available later
        self.stride = (1, 1)

        # Initialize weights
        sigma = np.sqrt(2 / (in_dim + out_dim))
        self.W = np.random.normal(0., sigma, size=(out_dim, in_dim, self.k_size[0], self.k_size[1]))
        self.b = np.zeros(out_dim)

        # Placeholder for differentials
        self._dW = None
        self._db = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Pad batch of images
        # TODO make it optional later
        p = (self.W.shape[-1] - 1) // 2
        x = np.pad(x, [(0, 0), (p, p), (p, p), (0, 0)])

        # Get sliding window view
        x_view = as_stride(x, self.k_size, self.stride)

        # Save. Make a copy to surely save arr instead of view
        self._prev_Z = x_view.copy()

        prod = np.einsum('bhwijc,kcij->bhwk', x_view, self.W)

        out = prod + self.b
        return out

    def backward(self, dZ: np.ndarray) -> np.ndarray:

        # dZ shape: batch_size x height x width x out_channels
        # prevZ shape: batch_size x height x width x kernel_height x kernel_width x in_channels
        # out shape: out_channels x in_channels x kernel_height x kernel_width
        self._dW = np.einsum('bhwo,bhwjki->oijk', dZ, self._prev_Z) / dZ.shape[0]

        # bias
        self._db = dZ.mean(axis=0)

        # dZ shape: batch_size x height x width x out_channels
        # kernel shape: out_channels x in_channels x kernel_height x kernel_width
        # next_dZ shape: batch_size x height x width x in_channels
        next_dZ = np.einsum('bhwo,oijk->bhwi', dZ, self.W) / dZ.shape[0]

        return next_dZ
