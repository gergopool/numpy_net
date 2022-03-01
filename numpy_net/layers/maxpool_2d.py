import numpy as np
from .layer import Layer
from .utils import as_stride

from typing import Union, Tuple


class MaxPool2D(Layer):
    """MaxPool2D

    Performs pooling on a 4d tensor, groupping the middle
    two dimensions and taking max values only.

    E.g.
        in_shape: 10 x 32 x 32 x 7
        out_shape: 10 x 16 x 16 x 7

    Parameters
    ----------
    pool_size : Union[int, Tuple[int, int]], optional
        The size of pool by default 2
    """

    def __init__(self, pool_size: Union[int, Tuple[int, int]] = 2):
        super(MaxPool2D, self).__init__()
        self.pool_size = pool_size

        # Check on pool size
        if isinstance(self.pool_size, int):
            self.pool_size = (self.pool_size, self.pool_size)
        assert isinstance(self.pool_size, list) or isinstance(self.pool_size, tuple)
        assert len(self.pool_size) == 2

        # TODO: Situations where these are not equal
        self.stride = self.pool_size

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Get a view in shape
        # b_size x h//pool x w//pool x pool x pool x channels
        x_strided = as_stride(x, self.pool_size, self.stride)

        # Change dimensions to
        # b_size x h//pool x w//pool x channels x pool x pool
        x_perm = x_strided.transpose(0, 1, 2, 5, 3, 4)

        # Flat last dimensions to
        # b_size x h//pool x w//pool x channels x (pool * pool)
        x_perm_flat = x_perm.reshape(*x_perm.shape[:4], -1)

        # Save input in this form instead
        self._prev_Z = x_perm_flat.copy()

        # Return with max pooled values
        return x_perm_flat.max(axis=-1)

    def backward(self, dZ):

        # Get argmax values in each cell in form
        # (b_size * h//pool * w//pool * channels) x (pool * pool)
        n = np.prod(self.pool_size)
        indices = np.eye(n)[self._prev_Z.argmax(axis=-1).ravel()]

        # Replace ones with dZ values
        dZ = indices * dZ.ravel()[:, None]

        # Reshape dZ to
        # b_size x h//pool x w//pool x channels x (pool * pool)
        dZ = dZ.reshape(*self._prev_Z.shape)

        # Get back original shape as
        # b_size x h x w x channels
        # Currently this is only safe when h%pool and w%pool are both 0
        dZ = self.reconstruct(dZ, self.pool_size)
        return dZ

    def reconstruct(self, arr: np.ndarray, pool_size: tuple) -> np.ndarray:
        """reconstruct
        Upsampling the output of maxpool back into the input.

        Parameters
        ----------
        arr : np.ndarray
            The input array in shape
            b_size x h//pool x w//pool x channels x (pool * pool)
        pool_size : tuple
            The original pool size

        Returns
        -------
        np.ndarray
            The array in order:
            b_size x h x w x channels
        """
        b_size, h, w, c, _ = arr.shape
        ph, pw = pool_size
        target_shape = (b_size, h * ph, w * pw, c)
        z = np.zeros(target_shape, dtype=arr.dtype)

        for h_i in range(h * ph):
            for w_i in range(w * pw):
                h_j = h_i // ph
                h_k = h_i % ph
                w_j = w_i // pw
                w_k = w_i % pw
                k = h_k * pw + w_k

                z[:, h_i, w_i] = arr[:, h_j, w_j, :, k]

        return z