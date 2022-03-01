import numpy as np
from typing import Tuple


def as_stride(x: np.ndarray, window_shape: Tuple[int, int], stride: Tuple[int, int]) -> np.ndarray:
    """as_stride
    Takes a batch_size x height x width x channels tensor and
    creates a moving-window view on it. E.g.

        # Convolution example
        x: 8 x 34 x 34 x 64 tensor
        window_shape: (3,3)
        stride: (1,1)
        ==> 8 x 32 x 32 x 3 x 3 x 64 tensor

        # Maxpool example
        x: 8 x 32 x 32 x 64 tensor
        window_shape: (2,2)
        stride: (2,2)
        ==> 8 x 16 x 16 x 2 x 2 x 64 tensor

    Parameters
    ----------
    x : np.ndarray
        Input Tensor
    window_shape : Tuple[int, int]
        Window view shape in height and weight respectively
    stride : Tuple[int, int]
        Stride shape in height and weight respectively

    Returns
    -------
    np.ndarray
        The x tensor from the view:
        batch_size x new_height x new_width x view_height x view_width x channels
    """

    # Memory cell step sizes in bytes
    s3 = x.strides[-1]

    # Batch, height, weight, channel
    b, h, w, c = x.shape

    # Kernel-height, Kernel-width
    kh, kw = window_shape

    # Calculate output height x width
    next_h = 1 + (h - kh) // stride[0]
    next_w = 1 + (w - kw) // stride[1]
    view_shape = (b, next_h, next_w, kh, kw, c)

    # Calculate next byte-step sizes, the numpy strides
    c_stride = s3
    kw_stride = c_stride * c
    kh_stride = c_stride * c * w
    next_w_stride = c_stride * c * stride[1]
    next_h_stride = c_stride * c * w * stride[0]
    b_stride = c_stride * c * w * h
    strides = (b_stride, next_h_stride, next_w_stride, kh_stride, kw_stride, c_stride)

    # Output tensor in form
    # batch_size x next_h x next_w x kh x kw x channel
    out = np.lib.stride_tricks.as_strided(x, view_shape, strides=strides)
    return out.copy()