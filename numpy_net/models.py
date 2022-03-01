import numpy as np
from .layers import *


class Model:
    """Model
    General asbtract Model

    Parameters
    ----------
    inp_dim : int
        Input chanenls
    out_dim : int
        Output channels, number of classes
    """

    def __init__(self, inp_dim: int, out_dim: int):
        self.inp_dim = inp_dim
        self.out_dim = out_dim

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def update(self, dZ: np.ndarray, lr: float = 0.1) -> np.ndarray:

        # Calculate gradients
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ)

        # Update weights
        for layer in reversed(self.layers):
            if layer.is_trainable:
                layer.update(lr)


class DenseModel(Model):
    """DenseModel
    Simple feed forward linear model.

    Parameters
    ----------
    n_hidden : int, optional
        Number of hidden layers, by default 5
    h_dim : int, optional
        Dimension of hidden layers, by default 32
    """

    def __init__(self, *args, n_hidden: int = 5, h_dim: int = 32, **kwargs):
        super(DenseModel, self).__init__(*args, **kwargs)
        self.layers = [Linear(self.inp_dim, h_dim)]
        for i in range(n_hidden):
            self.layers.extend([Linear(h_dim, h_dim), ReLU()])
        self.layers.append(Linear(h_dim, self.out_dim))


class ConvModel(Model):
    """ConvModel
    Simple feed forward convolutional model.

    Parameters
    ----------
    h_dim : int, optional
        Dimension of hidden layers after flatten layer, by default 128
    conv_dim : int, optional
        Dimension of conv layers, by default 8
    """

    def __init__(self, *args, h_dim: int = 128, conv_dim: int = 8, **kwargs):
        super(ConvModel, self).__init__(*args, **kwargs)

        self.layers = [
            Conv2D(self.inp_dim, conv_dim),  # 28
            ReLU(),
            MaxPool2D(),
            Conv2D(conv_dim, conv_dim),  # 14
            ReLU(),
            MaxPool2D(),  # 7
            Flatten(),
            Linear(conv_dim * 7 * 7, h_dim),
            ReLU(),
            Linear(h_dim, self.out_dim)
        ]