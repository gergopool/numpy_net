# Credits: https://github.com/hsjeong5/MNIST-for-Numpy

import os
import numpy as np
from urllib import request
import gzip
import pickle
import uuid
import shutil

__all__ = ['load']

filenames = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "val_x": "t10k-images-idx3-ubyte.gz",
    "val_y": "t10k-labels-idx1-ubyte.gz"
}


# Download mnist to temp folder
def download_mnist(save_folder):
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name, filename in filenames.items():
        print("Downloading " + name + "...")
        url = os.path.join(base_url, filename)
        save_path = os.path.join(save_folder, filename)
        request.urlretrieve(url, save_path)
    print("Download complete.")


# Mnist files to pickle
def save_mnist(save_folder):
    mnist = {}
    for name, filename in filenames.items():
        path = os.path.join(save_folder, filename)

        with gzip.open(path, 'rb') as f:
            if name.endswith('_x'):
                mnist[name] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
            else:
                mnist[name] = np.frombuffer(f.read(), np.uint8, offset=8)

    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist, f)
    print("Save complete.")


# Create temp folder
def create_temp_folder(save_folder):
    os.makedirs(save_folder, exist_ok=True)


# Delete temp folder with all content
def remove_temp_folder(save_folder):
    shutil.rmtree(save_folder)


# Download mnist and save locally
def init():
    save_folder = str(uuid.uuid4())  # Temp folder, will be created and deleted
    create_temp_folder(save_folder)
    download_mnist(save_folder)
    save_mnist(save_folder)
    remove_temp_folder(save_folder)


# Load in mnist data
def load(download=True):
    if download and not os.path.isfile("mnist.pkl"):
        init()
    with open("mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    return [mnist[k] for k in ['train_x', 'train_y', 'val_x', 'val_y']]
