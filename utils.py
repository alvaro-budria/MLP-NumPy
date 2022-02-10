import os
import gzip
from typing import Iterator

import requests
import matplotlib.pyplot as plt

import numpy as np


def batch_generator(
    idata: np.ndarray,
    target: np.ndarray,
    batch_size: int,
    shuffle: bool = True
) -> Iterator[np.ndarray]:

    nsamples = len(idata)
    if shuffle:
        perm = np.random.permutation(nsamples)
    else:
        perm = range(nsamples)

    for i in range(0, nsamples, batch_size):
        batch_idx = perm[i:i+batch_size]
        if target is not None:
            yield idata[batch_idx], target[batch_idx]
        else:
            yield idata[batch_idx], None


def flatten_image(im: np.ndarray) -> np.ndarray:
    assert(2 <= len(im.shape) <= 3)
    if len(im.shape) == 2:  # no batch dimension, single image
        return im.reshape(-1)
    return im.reshape(im.shape[0], -1)


def unflatten_image(im: np.ndarray, n: int, m: int) -> np.ndarray:
    assert(1 <= len(im.shape) <= 2)
    if len(im.shape) == 1:
        return im.reshape(n, m)  # no batch dimension, single image
    return im.reshape(im.shape[0], n, m)


def vectorize_target(y: np.array) -> np.array:
    e = np.zeros((10))
    e[y] = 1.0
    return e


def normalize_image(im: np.array) -> np.array:
    return (im - im.mean()) / im.std()


def get_file(
    fname: str,
    origin: str,
    cache_subdir: str
) -> str:
    """Downloads the file from URL is it is not yet in cache
    Arguments
    fname: name of the file
    origin: Remote URL of the file

    Return: Path to downloaded file
    """

    cache_dir = os.path.join(os.path.expanduser('~'), '.datasets') # create temporary download location
    datadir = os.path.join(cache_dir, cache_subdir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    fpath = os.path.join(datadir, fname)

    if not os.path.exists(fpath):
        print(fname, "does not exist")
        print("Downloading data from", origin)
        r = requests.get(origin, stream = True)
        with open(fpath, 'wb') as file:
            for chunk in r.iter_content(chunk_size = 1024):
                if chunk:
                    file.write(chunk)
        print("Finished downloading ", fname)

    return fpath


def load_data():
    """ Loads the Fashion MNIST dataset
    return: x_train, y_train, x_test, y_test
    """
    dirname = os.path.join('datasets', 'fashion_mnist')
    base = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']

    paths = []

    for fname in files:
        paths.append(get_file(fname, origin=base + fname, cache_subdir=dirname))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8,
                                offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8,
                               offset=16).reshape(len(y_test), 28, 28)

    y_train = np.array([vectorize_target(i) for i in y_train])
    y_test = np.array([vectorize_target(i) for i in y_test])

    return x_train, y_train, x_test, y_test


def draw_metrics(
    lst_iter: np.array,
    train: np.array,
    title_train: np.array,
    val: np.array,
    title_val: np.array,
    title: np.array
) -> None:

    plt.plot(lst_iter, train, '-b', label=title_train)
    plt.plot(lst_iter, val, '-r', label=title_val)

    plt.xlabel("epoch")
    plt.legend(loc='upper left')
    plt.title(title)

    # save image
    plt.savefig(title+".png")
    plt.close()
