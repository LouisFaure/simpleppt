import numpy as np
from . import __path__
import os


def load_example():

    diff = os.path.join(__path__[0], "data", "diffusion.csv")
    umap = os.path.join(__path__[0], "data", "umap.csv")

    data = np.loadtxt(diff, delimiter=",")
    emb = np.loadtxt(umap, delimiter=",")

    return data, emb
