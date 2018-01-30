import numpy as np


def load_from_csv(filename, task):
    return np.genfromtxt(("./regression/" if task == 1 else "./classification/") + filename, delimiter=",")
