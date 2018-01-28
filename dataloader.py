import numpy as np


def load_from_csv(filename):
    return np.genfromtxt("./regression/" + filename, delimiter=",")
