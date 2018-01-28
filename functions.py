import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def calculate_weight(input_vec, output_vec):
    t = np.transpose(input_vec)
    return (1/(np.dot(t, input_vec))) * np.dot(t, output_vec)


def create_io_arrays(dataset):
    inputs = []
    set_size = len(dataset[0])
    for data in dataset:
        o = [1]
        for i in range(0, set_size - 1):
            o.append(data[i])
        inputs.append(o)
    output = []
    for data in dataset:
        output.append(data[set_size - 1])
    res = [inputs, output]
    return res


def find_ols(inputs, outputs):
    transposed_input = np.transpose(inputs)

    dot = np.dot(transposed_input, inputs)
    a = np.linalg.pinv(dot, -1)
    b = np.dot(transposed_input, outputs)
    return np.dot(a, b)


def calcError(inputs, outputs, weights):
    a = np.subtract(np.dot(inputs, weights), outputs)
    return np.dot(np.transpose(a), a)


def split_twodim_arr(arr):
    x = []
    y = []
    for set in arr:
        x.append(set[0])
        y.append(set[1])
    return [x, y]


def create_linreg_line(data, weights):
    line = []
    for set in data:
        line.append([set[0], set[0]*weights[1] + weights[0]])
    return line


def plot(arr, arr1, line):
    arr = split_twodim_arr(arr)
    arr1 = split_twodim_arr(arr1)
    line = split_twodim_arr(line)
    mpl.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots()
    ax.plot(arr[0], arr[1], 'bo')
    ax.plot(arr1[0], arr1[1], 'yo')
    ax.plot(line[0], line[1], 'k-', linewidth=5.0)
    ax.set_title('Using hyphen instead of Unicode minus')
    plt.show()