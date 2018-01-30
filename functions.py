import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mpl_toolkits.mplot3d as mplt

def calculate_weight(input_vec, output_vec):
    t = np.transpose(input_vec)
    return (1/(np.dot(t, input_vec))) * np.dot(t, output_vec)


def create_io_arrays(dataset, linear):
    inputs = []
    set_size = len(dataset[0])
    for data in dataset:
        o = [1]
        for i in range(0, set_size - 1):
            o.append(data[i])
        if not linear:
            for i in range(1, len(o)):
                o.append(o[i]*o[i])
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


# Task 2

def plot_error(train, test):
    mpl.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots()
    ax.plot(train, 'bo')
    ax.plot(test, 'ro')
    ax.set_title('Blue is train, red is red is test')
    plt.show()


def apply_legend(fig):
    test_0 = mpatches.Patch(color="red")
    test_1 = mpatches.Patch(color="blue")
    train_0 = mpatches.Patch(color="orange")
    train_1 = mpatches.Patch(color="black")
    fig.legend(handles=[test_0, test_1, train_0, train_1], labels=["test data output 0", "test data output 1",
                                                                   "train data output 0", "train data output 1"])


def scatter_plot(ax, dataset, isTest):
    """

    :param ax: subplot
    :param dataset: set of inputs and outputs
    :param isTest: indicates if the scatter data is test data or training data
    :return:
    """
    one = "blue" if isTest else "black"
    zero = "red" if isTest else "orange"
    for i, set in enumerate(dataset[0]):
        # Creates a subplot for each dataset. Colors the point according to which value it holds as output.
        ax.scatter(set[1], set[2], dataset[1][i], c=(one if dataset[1][i] > 0.5 else zero))


def plot3d(x, t, w, linear):
    fig = plt.figure()
    # Creating two planes which will be manipulated to represent decision boundary
    xx, yy = np.meshgrid(np.arange(0, 1, 0.05), np.arange(0, 1, 0.04))
    ax = fig.add_subplot(111, projection='3d')
    # Builds up
    zz = w[0] + w[1]*xx + w[2]*yy + (w[3]*xx*xx + w[4]*yy*yy if not linear else 0)
    ax.plot_wireframe(xx, yy, zz, color="y")
    ax.set_zlim(-1, 2)
    apply_legend(fig)
    scatter_plot(ax, x, True)
    scatter_plot(ax, t, False)
    mpl.pyplot.show()


def h_function(weights, inputs):
    return np.dot(np.transpose(weights), inputs)


def sigmoid(h):
    return 1 / (1 + np.exp(-h))


def cross_entropy(weights, inputs, outputs):
    error = 0
    for i, set in enumerate(inputs):
        y = outputs[i]
        s = sigmoid(h_function(weights, set))
        error += (y * np.log(s)) + ((1 - y) * np.log(1 - s))
    return -(error/len(inputs))


def train_function(weights, inputs, outputs, leaning_rate):
    experience = [0] * len(inputs[0])
    for i, set in enumerate(inputs):
        h = h_function(weights, set)
        experience = np.add(experience, np.multiply((sigmoid(h) - outputs[i]), set))
    return np.subtract(weights, np.multiply(leaning_rate, experience))