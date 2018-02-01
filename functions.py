import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mpl_toolkits.mplot3d as mplt


def create_io_arrays(dataset, linear):
    """
    Using a matrix where the last number is considered an output; this will split the input and output params into
    two different arrays. Also, if the tast is not considered solvable linearly, two additional parameters
    are added to the input sets. These are just the squares of each input already present.
    :param dataset: matrix
    :param linear: boolean
    :return: array
    """
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
    """
    Will find ordinary least squares which is used as weights. Following function is used:
    (X^T X)^-1 X^T y
    :param inputs: inputs array
    :param outputs: outputs array
    :return: array
    """
    transposed_input = np.transpose(inputs)
    dot = np.dot(transposed_input, inputs)
    a = np.linalg.pinv(dot, -1)
    b = np.dot(transposed_input, outputs)
    return np.dot(a, b)


def calcError(inputs, outputs, weights):
    """
    Using the following formula to calculate the error:
    (Xw - y)^T (Xw - y)
    :param inputs: inputs array
    :param outputs: outputs array
    :param weights: weights array
    :return: float
    """
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
    """
    Using the calculated weights with respect to the data sets, this will create a line which can be plotted.
    It will represent a suggestion for linear regression.
    :param data: dataset array
    :param weights: weights array
    :return: array
    """
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
    fig.legend([test_0, test_1, train_0, train_1], ["test data output 0", "test data output 1", "train data output 0", "train data output 1"])


def scatter_plot(ax, dataset, isTest):
    """
    Using a subplot, the data from dataset will be plotted. isTest indicates if the data is training or test data.
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
    """
    Will create meshgrids for x and y axis as well as a grid for the z-axis for linear and non-linear problems.
    :param x: dataset matrix
    :param t: dataset matrix
    :param w: weights array
    :param linear: boolean indicating if the plot should only account for linear or non-linear data.
    """
    fig = plt.figure()
    apply_legend(fig)
    # Creating two matrices which will be manipulated to represent decision boundary
    xx, yy = np.meshgrid(np.arange(0, 1, 0.05), np.arange(0, 1, 0.04))
    ax = fig.add_subplot(111, projection='3d')
    # Builds up the z modifiers of the decision boundary
    zz = w[0] + w[1]*xx + w[2]*yy + (w[3]*xx*xx + w[4]*yy*yy if not linear else 0)
    # Plots a figure using the values created above.
    ax.plot_wireframe(xx, yy, zz, color="y")
    ax.set_zlim(-1, 2)
    scatter_plot(ax, x, True)
    scatter_plot(ax, t, False)
    mpl.pyplot.show()


def h_function(weights, inputs):
    """
    Implementation of the following function:
    h(x) = w^T x
    :param weights: weights array
    :param inputs: inputs matrix
    :return:
    """
    return np.dot(np.transpose(weights), inputs)


def sigmoid(h):
    """
    Implementation of the following function:
    σ(z) = 1/(1+e^-z)
    :param h: value returned from h_function
    :return: float
    """
    return 1 / (1 + np.exp(-h))


def cross_entropy(weights, inputs, outputs):
    """
    Using an iteration of weights, this will iterate over all the inputs and calculate the difference of predicted
    versus actual output.
    Implementation of the following function:
    (-1/2) * sum from i = 0 to n, yi*lnσ(z) + (1-yi) * ln(1-σ(z))
    :param weights: weights array
    :param inputs: inputs matrix
    :param outputs: outputs array
    :return: float
    """
    error = 0
    for i, set in enumerate(inputs):
        y = outputs[i]
        s = sigmoid(h_function(weights, set))
        error += (y * np.log(s)) + ((1 - y) * np.log(1 - s))
    return -(error/len(inputs))


def train_function(weights, inputs, outputs, leaning_rate):
    """
    Training function implemented from the exercise paper.
    Iterates over all the inputs, adding up differences in predicted vs actual data with respect to the individual sets
    of data.
    :param weights: weights array
    :param inputs: inputs matrix
    :param outputs: outputs array
    :param leaning_rate: rate of which the algorithm should modify the weights.
    :return: float
    """
    experience = [0] * len(inputs[0])
    for i, set in enumerate(inputs):
        h = h_function(weights, set)
        experience = np.add(experience, np.multiply((sigmoid(h) - outputs[i]), set))
    return np.subtract(weights, np.multiply(leaning_rate, experience))