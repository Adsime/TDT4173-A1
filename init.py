import dataloader as dl
import functions

train_files = ["train_1d_reg_data.csv", "train_2d_reg_data.csv", "cl_train_1.csv", "cl_train_2.csv"]
test_files = ["test_1d_reg_data.csv", "test_2d_reg_data.csv", "cl_test_1.csv", "cl_test_2.csv"]


def task_1_2():
    print("Start task 1.2\n")
    train_dataset = dl.load_from_csv(train_files[1], 1)
    test_dataset = dl.load_from_csv(test_files[1], 1)

    train_io_array = functions.create_io_arrays(train_dataset, True)
    test_io_array = functions.create_io_arrays(test_dataset, True)

    train_weights = functions.find_ols(train_io_array[0], train_io_array[1])

    print("Train data:\nWeights: " + train_weights.__str__())
    print("Error: " + functions.calcError(train_io_array[0], train_io_array[1], train_weights).__str__())
    print("\n\nTest data:\nWeights: " + train_weights.__str__())
    print("Error: " + functions.calcError(test_io_array[0], test_io_array[1], train_weights).__str__())
    print("\nEnd task 1.2")


def task_2_1_2():
    print("\n\nStart task 2.1.2")
    train_dataset = dl.load_from_csv(train_files[0], 1)
    train_io_array = functions.create_io_arrays(train_dataset, True)
    weights = functions.find_ols(train_io_array[0], train_io_array[1])
    test_dataset = dl.load_from_csv(test_files[0], 1)
    line = functions.create_linreg_line(test_dataset, weights)
    functions.plot(train_dataset, test_dataset, line)
    print("End task 2.1.2")


def load_data(linear, dataset):
    train_dataset = functions.create_io_arrays(dl.load_from_csv(train_files[dataset], 2), linear)
    test_dataset = functions.create_io_arrays(dl.load_from_csv(test_files[dataset], 2), linear)

    weights = [0] * len(train_dataset[0][0])

    train_error = []
    test_error = []

    # Does 1000 epochs of training.
    for i in range(0, 1000):
        weights = functions.train_function(weights, train_dataset[0], train_dataset[1], 0.1)
        train_error.append(functions.cross_entropy(weights, train_dataset[0], train_dataset[1]))
        test_error.append(functions.cross_entropy(weights, test_dataset[0], test_dataset[1]))
    return [train_dataset, test_dataset, weights, train_error, test_error]


def task_2_2_1():
    print("\n\nStart task 2.2.1")
    linear = True
    data = load_data(linear, 2)
    train_dataset = data[0]
    test_dataset = data[1]
    weights = data[2]
    train_error = data[3]
    test_error = data[4]
    functions.plot_error(train_error, test_error)
    functions.plot3d(train_dataset, test_dataset, weights, linear)
    print("End task 2.2.1")


def task_2_2_2(linear):
    print("\n\nStart task 2.2.2")
    data = load_data(linear, 3)
    train_dataset = data[0]
    test_dataset = data[1]
    weights = data[2]
    train_error = data[3]
    test_error = data[4]
    functions.plot_error(train_error, test_error)
    functions.plot3d(train_dataset, test_dataset, weights, linear)
    print("End task 2.2.2")


task_1_2()
task_2_1_2()
task_2_2_1()
task_2_2_2(True)
task_2_2_2(False)