import dataloader as dl
import functions
import numpy as np

train_files = ["train_1d_reg_data.csv", "train_2d_reg_data.csv", "cl_train_1.csv", "cl_train_2.csv"]
test_files = ["test_1d_reg_data.csv", "test_2d_reg_data.csv", "cl_test_1.csv", "cl_test_2.csv"]


def task_1_2():
    train_dataset = dl.load_from_csv(train_files[1])
    test_dataset = dl.load_from_csv(test_files[1])

    train_io_array = functions.create_io_arrays(train_dataset)
    test_io_array = functions.create_io_arrays(test_dataset)

    train_weights = functions.find_ols(train_io_array[0], train_io_array[1])

    print("Train data:\nWeights: " + train_weights.__str__())
    print("Error: " + functions.calcError(train_io_array[0], train_io_array[1], train_weights).__str__())
    print("\n\nTest data:\nWeights: " + train_weights.__str__())
    print("Error: " + functions.calcError(test_io_array[0], test_io_array[1], train_weights).__str__())

def task_2_1_2():
    train_dataset = dl.load_from_csv(train_files[0])
    train_io_array = functions.create_io_arrays(train_dataset)
    weights = functions.find_ols(train_io_array[0], train_io_array[1])
    test_dataset = dl.load_from_csv(test_files[0])
    line = functions.create_linreg_line(test_dataset, weights)
    functions.plot(train_dataset, test_dataset, line)

#task_1_2()
task_2_1_2()
