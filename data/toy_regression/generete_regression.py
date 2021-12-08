import numpy as np
from data.toy_regression.regression1d_data import ToyRegression

def generate_1d_dataset(show_plots=True, task_set=0, data_random_seed=42):
    """Generate a set of tasks for 1D regression.

    Args:
        show_plots: Visualize the generated datasets.
        data_random_seed: Random seed that should be applied to the
            synthetic data generation.
        task_set: int for the regression task 

    Returns:
        data_handlers: A data handler
    """

    map_funcs = [lambda x: (x) ** 3.,
                 lambda x: (3.*x),
                 lambda x: 2. * np.power(x, 2) - 1,
                 lambda x: np.power(x - 3., 3),
                 lambda x: x*np.sin(x),
                 lambda x: x*(1+np.sin(x))]
    x_domains = [[-3.5, 3.5],[-2, 2], [-1, 1], [2, 4],[2,6],[3,12]]
    test_domains = [[-5.0,5.0],[-3, +3], [-2.5, 2.5], [.5, 4.1],[0,7],[2,13]]
    std = [3,0.05,0.05,0.05,0.25,0.6]
    num_train = 90
    blob = [None,None,None,None,[[1.5,2.5],[4.5,6.0]],[[4.5,5],[7.5,8.5],[10,11]]]


    data = ToyRegression(train_inter=x_domains[task_set],
                                       num_train=num_train, test_inter=test_domains[task_set], blob = blob[task_set], num_test=100,
                                       val_inter=x_domains[task_set], num_val=100,
                                       map_function=map_funcs[task_set], std=std[task_set], rseed=data_random_seed)
    return data

