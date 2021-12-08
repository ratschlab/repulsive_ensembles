import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from warnings import warn

from data.dataset import Dataset

class ToyRegression(Dataset):
    """An instance of this class represents a simple regression task.

    Attributes: (additional to baseclass)
        train_inter(tuple): tuple representing the interval in which sample training datapoints.
        num_train(int): number of training datapoints.
        test_inter(tuple): tuple representing the interval in which sample testing datapoints.
        blob(list): used to create blob regression task, list containing 4 extremes of two intervals in which sample
            training datapoints
        num_test(int) number of tasting datapoints
        val_inter(tuple):
        num_val(int):
        map_function: function that maps inputs to output
        std(float): std of gaussian noise to add to the outputs
        rseed(int): random state

    """
    def __init__(self, train_inter=[-10, 10], num_train=20,
                 test_inter=[-10, 10], blob = None, num_test=80, val_inter=None,
                 num_val=None, map_function=lambda x : x, std=0, rseed=None):

        super().__init__()

        assert(val_inter is None and num_val is None or \
               val_inter is not None and num_val is not None)

        if rseed is None:
            rand = np.random
        else:
            rand = np.random.seed(rseed)
        if blob is None:
            train_x = rand.uniform(low=train_inter[0], high=train_inter[1],
                                   size=(num_train, 1))
        else:
            train_x = np.vstack([np.random.uniform(low=i[0], high=i[1],size=(int(num_train/len(blob)), 1)) for i in blob])
        #train_x = np.asarray([[-0.8, -0.1, 0.02, 0.2, 0.6, 0.8]]).T
        #num_train = train_x.shape[0]
        test_x = np.linspace(start=test_inter[0], stop=test_inter[1],
                             num=num_test).reshape((num_test, 1))

        train_y = map_function(train_x)
        test_y = map_function(test_x)

        # Perturb training outputs.
        if std > 0:
            train_eps = np.random.normal(loc=0.0, scale=std, size=(num_train, 1))
            train_y += train_eps

        # Create validation data if requested.
        if num_val is not None:
            val_x = np.linspace(start=val_inter[0], stop=val_inter[1],
                                num=num_val).reshape((num_val, 1))
            val_y = map_function(val_x)

            in_data = np.vstack([train_x, test_x, val_x])
            out_data = np.vstack([train_y, test_y, val_y])
        else:
            in_data = np.vstack([train_x, test_x])
            out_data = np.vstack([train_y, test_y])

        # Specify internal data structure.
        self._data['classification'] = False
        self._data['sequence'] = False
        self._data['in_data'] = in_data
        self._data['in_shape'] = [1]
        self._data['out_data'] = out_data
        self._data['out_shape'] = [1]
        self._data['train_inds'] = np.arange(num_train)
        self._data['test_inds'] = np.arange(num_train, num_train + num_test)

        if num_val is not None:
            n_start = num_train + num_test
            self._data['val_inds'] = np.arange(n_start, n_start + num_val)

        self._map = map_function
        self._train_inter = train_inter
        self._test_inter = test_inter
        self._val_inter = val_inter

    @property
    def train_x_range(self):
        """Getter for read-only attribute train_x_range."""
        return self._train_inter

    @property
    def test_x_range(self):
        """Getter for read-only attribute test_x_range."""
        return self._test_inter

    @property
    def val_x_range(self):
        """Getter for read-only attribute val_x_range."""
        return self._val_inter

    def _get_function_vals(self, num_samples=100, x_range=None):
        """Get real function values for x values in a range that
        covers the test and training data. These values can be used to plot the
        ground truth function.

        Args:
            num_samples: Number of samples to be produced.
            x_range: If a specific range should be used to gather function
                values.

        Returns:
            x, y: Two numpy arrays containing the corresponding x and y values.
        """
        if x_range is None:
            min_x = min(self._train_inter[0], self._test_inter[0])
            max_x = max(self._train_inter[1], self._test_inter[1])
            if self.num_val_samples > 0:
                min_x = min(min_x, self._val_inter[0])
                max_x = max(max_x, self._val_inter[1])
        else:
            min_x = x_range[0]
            max_x = x_range[1]

        slack_x = 0.05 * (max_x - min_x)

        sample_x = np.linspace(start=min_x-slack_x, stop=max_x+slack_x,
                               num=num_samples).reshape((num_samples, 1))
        sample_y = self._map(sample_x)

        return sample_x, sample_y

    def plot_dataset(self, show=True):
        """Plot the whole dataset.

        Args:
            show: Whether the plot should be shown.
        """

        train_x = self.get_train_inputs().squeeze()
        train_y = self.get_train_outputs().squeeze()

        test_x = self.get_test_inputs().squeeze()
        test_y = self.get_test_outputs().squeeze()

        if self.num_val_samples > 0:
            val_x = self.get_val_inputs().squeeze()
            val_y = self.get_val_outputs().squeeze()

        sample_x, sample_y = self._get_function_vals()

        # The default matplotlib setting is usually too high for most plots.
        plt.locator_params(axis='y', nbins=2)
        plt.locator_params(axis='x', nbins=6)

        plt.plot(sample_x, sample_y, color='k', label='f(x)',
                 linestyle='dashed', linewidth=.5)
        plt.scatter(train_x, train_y, color='r', label='Train')
        plt.scatter(test_x, test_y, color='b', label='Test', alpha=0.8)
        if self.num_val_samples > 0:
            plt.scatter(val_x, val_y, color='g', label='Val', alpha=0.5)
        plt.legend()
        plt.title('1D-Regression Dataset')
        plt.xlabel('$x$')
        plt.ylabel('$y$')

        if show:
            plt.show()


    def get_identifier(self):
        """Returns the name of the dataset."""
        return '1DRegression'

    def _plot_sample(self, fig, inner_grid, num_inner_plots, ind, inputs,
                     outputs=None, predictions=None):

        raise NotImplementedError('TODO implement')

if __name__ == '__main__':
    pass


