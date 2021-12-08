from data.dataset import  Dataset
from sklearn import datasets
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import torch
import numpy as np
import sklearn

class oned_gaussian(Dataset):
    """An instance of this class shall represent a regression task where the
    input samples :math:`x` are drawn from a Gaussian with given mean and
    variance.

    Due to plotting functionalities, this class only supports 2D inputs and
    1D outputs.

    Attributes:
        mean: Mean vector.
        cov: Covariance matrix.
    """

    def __init__(self, rseed=1234, use_one_hot=True, noise=0.1, n_points = 300):
        """Generate a new dataset.

        The input data x for train and test samples will be drawn iid from the
        given Gaussian. Per default, the map function is the probability
        density of the given Gaussian: y = f(x) = p(x).

        Args:
            mean: The mean of the Gaussian.
            cov: The covariance of the Gaussian.
            num_train: Number of training samples.
            num_test: Number of test samples.
            map_function (optional): A function handle that receives input
                samples and maps them to output samples. If not specified, the
                density function will be used as map function.
            rseed (int): If ``None``, the current random state of numpy is used
                to generate the data. Otherwise, a new random state with the
                given seed is generated.
        """
        super().__init__()

        if rseed is None:
            rand = np.random
        else:
            rand = np.random.RandomState(rseed)

        g_1 = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        g_2 = torch.distributions.normal.Normal(torch.tensor([10.0]), torch.tensor([1.0]))

        x = torch.cat([g_1.sample(torch.Size([n_points])), g_2.sample(torch.Size([n_points]))])
        labels = np.repeat([0,1],n_points)

        train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(x, labels, test_size = 0.33, shuffle=True, random_state = 42)

        in_data = np.vstack([train_x, test_x])
        out_data = np.vstack([np.expand_dims(train_y, 1), np.expand_dims(test_y, 1)])

        # Specify internal data structure.
        self._data['classification'] = True
        self._data['sequence'] = False
        self._data['in_data'] = in_data
        self._data['in_shape'] = [1]
        self._data['num_classes'] = 2
        if use_one_hot:
            out_data = self._to_one_hot(out_data)
        self._data['out_data'] = out_data
        self._data['out_shape'] = [2]
        self._data['train_inds'] = np.arange(train_x.shape[0])
        self._data['test_inds'] = np.arange(train_x.shape[0], train_x.shape[0] + test_x.shape[0])

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'Moons_dataset'

    def get_input_mesh(self, x1_range=None, grid_size=1000):

        x1 = np.linspace(start=x1_range[0], stop=x1_range[1], num=grid_size)


        return x1

    def _plot_sample(self, fig, inner_grid, num_inner_plots, ind, inputs,
                     outputs=None, predictions=None):
        colors = ListedColormap(['#FF0000', '#0000FF'])

        # Create plot
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111)

        x_train_0 = self.get_train_inputs()
        y_train_0 = self.get_train_outputs()
        x_test_0 = self.get_test_inputs()
        y_test_0 = self.get_test_outputs()

        ax.scatter(x_train_0[:, 0], x_train_0[:, 1], alpha=1, marker='o', c=np.argmax(y_train_0, 1), cmap=colors,
                   edgecolors='k', s=50, label='Train')
        ax.scatter(x_test_0[:, 0], x_test_0[:, 1], alpha=0.6, marker='s', c=np.argmax(y_test_0, 1), cmap=colors,
                   edgecolors='k', s=50, label='test')
        plt.title('Data', fontsize=30)
        plt.legend(loc=2, fontsize=30)
        plt.show()
