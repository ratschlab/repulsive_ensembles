from data.dataset import  Dataset
from sklearn import datasets
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from data.dataset import  Dataset
from sklearn import datasets
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import math
import random
import torch
import itertools
from sklearn.model_selection import train_test_split

class Donuts(Dataset):
    """An instance of this class shall represent the two moons classification task
    """

    def __init__(self,r_1, r_2, c_outer_1, c_outer_2, c_inner_1=None, c_inner_2=None,  rseed=1234,  use_one_hot=True, noise=0.1, n_train=100, n_test=80):

        """Generate a new dataset.

        The input data x for train and test samples are drawn from the moons dataset.
        Args:
            rseed (int):
            noise: Noise to inject in the input data
            n_train: Number of training samples.
            n_test: Number of test samples..

        """
        super().__init__()

        if rseed is None:
            rand = np.random
        else:
            rand = np.random.RandomState(rseed)
            random.seed(rseed)

        donut_1 = self.sample(r_outer=r_1[0], r_inner=r_1[1], x_outer=c_outer_1[0], y_outer=c_outer_1[1], size=n_train)
        donut_2 = self.sample(r_outer=r_2[0], r_inner=r_2[1],  x_outer=c_outer_2[0], y_outer=c_outer_2[1], size=n_train)

        X_train = torch.tensor(np.vstack([donut_1, donut_2]), dtype=torch.float)

        class_labels_train = np.asarray(
            list(itertools.chain.from_iterable(itertools.repeat(x, int(X_train.shape[0] / 2)) for x in range(0, 2))))

        train_x, test_x, train_y, test_y = train_test_split(X_train, class_labels_train, test_size=0.33,
                                                            random_state=42)
        in_data = np.vstack([train_x, test_x])
        out_data = np.vstack([np.expand_dims(train_y, 1), np.expand_dims(test_y, 1)])

        # Specify internal data structure.
        # self._r_outer = r_outer
        # self._r_inner = r_inner
        # self._x_outer = x_outer
        # self._y_outer = y_outer
        # self._x_inner = x_inner
        # self._y_inner = y_inner
        self._data['classification'] = True
        self._data['sequence'] = False
        self._data['in_data'] = in_data
        self._data['in_shape'] = [2]
        self._data['num_classes'] = 2
        if use_one_hot:
            out_data = self._to_one_hot(out_data)
        self._data['out_data'] = out_data
        self._data['out_shape'] = [2]
        self._data['train_inds'] = np.arange(train_x.shape[0])
        self._data['test_inds'] = np.arange(train_x.shape[0], train_x.shape[0] + test_x.shape[0])

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'Donuts_dataset'


    def sample(self,r_outer, r_inner, x_outer, y_outer, x_inner=None, y_inner=None,size = 1):
        """
        Sample uniformly from (x, y) satisfiying:

           x**2 + y**2 <= r_outer**2

           (x-x_inner)**2 + (y-y_inner)**2 > r_inner**2

        Assumes that the inner circle lies inside the outer circle;
        i.e., that hypot(x_inner, y_inner) <= r_outer - r_inner.
        """
        # Sample from a normal annulus with radii r_inner and r_outer.
        if x_inner is None:
            x_inner = x_outer
            y_inner = y_outer

        l = []
        for _ in range(size):
            rad = math.sqrt(random.uniform(r_inner ** 2, r_outer ** 2))
            angle = random.uniform(-math.pi, math.pi)
            x, y = rad * math.cos(angle) + x_outer, rad * math.sin(angle) + y_outer

            # If we're inside the forbidden hole, reflect.
            if math.hypot(x - x_inner, y - y_inner) < r_inner:
                x, y = x_inner - x, y_inner - y
            l.append([x, y])

        return np.asarray(l)

    def get_input_mesh(self, x1_range=None, x2_range=None, grid_size=1000):
        """Create a 2D grid of input values to make useful plots like

        Note:
            This method is only implemented for 2D datasets.

        Args:
            x1_range (tuple, optional): The min and max value for the first
                input dimension. If not specified, the range will be
                automatically inferred.

                Automatical inference is based on the underlying data (train
                and test). The range will be set, such that all data can be
                drawn inside.
            x2_range (tuple, optional): Same as ``x1_range`` for the second
                input dimension.
            grid_size (int or tuple): How many input samples per dimension.
                If an integer is passed, then the same number grid size will be
                used for both dimension. The grid is build by equally spacing
                ``grid_size`` inside the ranges ``x1_range`` and ``x2_range``.

        Returns:
            (tuple): Tuple containing:

            - **x1_grid** (numpy.ndarray): A 2D array, containing the grid
              values of the first dimension.
            - **x2_grid** (numpy.ndarray): A 2D array, containing the grid
              values of the second dimension.
            - **flattended_grid** (numpy.ndarray): A 2D array, containing all
              samples from the first dimension in the first column and all
              values corresponding to the second dimension in the second column.
              This format correspond to the input format as, for instance,
              returned by methods such as
              :meth:`data.dataset.Dataset.get_train_inputs`.
        """
        if self.in_shape[0] != 2:
            raise ValueError('This method only applies to 2D datasets.')

        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)
        else:
            assert len(grid_size) == 2

        if x1_range is None or x2_range is None:
            min_x1 = self._data['in_data'][:, 0].min()
            min_x2 = self._data['in_data'][:, 1].min()
            max_x1 = self._data['in_data'][:, 0].max()
            max_x2 = self._data['in_data'][:, 1].max()

            slack_1 = (max_x1 - min_x1) * 0.02
            slack_2 = (max_x2 - min_x2) * 0.02

            if x1_range is None:
                x1_range = (min_x1 - slack_1, max_x1 + slack_1)
            else:
                assert len(x1_range) == 2

            if x2_range is None:
                x2_range = (min_x2 - slack_2, max_x2 + slack_2)
            else:
                assert len(x2_range) == 2

        x1 = np.linspace(start=x1_range[0], stop=x1_range[1], num=grid_size[0])
        x2 = np.linspace(start=x2_range[0], stop=x2_range[1], num=grid_size[1])

        X1, X2 = np.meshgrid(x1, x2)
        X = np.vstack([X1.ravel(), X2.ravel()]).T

        return X1, X2, X

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




