import torch.nn as nn
import torch
import numpy as np
import math
import torch.nn.functional as F
from utils.utils import dnorm2

class Net(nn.Module):
    """
    Implementation of Fully connected neural network

    Args:
        layer_sizes(list): list containing the layer sizes
        classification(bool): if the net is used for a classification task
        act: activation function in the hidden layers
        out_act: activation function in the output layer, if None then linear
        bias(Bool): whether or not the net has biases
    """

    def __init__(self, layer_sizes, classification = False, act=F.sigmoid,d_logits = False, out_act=None, bias=True, no_weights = True):
        super(Net, self).__init__()
        self.layer_sizes = layer_sizes
        self.layer_list = []
        self.classification = classification
        self.bias = bias
        self.d_logits = d_logits
        self.ac = act
        self.out_act = out_act
        for l in range(len(layer_sizes[:-1])):
            layer_l = nn.Linear(layer_sizes[l], layer_sizes[l+1], bias=self.bias)
            self.add_module('layer_' + str(l), layer_l)

        self.num_params = sum(p.numel() for p in self.parameters())

        self.param_shapes = [list(i.shape) for i in self.parameters()]

        self._weights = None
        if no_weights:
            return

        ### Define and initialize network weights.
        # Each odd entry of this list will contain a weight Tensor and each
        # even entry a bias vector.
        self._weights = nn.ParameterList()

        for i, dims in enumerate(self.param_shapes):
            self._weights.append(nn.Parameter(torch.Tensor(*dims),
                                              requires_grad=True))


        for i in range(0, len(self._weights), 2 if bias else 1):
            if bias:
                self.init_params(self._weights[i], self._weights[i + 1])
            else:
                self.init_params(self._weights[i])


    def init_params(self,weights, bias=None):
        """Initialize the weights and biases of a linear or (transpose) conv layer.

        Note, the implementation is based on the method "reset_parameters()",
        that defines the original PyTorch initialization for a linear or
        convolutional layer, resp. The implementations can be found here:

            https://git.io/fhnxV
        Args:
            weights: The weight tensor to be initialized.
            bias (optional): The bias tensor to be initialized.
        """
        nn.init.kaiming_uniform_(weights, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)


    def forward(self, x, weights=None, ret_pre = False):
        """Can be used to make the forward step and make predictions.

        Args:
            x(torch tensor): The input batch to feed the network.
            weights(list): A reshaped particle
        Returns:
            (tuple): Tuple containing:

            - **y**: The output of the network
            - **hidden** (optional): if out_act is not None also the linear output before activation is returned
        """

        if weights is None:
            weights = self._weights
        else:
            shapes = self.param_shapes
            assert (len(weights) == len(shapes))
            for i, s in enumerate(shapes):
                assert (np.all(np.equal(s, list(weights[i].shape))))

        hidden = x

        if self.bias:
            num_layers = len(weights) // 2
            step_size = 2
        else:
            num_layers = len(weights)
            step_size = 1

        for l in range(0, len(weights), step_size):
            W = weights[l]
            if self.bias:
                b = weights[l + 1]
            else:
                b = None

            if l==len(weights)-2 and self.d_logits:
                pre_out = hidden
                distance_logits = dnorm2(pre_out, W)

            hidden = F.linear(hidden, W, bias=b)

            # Only for hidden layers.
            if l / step_size + 1 < num_layers:
                if self.ac is not None:
                    hidden = self.ac(hidden)

        if self.d_logits:
            hidden = -distance_logits
        if self.out_act is not None:
            return self.out_act(hidden), hidden #needed so that i can use second output for training first for predict
        else:
            return hidden