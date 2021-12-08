import numpy as np
import torch

class Ensemble():
    """Implementation of an ensemble of models

    This is a simple class to manage and make predictions using an ensemble with or without particles
    Args:
        device: Torch device (cpu or gpu).
        net: pytorch model to create the ensemble
        particles(Tensor): Tensor (n_particles, n_params) containing squeezed parameter value of the specified model,
            if None, particles will be sample from a gaussian N(0,1)
        n_particles(int): if no particles are provided the ensemble is initialized and the number of members is required

    """

    def __init__(self, device, net, particles=None, n_particles=1):
        self.net = net
        if particles is None:
            self.particles = (1*torch.randn(n_particles, *torch.Size([self.net.num_params]))).to(device)
            #self.particles =torch.FloatTensor(n_particles, self.net.num_params).uniform_(-0.1, 0.1).to(device)
        else:
            self.particles = particles

        self.weighs_split = [np.prod(w) for w in net.param_shapes]

    def reshape_particles(self, z):
        reshaped_weights = []
        z_splitted = torch.split(z, self.weighs_split, 1)
        for j in range(z.shape[0]):
            l = []
            for i, shape in enumerate(self.net.param_shapes):
                l.append(z_splitted[i][j].reshape(shape))
            reshaped_weights.append(l)
        return reshaped_weights

    def forward(self, x, W=None):
        if W is None:
            W = self.particles
        models = self.reshape_particles(W)
        if self.net.out_act is None:
            pred = [self.net.forward(x, w) for w in models]
            return [torch.stack(pred)] #.unsqueeze(0)
        else:
            pred,hidden = zip(*(list(self.net.forward(x,w)) for w in models))
            return torch.stack(pred), torch.stack(hidden)