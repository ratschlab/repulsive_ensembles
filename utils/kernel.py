import torch
import math

class RBF(torch.nn.Module):
    def __init__(self, sigma=None):
        super(RBF, self).__init__()

        self.sigma = sigma

    def median(self, tensor):
        tensor = tensor.flatten().sort()[0]
        length = tensor.shape[0]

        if length % 2 == 0:
            szh = length // 2
            kth = [szh - 1, szh]
        else:
            kth = [(length - 1) // 2]
        return tensor[kth].mean()

    def forward(self, X, Y):
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.sigma is None:
            sigma = self.median(dnorm2.detach()) / (2 * torch.tensor(math.log(X.size(0) + 1)))
        else:
            sigma = self.sigma ** 2

        gamma = 1.0 / (2 * sigma)
        K_XY = (-gamma * dnorm2).exp()

        return K_XY