import torch
import torch.autograd as autograd

"""

The original implementation can be found here https://github.com/AntixK/Spectral-Stein-Gradient/blob/master/score_estimator/spectral_stein.py

"""


class SpectralSteinEstimator():
    def __init__(self, eta=None, num_eigs=None, K=None, xm = None, device = None):
        self.eta = eta
        self.num_eigs = num_eigs
        self.K = K
        self.xm = xm
        if xm is not None:
            self.beta, self.eigen_vals, self.eigen_vec = self.compute_beta(xm)
        self.device = device 

    def nystrom_method(self, x, eval_points, eigen_vecs, eigen_vals):
        """
        Implements the Nystrom method for approximating the
        eigenfunction (generalized eigenvectors) for the kernel
        at x using the M eval_points (x_m). It is given
        by -
         .. math::
            phi_j(x) = \frac{M}{\lambda_j} \sum_{m=1}^M u_{jm} k(x, x_m)
        :param x: (Tensor) Point at which the eigenfunction is evaluated [N x D]
        :param eval_points: (Tensor) Sample points from the data of ize M [M x D]
        :param eigen_vecs: (Tensor) Eigenvectors of the gram matrix [M x M]
        :param eigen_vals: (Tensor) Eigenvalues of the gram matrix [M x 2]
        :return: Eigenfunction at x [N x M]
        """
        M = torch.tensor(eval_points.size(-2), dtype=torch.float)

        Kxxm = self.K(x, eval_points)
        phi_x = torch.sqrt(M) * Kxxm @ eigen_vecs

        phi_x *= 1. / eigen_vals[:, 0]  # Take only the real part of the eigenvals
        # as the Im is 0 (Symmetric matrix)
        return phi_x

    def compute_beta(self,xm):

        M = torch.tensor(xm.size(-2), dtype=torch.float)

        xm = xm.detach().requires_grad_(True)

        Kxx = self.K(xm, xm.detach())

        dKxx_dx = autograd.grad(Kxx.sum(), xm)[0]

        # Kxx = Kxx + eta * I
        if self.eta is not None:
            Kxx += self.eta * torch.eye(xm.size(-2)).to(self.device)

        eigen_vals, eigen_vecs = torch.eig(Kxx, eigenvectors=True)

        if self.num_eigs is not None:
            eigen_vals = eigen_vals[:self.num_eigs]
            eigen_vecs = eigen_vecs[:, :self.num_eigs]



        # Compute the Monte Carlo estimate of the gradient of
        # the eigenfunction at x
        dKxx_dx_avg = -dKxx_dx/xm.shape[0] # [M x D]

        beta = - torch.sqrt(M) * eigen_vecs.t() @ dKxx_dx_avg
        beta *= (1. / eigen_vals[:, 0].unsqueeze(-1))

        return beta, eigen_vals, eigen_vecs

    def compute_score_gradients(self, x, xm = None):
        """
        Computes the Spectral Stein Gradient Estimate (SSGE) for the
        score function. The SSGE is given by
        .. math::
            \nabla_{xi} phi_j(x) = \frac{1}{\mu_j M} \sum_{m=1}^M \nabla_{xi}k(x,x^m) \phi_j(x^m)
            \beta_{ij} = -\frac{1}{M} \sum_{m=1}^M \nabla_{xi} phi_j (x^m)
            \g_i(x) = \sum_{j=1}^J \beta_{ij} \phi_j(x)
        :param x: (Tensor) Point at which the gradient is evaluated [N x D]
        :param xm: (Tensor) Samples for the kernel [M x D]
        :return: gradient estimate [N x D]
        """
        if xm is None:
            xm = self.xm
            beta = self.beta
            eigen_vecs = self.eigen_vecs
            eigen_vals = self.eigen_vals
        else:
            beta,eigen_vals,eigen_vecs = self.compute_beta(xm)

        phi_x = self.nystrom_method(x, xm, eigen_vecs, eigen_vals)  # [N x M]
        # assert beta.allclose(beta1), f"incorrect computation {beta - beta1}"
        g = phi_x @ beta  # [N x D]
        return g
