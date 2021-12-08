import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

class Unorm_post():
    """
    Implementation of unnormalized posterior for a neural network model. It assume gaussian likelihood with variance
    config.pred_dist_std. The prior can be freely specified, the only requirement is a log_prob method to return the
    log probability of the particles.

    Args:
        ensemble: ensemble instance from MLP.py
        prior: prior instance from torch.distributions or custom, .log_prob() method is required
        config: Command-line arguments.
        n_train: number of training datapoints to rescale the likelihood

    """
    def __init__(self, ensemble, prior, config, n_train,add_prior = True):
        self.prior = prior
        self.ensemble = ensemble
        self.config = config
        self.num_train = n_train
        self.add_prior = add_prior


    def log_prob(self, particles, X, T, return_loss=False, return_pred = False, pred_idx = 1):
        pred = self.ensemble.forward(X, particles)

        if self.ensemble.net.classification:
            if pred_idx == 1:
                #loss = -(T.expand_as(pred[1])*F.log_softmax(pred[1],2)).sum((1,2))/X.shape[0]
                #loss = (-(T.expand_as(pred[1])*F.log_softmax(pred[1],2))).max(2)[0].sum(1)/X.shape[0]
                loss = torch.stack([F.nll_loss(F.log_softmax(p), T.argmax(1)) for p in pred[1]])
            else:
                #loss = -(T.expand_as(pred[0]) * torch.log(pred[0]+1e-15)).sum((1, 2)) / X.shape[0]
                #loss = -(torch.log(pred[0]+1e-15)[T.expand_as(pred[0]).type(torch.ByteTensor)].reshape(pred[0].shape[:-1])).sum(1)/ X.shape[0]
                loss = (-(T.expand_as(pred[1])*torch.log(pred[0]+1e-15))).max(2)[0].sum(1)/X.shape[0]

            #pred = F.softmax(pred[1],2) #I have to do this to allow derivative and to not have nans 
        else:
            #loss = 0.5*torch.mean(F.mse_loss(prebpd[0], T, reduction='none'), 1)
            loss = 0.5*torch.mean((T.expand_as(pred[0])-pred[0])**2,1)


        ll = -loss*self.num_train / self.config.pred_dist_std ** 2

        if particles is None:
            particles = self.ensemble.particles

        if self.add_prior:
            log_prob = torch.add(self.prior.log_prob(particles).sum(1), ll)
        else:
            log_prob = ll
#        log_prob = ll
        if return_loss:
            return torch.mean(loss),pred[0]
        elif return_pred:
            return log_prob,pred #0 softmax, 1 is logit
        else:
            return log_prob

class Unorm_post_hyper():
    """
    Implementation of unnormalized posterior for a neural network model. It assume gaussian likelihood with variance
    config.pred_dist_std. The prior can be freely specified, the only requirement is a log_prob method to return the
    log probability of the particles.

    Args:
        ensemble: ensemble instance from MLP.py
        prior: prior instance from torch.distributions or custom, .log_prob() method is required
        config: Command-line arguments.
        n_train: number of training datapoints to rescale the likelihood

    """
    def __init__(self, ensemble, prior, config, n_train,add_prior = True):
        self.priors = prior
        self.ensemble = ensemble
        self.config = config
        self.num_train = n_train
        self.add_prior = add_prior


    def log_prob(self, particles, X, T, return_loss=False, return_pred = False, pred_idx = 1):
        pred = self.ensemble.forward(X, particles)

        if self.ensemble.net.classification:
            if pred_idx == 1:
                #loss = -(T.expand_as(pred[1])*F.log_softmax(pred[1],2)).sum((1,2))/X.shape[0]
                #loss = (-(T.expand_as(pred[1])*F.log_softmax(pred[1],2))).max(2)[0].sum(1)/X.shape[0]
                loss = torch.stack([F.nll_loss(F.log_softmax(p), T.argmax(1)) for p in pred[1]])
            else:
                #loss = -(T.expand_as(pred[0]) * torch.log(pred[0]+1e-15)).sum((1, 2)) / X.shape[0]
                #loss = -(torch.log(pred[0]+1e-15)[T.expand_as(pred[0]).type(torch.ByteTensor)].reshape(pred[0].shape[:-1])).sum(1)/ X.shape[0]
                loss = (-(T.expand_as(pred[1])*torch.log(pred[0]+1e-15))).max(2)[0].sum(1)/X.shape[0]

            #pred = F.softmax(pred[1],2) #I have to do this to allow derivative and to not have nans 
        else:
            #loss = 0.5*torch.mean(F.mse_loss(prebpd[0], T, reduction='none'), 1)
            loss = 0.5*torch.mean((T.expand_as(pred[0])-pred[0])**2,1)


        ll = -loss*self.num_train / self.config.pred_dist_std ** 2

        if particles is None:
            particles = self.ensemble.particles

        log_priors = []
        for ind,p in enumerate(particles):
             log_priors.append(self.priors[ind].log_prob(p).sum())

        log_prob = torch.add(torch.stack(log_priors), ll)

#        log_prob = ll
        if return_loss:
            return torch.mean(loss),pred[0]
        elif return_pred:
            return log_prob,pred #0 softmax, 1 is logit
        else:
            return log_prob


class Gaus_mix_multi:

    def __init__(self, mu_1=-3., mu_2=3., sigma_1=1., sigma_2=1.):
        self.prior_1 = MultivariateNormal(torch.tensor([7., 0.]), sigma_1 * torch.eye(2))
        self.prior_2 = MultivariateNormal(torch.tensor([-7., 0.]), sigma_2 * torch.eye(2))
        self.prior_3 = MultivariateNormal(torch.tensor([0., 7.]), sigma_1 * torch.eye(2))
        self.prior_4 = MultivariateNormal(torch.tensor([0., -7.]), sigma_1 * torch.eye(2))

    def log_prob(self, z):
        log_prob = torch.log(
            0.25 * torch.exp(self.prior_1.log_prob(z)) + 0.25 * torch.exp(self.prior_2.log_prob(z)) + 0.25 * torch.exp(
                self.prior_3.log_prob(z)) + 0.25 * torch.exp(self.prior_4.log_prob(z)))
        return log_prob

    def sample(self, n_samples):
        s = []
        for i in range(n_samples):
            a = np.random.uniform()
            if a < 0.25:
                s.append(self.prior_1.sample(torch.Size([1])).detach().numpy()[0])
            elif a > 0.25 and a < 0.5:
                s.append(self.prior_2.sample(torch.Size([1])).detach().numpy()[0])
            elif a > 0.5 and a < 0.75:
                s.append(self.prior_3.sample(torch.Size([1])).detach().numpy()[0])
            else:
                s.append(self.prior_4.sample(torch.Size([1])).detach().numpy()[0])
        return np.stack(s)


class Gaus_mix_multi_2:

    def __init__(self, mu_1=-5., mu_2=3., sigma_1=1., sigma_2=1.):
        self.prior_1 = MultivariateNormal(mu_1 * torch.ones(2), sigma_1 * torch.eye(2))
        self.prior_2 = MultivariateNormal(mu_2 * torch.ones(2), sigma_2 * torch.eye(2))

    def log_prob(self, z):
        log_prob = torch.log(0.5 * torch.exp(self.prior_1.log_prob(z)) + 0.5 * torch.exp(self.prior_2.log_prob(z)))
        return log_prob

    def sample(self, n_samples):
        s = []
        for i in range(n_samples):
            a = np.random.uniform()
            if a < 0.5:
                s.append(self.prior_1.sample(torch.Size([1])).detach().numpy()[0])
            else:
                s.append(self.prior_2.sample(torch.Size([1])).detach().numpy()[0])
        return np.stack(s)