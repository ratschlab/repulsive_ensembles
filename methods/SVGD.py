import torch
import torch.autograd as autograd

""" 
In this file a lot of different SVGD implementations are collected, the basic structure of the class is the same of the 
standard SVGD.
"""

class SVGD:
    """
    Implementation of SVGD

    Args:
        P: instance of a distribution returning the log_prob, see distributions.py for examples
        K: kernel instance, see kernel.py for examples
        optimizer: instance of an optimizer SGD,Adam
    """
    def __init__(self, P, K, optimizer,config, ann_sch,num_train = False, noise = False):
        self.P = P
        self.K = K
        self.optim = optimizer
        self.gamma = config.gamma
        self.ann_schedule = ann_sch
        self.num_train = num_train
        self.noise = noise


    def phi(self, W,X,T,step):
        """
        Computes the update of the SVGD rule

        Args:
            W: particles
            X: inputs training batch
            T: labels training batch

        Return:
            phi: the update to feed the optimizer
            driving force: first term in the update rule
            repulsive force: second term in the update rule
        """
        
        if self.num_train: 
            num_t = self.P.num_train
        else: 
            num_t = 1
            
        W = W.detach().requires_grad_(True)
        
        #computing the driving force
        log_prob = self.P.log_prob(W,X,T)
        score_func = autograd.grad(log_prob.sum(), W)[0]
        
        #computing the repusive force
        K_W = self.K(W, W.detach())
        grad_K = -autograd.grad(K_W.sum(), W)[0]
        
        if self.noise: 
            lr = self.optim.state_dict()['param_groups'][0]['lr']
            K_W_exp =  torch.sqrt(2*K_W.repeat(W.shape[1],1,1)/(W.size(0)*lr))  #K_XX.repeat(X.shape[1],1,1)*2 
            langevin_noise = torch.randn_like(K_W_exp)*K_W_exp
            phi = (self.ann_schedule[step]*K_W.detach().matmul(score_func) +num_t*grad_K) / W.size(0) + langevin_noise.sum(2).T
        else:
            phi = (self.ann_schedule[step]*K_W.detach().matmul(score_func) +num_t*grad_K) / W.size(0)


        return phi, self.ann_schedule[step]*K_W.detach().matmul(score_func), num_t*grad_K

    def step(self, W,X,T,step):
        """
        Customization of the optimizer step where I am forcing the gradients to be instead the SVGD update rule

        Args:
            W: particles
            X: input training batch
            T:  label training batch
        Return:
            driving force: first term in the update rule
            repulsive force: second term in the update rule
        """
        self.optim.zero_grad()
        update = self.phi(W,X,T,step)
        W.grad = -update[0]
        self.optim.step()
        return update[1], update[2]

class SVGLD:
    def __init__(self, P, K, optimizer,config, ann_sch, beta = 1.0, alpha = 1.0):
        self.P = P
        self.K = K
        self.optim = optimizer
        self.gamma = config.gamma
        self.beta = beta
        self.alpha = alpha #useful to remove the first additional score
        self.ann_schedule=ann_sch
        

    def phi(self, W,X,T,step):
        W = W.detach().requires_grad_(True)

        log_prob = self.P.log_prob(W,X,T)
        score_func = autograd.grad(log_prob.sum(), W)[0]

        K_W = self.K(W, W.detach())
        grad_K = -autograd.grad(K_W.sum(), W)[0]
        
        driv = self.alpha/self.beta*score_func + K_W.detach().matmul(score_func)/W.size(0) 
        rep = self.P.num_train*grad_K / W.size(0)

        phi = self.ann_schedule[step]*driv + rep
        lr = self.optim.state_dict()['param_groups'][0]['lr']

        langevin_noise = torch.distributions.Normal(torch.zeros(W.shape[0]),torch.ones(W.shape[0])/torch.sqrt(self.beta*torch.tensor(lr)))
        noise = langevin_noise.sample().unsqueeze(1)
        phi += -noise

        return phi,self.ann_schedule[step]*driv,rep,-noise

    def step(self, W,X,T,step):
        self.optim.zero_grad()
        update = self.phi(W,X,T,step)
        W.grad = -update[0]
        self.optim.step()
        return update[1], update[2], update[3]

class SGD:
    def __init__(self, P, optimizer):
        self.P = P
        self.optim = optimizer

    def phi(self, W,X,T):
        W = W.detach().requires_grad_(True)

        log_prob = self.P.log_prob(W,X,T)
        score_func = autograd.grad(log_prob.sum(), W)[0]

        phi = score_func

        return phi

    def step(self, W,X,T):
        self.optim.zero_grad()
        W.grad = -self.phi(W,X,T)
        self.optim.step()


    def step(self, W,X,T,step,X_add = None):
        self.optim.zero_grad()
        update = self.phi(W,X,T,step, X_add)
        W.grad = -update[0]
        self.optim.step()
        return update[1], update[2]
    
class SGLD:
    def __init__(self, P, K, optimizer, device):
        self.P = P
        self.K = K
        self.optim = optimizer
        self.device = device

    def phi(self, W,X,T):
        W = W.detach().requires_grad_(True)

        log_prob = self.P.log_prob(W,X,T)
        score_func = autograd.grad(log_prob.sum(), W)[0]
        lr = self.optim.state_dict()['param_groups'][0]['lr']
        langevin_noise = torch.distributions.Normal(torch.zeros(W.shape[1]).to(self.device),(torch.ones(W.shape[1])/torch.sqrt(torch.tensor(lr))).to(self.device))
        phi = score_func + langevin_noise.sample(torch.Size([W.size(0)]))

        return phi

    def step(self, W,X,T):
        self.optim.zero_grad()
        W.grad = -self.phi(W,X,T)
        self.optim.step()
    