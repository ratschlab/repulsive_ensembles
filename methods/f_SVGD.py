import torch.autograd as autograd
import torch
""" 
In this file a lot of different SVGD implementations are collected, the basic structure of the class is the same of the 
standard SVGD.
"""
import time 
class Timer(object):
    def __init__(self, name=None, print_t = False):
        self.name = name
        self.print_t = print_t

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.print_t:
            print('Elapsed '+self.name+': %s' % (time.time() - self.tstart))

class f_s_SVGD:
    """
    Implementation of functional space SVGD

    Args:
        P: instance of a distribution returning the log_prob, see distributions.py for examples
        K: kernel instance, see kernel.py for examples
        optimizer: instance of an optimizer SGD,Adam
    """
    def __init__(self, P, K, optimizer,prior_grad_estim,config,ann_sch,pred_idx = 1,num_train = False, noise = False):
        self.P = P
        self.K = K
        self.optim = optimizer
        self.pge = prior_grad_estim
        self.gamma = config.gamma
        self.ann_schedule=ann_sch
        self.pred_idx = pred_idx
        self.num_train = num_train
        self.noise = noise


    def phi(self, W,X,T,step,X_add=None):
        """
        Computes the update of the f-SVGD rule as: 
            

        Args:
            W: particles
            X: input training batch
            T:  label training batch

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

        # Score function
        log_prob, pred = self.P.log_prob(W, X, T, return_pred=True,pred_idx=self.pred_idx)
        score_func = autograd.grad(log_prob.sum(), pred[self.pred_idx], retain_graph=True)[0]

        if X_add is not None:
            pred_add = (self.P.ensemble.forward(X_add,W)[self.pred_idx]).view(W.shape[0],-1)
        else:
            pred_add = pred[self.pred_idx].view(W.shape[0],-1) #[n_part, classesxB]
    

        ############## Repulsive force ##############
        
        with Timer('Repulsive force:'): 
            pred_k = pred_add
            K_f = self.K(pred_k, pred_k.detach())
            grad_K = -autograd.grad(K_f.sum(), pred_k)[0]

        grad_K = grad_K.view(W.shape[0],-1) #needed only for weird kernels
        score_func = score_func.view(W.shape[0],-1)
        #pred = pred[0].view(W.shape[0],-1) #[n_part, classesxB]

        ############## Gradient functional prior ##############
        
        with Timer('Gradient prior:'):
            #pred = pred[self.pred_idx].view(W.shape[0],-1) #[n_part, classesxB]
            #pred_j = pred[self.pred_idx].view(W.shape[0],-1) #[n_part, classesxB]

            pred = pred[self.pred_idx].view(W.shape[0],-1) #[n_part, classesxB]

            
            w_prior = self.P.prior.sample(torch.Size([W.shape[0]]))

            prior_pred = self.P.ensemble.forward(X, w_prior)[self.pred_idx].reshape(W.shape[0],-1) # changed index here

            grad_prior = self.pge.compute_score_gradients(pred, prior_pred)  # .mean(0)
            
        ############## Update rule ##############
        
        with Timer('Driving force:'): 
            driv = K_f.matmul(score_func + grad_prior)
            
            
        if self.noise: 
            lr = self.optim.state_dict()['param_groups'][0]['lr']
            K_W_exp =  torch.sqrt(2*K_f.repeat(pred.shape[1],1,1)/(W.size(0)*lr))  #K_XX.repeat(X.shape[1],1,1)*2
            langevin_noise = torch.randn_like(K_W_exp)*K_W_exp
            f_phi = (self.ann_schedule[step]*driv + num_t*grad_K) / W.size(0) + langevin_noise.sum(2).T
        else:
            f_phi = (self.ann_schedule[step]*driv + num_t*grad_K) / W.size(0)
        #f_phi = score_func + grad_prior
        with Timer('function to weight + Jacobian :'):
            w_phi = autograd.grad(pred,W,grad_outputs=f_phi,retain_graph=False)[0]
            #w_phi = autograd.grad(pred,W,grad_outputs=f_phi,retain_graph=False)[0]

        #with Timer('function to weight :'): 
        #    w_phi = torch.einsum('mbw,mb->mw', [jacob,f_phi])
        return w_phi, self.ann_schedule[step]*driv, num_t*grad_K

    def step(self, W,X,T,step,X_add=None):
        """
        Customization of the optimizer step where I am forcing the gradient to be the SVGD update rule

        Args:
            W: particles
            X: input training batch
            T:  label training batch
        Return:
            driving force: first term in the update rule
            repulsive force: second term in the update rule
        """
        self.optim.zero_grad()
        update = self.phi(W,X,T,step,X_add)
        W.grad = -update[0]
        #torch.nn.utils.clip_grad_norm_(W,0.1,2)
        self.optim.step()
        return update[1], update[2]