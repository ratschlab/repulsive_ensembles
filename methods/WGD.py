import torch.autograd as autograd
import torch

""" 
In this file WGD implementations in weights and function space are collected.
"""
   
    
class WGD:
    """
    Implementation of WGD in weight space with KDE, SGE, SSGE approximation 

    Args:
        P: instance of a distribution returning the log_prob, see distributions.py for examples
        K: kernel instance, see kernel.py for examples
        optimizer: instance of an optimizer SGD,Adam
    """
    def __init__(self, P, K, optimizer,config,ann_sch,grad_estim=None,num_train = False, method = 'kde', device = None):
        self.P = P
        self.K = K
        self.optim = optimizer
        self.pge = grad_estim
        self.gamma = config.gamma
        self.ann_schedule=ann_sch
        self.num_train = num_train
        self.method = method
        self.device = device


    def phi(self, W,X,T,step):
        """
        Computes the update of the WGD rule

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
        
        if self.method == 'kde':
            K_W = self.K(W, W.detach())
            grad_K = autograd.grad(K_W.sum(), W)[0] 
            grad_density = grad_K/ K_W.sum(1,keepdim = True)
            
        elif self.method == 'ssge':
            grad_density = self.pge.compute_score_gradients(W, W)
            
        elif self.method == 'sge':
            eta = 0.01
            K_W = self.K(W, W.detach())
            grad_K = autograd.grad(K_W.sum(), W)[0] 
            K_ = K_W+eta*torch.eye(K_W.shape[0]).to(self.device)  
            grad_density = torch.linalg.solve(K_,grad_K)
        
        phi = ( self.ann_schedule[step]*score_func-grad_density) 


        return phi, self.ann_schedule[step]*score_func, grad_density

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
    
    
class f_WGD:
    """
    Implementation of WGD in weight space with KDE, SGE, SSGE approximation 

    Args:
        P: instance of a distribution returning the log_prob, see distributions.py for examples
        K: kernel instance, see kernel.py for examples
        optimizer: instance of an optimizer SGD,Adam
        prior_grad_estim: SSGE estimator for functional prior 
        config: configurator 
        ann_sch: list of annealing steps 
        pred_idx: if 1 the function space inference is on logits if 0 on softmax
        num_train: if True the repulsive force is multiplied by the number of datapoints
    """
    def __init__(self, P, K, optimizer,config,ann_sch,grad_estim=None,pred_idx = 1,num_train = False, method='kde', device = None):
        self.P = P
        self.K = K
        self.optim = optimizer
        self.pge = grad_estim
        self.gamma = config.gamma
        self.ann_schedule=ann_sch
        self.pred_idx = pred_idx
        self.num_train = num_train
        self.method = method
        self.device = device


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

        ######### Score function #########
        log_prob, pred = self.P.log_prob(W, X, T, return_pred=True,pred_idx=self.pred_idx)
        score_func = autograd.grad(log_prob.sum(), pred[self.pred_idx], retain_graph=True)[0]

        # Needed in case we want to compute the repulsion on additional points
        if X_add is not None:
            pred_add = (self.P.ensemble.forward(X_add,W)[self.pred_idx]).view(W.shape[0],-1)
        else:
            pred_add = pred[self.pred_idx].view(W.shape[0],-1) #[n_part, classesxB]
    

        ######### Repulsive force #########
        pred_k = pred_add
        score_func = score_func.view(W.shape[0],-1)
        #pred = pred[0].view(W.shape[0],-1) #[n_part, classesxB]

        ######### Gradient functional prior #########

        pred = pred[self.pred_idx].view(W.shape[0],-1) #[n_particles, classes x Batch]


        w_prior = self.P.prior.sample(torch.Size([W.shape[0]]))

        prior_pred = self.P.ensemble.forward(X, w_prior)[self.pred_idx].reshape(W.shape[0],-1) # changed index here

        grad_prior = self.pge.compute_score_gradients(pred, prior_pred)  # .mean(0)1
            
        ######### Update rule #########
        driv = score_func + grad_prior
        
        if self.method == 'kde':
            K_f = self.K(pred_k, pred_k.detach())
            grad_K = autograd.grad(K_f.sum(), pred_k)[0]
            grad_K = grad_K.view(W.shape[0],-1) 
            
            grad_density = grad_K/ K_f.sum(1,keepdim = True)
            
        elif self.method == 'ssge':
            grad_density = self.pge.compute_score_gradients(pred, pred)
        
        elif self.method == 'sge':
            eta = 0.01
            K_f = self.K(pred_k, pred_k.detach())
            grad_K = autograd.grad(K_f.sum(), pred_k)[0]
            grad_K = grad_K.view(W.shape[0],-1) 
            K_ = K_f+eta*torch.eye(K_f.shape[0]).to(self.device) 
            grad_density = torch.linalg.solve(K_,grad_K)
        
        f_phi = (self.ann_schedule[step]*driv - grad_density)
        #f_phi = driv/W.size(0)

        w_phi = autograd.grad(pred,W,grad_outputs=f_phi,retain_graph=False)[0]
        #w_phi = autograd.grad(pred,W,grad_outputs=f_phi,retain_graph=False)[0]

        return w_phi, self.ann_schedule[step]*driv, -grad_density

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