from math import floor

import torch

from utils.SSGE_squeeze import SpectralSteinEstimator
from f_SVGD import f_s_SVGD
from utils.kernel import RBF
from methods.SVGD import SGLD, SGD, SVGD, SVGLD
from methods.WGD import WGD, f_WGD


def create_method(config,P,optimizer,K=None, device = None):
    """
    Utils for the creation of the SVGD method
    """
    ann_sch = create_ann(config)
    pred_idx = config.logit_soft #1 logit, 0 softmax
    num_train = config.num_train
    
    if K is None: 
        K = RBF()

    if config.method == 'SGLD':
        method = SGLD(P, K, optimizer, device = device )
    elif config.method == 'SGD':
        method = SGD(P,optimizer)
    elif config.method == 'SVGD':
        K = RBF()
        method = SVGD(P,K,optimizer,config, ann_sch, num_train = num_train, noise = config.noise )
    elif config.method == 'SVGLD':
        K = RBF()
        method = SVGLD(P,K,optimizer,config,ann_sch, beta = 100)
    elif config.method == 'f_s_SVGD':
        ssge_k = RBF()
        ssge = SpectralSteinEstimator(0.01,None,ssge_k, device = device)
        method = f_s_SVGD(P, K, optimizer,ssge,config,ann_sch,pred_idx,num_train,noise = config.noise)
    elif config.method == 'kde_WGD':
        K = RBF()
        method = WGD(P,K,optimizer,config, ann_sch,grad_estim=None, num_train = num_train, method = 'kde' )
    elif config.method == 'sge_WGD':
        K = RBF()
        method = WGD(P,K,optimizer,config, ann_sch,grad_estim=None, num_train = num_train, method = 'sge', device = device )
    elif config.method == 'ssge_WGD':
        ssge_k = RBF()
        K = RBF()
        ssge = SpectralSteinEstimator(0.01,None,ssge_k, device = device)
        method = WGD(P, K, optimizer,config,ann_sch,grad_estim = ssge,  num_train = num_train, method = 'ssge') 
    elif config.method == 'kde_f_WGD':
        ssge_k = RBF()
        K = RBF()
        ssge = SpectralSteinEstimator(0.01,None,ssge_k, device = device)
        method = f_WGD(P, K, optimizer,config,ann_sch,grad_estim = ssge, pred_idx = pred_idx,num_train = num_train, method = 'kde')
    elif config.method == 'sge_f_WGD':
        ssge_k = RBF()
        K = RBF()
        ssge = SpectralSteinEstimator(0.01,None,ssge_k, device = device)
        method = f_WGD(P, K, optimizer,config,ann_sch,grad_estim = ssge, pred_idx = pred_idx,num_train = num_train, method = 'sge', device = device)
    elif config.method == 'ssge_f_WGD':
        ssge_k = RBF()
        K = RBF()
        ssge = SpectralSteinEstimator(0.01,None,ssge_k, device = device)
        method = f_WGD(P, K, optimizer,config,ann_sch,grad_estim = ssge, pred_idx = pred_idx,num_train = num_train, method = 'ssge')
        

    return method

# cosine annealing learning rate schedule
def cosine_annealing(epoch, n_epochs, n_cycles, lrate_max):
    epochs_per_cycle = floor(n_epochs/n_cycles)
    cos_inner =(epoch % epochs_per_cycle)/ (epochs_per_cycle)
    return (cos_inner)


def create_ann(config): 
    if config.ann_sch == 'linear': 
        ann_sch = torch.cat([torch.linspace(0,config.gamma,config.annealing_steps),config.gamma*torch.ones(config.epochs-config.annealing_steps)]) 
    elif config.ann_sch == 'hyper': 
        ann_sch =torch.cat([torch.tanh((torch.linspace(0,config.annealing_steps,config.annealing_steps)*1.3/config.annealing_steps)**10),config.gamma*torch.ones(config.epochs-config.annealing_steps)])
    elif config.ann_sch == 'cyclic':
        ann_sch = torch.cat([torch.tensor([cosine_annealing(a,config.annealing_steps,5,1)**10 for a in range(config.annealing_steps)]),config.gamma*torch.ones(config.epochs-config.annealing_steps)])
    elif config.ann_sch == 'None':
        ann_sch = config.gamma*torch.ones(config.epochs)
    return ann_sch