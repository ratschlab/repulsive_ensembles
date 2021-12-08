import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import pairwise_distances

from data.toy_classification.donuts import Donuts
from utils.distributions import Unorm_post
from methods.method_utils import create_method
from utils.utils import contour_plot
from utils.utils import ood_metrics_entropy

# import uncertainty_metrics as um
warnings.filterwarnings("ignore")
sns.set()
sns.set(style="whitegrid")

def train(data, ensemble, device, config,writer):
    """Train the particles using a specific ensemble.

    Args:
        data: A DATASET instance.
        mnet: The model of the main network.
        device: Torch device (cpu or gpu).
        config: The command line arguments.
        writer: The tensorboard summary writer.s
    """

# --------------------------------------------------------------------------------
# CREATING TESTING AND OOD DATASET 
# --------------------------------------------------------------------------------
    if config.dataset == 'moons':
        grid_mesh = data.get_input_mesh(x1_range=(-10, 10), x2_range=(-10, 10), grid_size=300)

        x_test_0 = torch.tensor(data.get_test_inputs(), dtype=torch.float).to(device)
        y_test_0 = torch.tensor(data.get_test_outputs(), dtype=torch.float).to(device)
        test_labels = torch.argmax(y_test_0, 1).type(torch.int).to(device)


        donuts = Donuts(r_1 = (9,10),r_2 = (3,4), c_outer_1 = (0,0), c_outer_2 = (0,0))
        ext_ring_ood = donuts.sample(9, 4, 0, 0, size=200)
        x_test_ood = torch.tensor(ext_ring_ood, dtype=torch.float).to(device)
    
    elif config.dataset == 'twod_gaussian':
        grid_mesh = data.get_input_mesh(x1_range=(-20, 20), x2_range=(-20, 20), grid_size=300)
    
        x_test_0 = torch.tensor(data.get_test_inputs(), dtype=torch.float).to(device)
        y_test_0 = torch.tensor(data.get_test_outputs(), dtype=torch.float).to(device)
        test_labels = torch.argmax(y_test_0, 1).type(torch.int).to(device)


        donuts = Donuts(r_1 = (9,10),r_2 = (3,4), c_outer_1 = (0,0), c_outer_2 = (0,0))
        ext_ring_ood = donuts.sample(10 + 20, 9 + 2, 0, 0, size=200)
        p_in = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor((0., 0.)), 0.4 * torch.eye(2))
        s_in = p_in.sample(torch.Size([50]))
        x_test_ood = torch.cat([s_in, torch.tensor(ext_ring_ood, dtype=torch.float), ]).to(device)
    
    elif config.dataset == 'donuts':
        grid_mesh = data.get_input_mesh(x1_range=(-20, 20), x2_range=(-20, 20), grid_size=300)

        x_test_0 = torch.tensor(data.get_test_inputs(), dtype=torch.float).to(device)
        y_test_0 = torch.tensor(data.get_test_outputs(), dtype=torch.float).to(device)
        test_labels = torch.argmax(y_test_0, 1).type(torch.int).to(device)
        ext_ring_ood = data.sample(10 + 20, 9 + 2, 0, 0, size=200)
        p_in = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor((0., 0.)), 0.4 * torch.eye(2))
        s_in = p_in.sample(torch.Size([50]))
        x_test_ood = torch.cat([s_in, torch.tensor(ext_ring_ood, dtype=torch.float), ]).to(device)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)


    ax.scatter(x_test_0.cpu().numpy()[:, 0], x_test_0.cpu().numpy()[:, 1], alpha=0.7, marker='o', c=np.argmax(y_test_0.cpu().numpy(), 1), cmap="viridis",
               edgecolors='k', s=50, label='Test')

    ax.scatter(x_test_ood[:, 0].cpu().numpy(), x_test_ood.cpu().numpy()[:, 1], alpha=0.7, marker='o', c='red',
               edgecolors='k', s=50, label='OOD')
    writer.add_figure('DATA', plt.gcf(),
                      0, close=not config.show_plots)
# --------------------------------------------------------------------------------
# SETTING PROBABILISTIC and UTILS
# --------------------------------------------------------------------------------
    names = {'f_s_SVGD': 'f-SVGD',
             'mixed_f_p_SVGD':'h-SVGD',
             'f_p_SVGD':'fw-SVGD', 
             'ssge_WGD': 'ssge-WGD',
             'kde_WGD':'kde-WGD',
             'sge_WGD':'sge-WGD',
             'SGD':'Deep Ensemble', 
             'SGLD': 'pSGLD', 
             'SVGD':'w-SVGD',
             'ssge_f_WGD':'ssge-fWGD',
             'sge_f_WGD':'sge-fWGD',
             'kde_f_WGD':'kde-fWGD'}
    
    
    W = ensemble.particles
    samples = []
    if config.optim == 'Adam':
        optimizer = torch.optim.Adam([W], config.lr, weight_decay=config.weight_decay,
                                     betas=[config.adam_beta1, 0.999])
    else:
        optimizer = torch.optim.SGD([W], config.lr)
    prior = torch.distributions.normal.Normal(torch.zeros(ensemble.net.num_params).to(device),
                          torch.ones(ensemble.net.num_params).to(device) * config.prior_variance)
    if config.method == 'f_s_SVGD': 
        add_prior = False 
    else:
        add_prior = True

    
    #w_prior = prior.sample(torch.Size([config.n_particles]))


    #prior_pred = ensemble.forward(x_test_0[:4],w_prior)[0].reshape(config.n_particles,-1)

    #ssge_k = RBF(sigma = 1)

    #ssge = SpectralSteinEstimator(0.9,None,ssge_k,prior_pred)

    P = Unorm_post(ensemble, prior, config, data.num_train_samples,add_prior)
    #log_scale = torch.log2(torch.tensor(data_train.out_shape[0], dtype=torch.float))
    variance_noise = 0.25
    noise = torch.distributions.normal.Normal(torch.zeros(data.in_shape[0]).to(device),
                      torch.ones(data.in_shape[0]).to(device)*variance_noise)

# --------------------------------------------------------------------------------
# SVGD ALGORITHM SPECIFICATIONS
# --------------------------------------------------------------------------------

    method = create_method(config, P, optimizer, device = device)
    #K = RBF()
    #method = SVGD(P, ssge_k, optimizer,ssge, prior)

# --------------------------------------------------------------------------------
# SVGD TRAINING
# --------------------------------------------------------------------------------

    driving_l = []
    repulsive_l = []
    print('-------------------------'+'Start training'+'-------------------------')
    for i in range(config.epochs):
        optimizer.zero_grad()

        batch_train = data.next_train_batch(config.batch_size)
        batch_test = data.next_test_batch(config.batch_size)
        X = data.input_to_torch_tensor(batch_train[0], device, mode='train')
        T = data.output_to_torch_tensor(batch_train[1], device, mode='train')
        X_t = data.input_to_torch_tensor(batch_test[0], device, mode='train')
        T_t = data.output_to_torch_tensor(batch_test[1], device, mode='train')


        if config.method == 'SGD' or config.method == 'SGLD':
            method.step(W, X, T)
        elif config.method == 'f_p_SVGD' or config.method == 'mixed_f_p_SVGD' or config.method == 'f_s_SVGD':
            #noise_samples = noise.sample(torch.Size([config.batch_size]))
            if config.where_repulsive == 'train':
                driving,repulsive = method.step(W, X, T,i, None)
            elif config.where_repulsive == 'noise':
                blurred_train = X+noise.sample((X.shape[0],))
                driving,repulsive = method.step(W,X,T,i,blurred_train)
            elif config.where_repulsive == 'test':
                driving,repulsive = method.step(W,X,T,i,X_t)

            #driving,repulsive = method.step(W, X, T,i, None)
        elif config.method == 'SVGLD':
            driving,repulsive,langevin_noise = method.step(W, X, T,i)

        else:
            driving,repulsive = method.step(W, X, T,i)
            
        if hasattr(method, 'ann_schedule'):
            writer.add_scalar('train/annealing', method.ann_schedule[i], i)

        if i % 10 == 0:
            train_loss, train_pred = P.log_prob(W, X, T, return_loss=True)
            test_loss, test_pred = P.log_prob(W, x_test_0, y_test_0, return_loss=True)
            writer.add_scalar('train/train_mse', train_loss, i)
            writer.add_scalar('test/test_loss', test_loss, i)
            if 'driving' in locals():
                writer.add_scalar('train/driving_force', torch.mean(driving.abs()), i)
                writer.add_scalar('train/repulsive_force', torch.mean(repulsive.abs()), i)
                writer.add_scalar('train/forces_ratio', torch.mean(repulsive.abs())/torch.mean(driving.abs()), i)
            if config.method == 'SVGLD': 
                writer.add_scalar('train/langevin_noise', torch.mean(langevin_noise.abs()), i)
                #writer.add_scalar('train/bandwith', K.h, i)


            if ensemble.net.classification:
                std_test = test_pred.std(0).mean()
                Y = torch.mean(train_pred, 0)
                Y_t = torch.mean(test_pred, 0)
                entropies_test = -torch.sum(torch.log(Y_t + 1e-20) * Y_t, 1)
                entropies_train = -torch.sum(torch.log(Y + 1e-20) * Y, 1)
        

                train_accuracy = (torch.argmax(Y, 1) == torch.argmax(T, 1)).sum().item() / Y.shape[0] * 100
                test_accuracy = (torch.argmax(Y_t, 1) == test_labels).sum().item() / Y_t.shape[0] * 100
                writer.add_scalar('train/accuracy', train_accuracy, i)
                writer.add_scalar('test/accuracy', test_accuracy, i)
                writer.add_scalar('test/entropy', entropies_test.mean(), i)
                writer.add_scalar('train/entropy', entropies_train.mean(), i)
                dist = np.mean(pairwise_distances(W.detach().cpu().numpy()))
                writer.add_scalar('train/distance', dist, i)




                
                #accurcacy vs uncertainty plots: 
                #s, indices = torch.sort(entropies_test, dim = 0)
                #sorted_test = x_test_0[indices.squeeze()]
                
                #fig = plt.figure()
                #ax1 = fig.add_subplot(111)

                #ax1.plot(intensity, ece_l, markersize=20, c='b', marker="o", label='ECE')
                #ax1.scatter(bs_l,intensity, s=10, c='r', marker="o", label='Brier')
                #ax1.plot(intensity,corr_accuracy_l, markersize=20, c='r', marker="o", label='ACC') 
                #ax1.plot(intensity,roc_auc_l, markersize=20, c='orange', marker="o", label='AUROC')
                #ax1.legend()


                #writer.add_figure('corruption/Scores', plt.gcf(),i, close=not config.show_plots)
                #plt.close()
                sys.stdout.write('\rEpoch: %d, Train accuracy: %f, Test accuracy: %f       ' % \
                 (i, train_accuracy, test_accuracy))
                sys.stdout.flush()
                
# --------------------------------------------------------------------------------
# OOD TESTS
# --------------------------------------------------------------------------------
        if i % 500 == 0:
            pred_tensor_grid = ensemble.forward(torch.tensor(grid_mesh[2], dtype=torch.float).to(device))
            #pred_tensor = ensemble.forward(torch.tensor(np.expand_dims(grid_mesh[2].sum(1), 1), dtype=torch.float))[0]
            average_prob_grid = pred_tensor_grid[0].mean(0)
            std_prob_grid = pred_tensor_grid[0].std(0)
            entropies_grid = -torch.sum(torch.log(average_prob_grid + 1e-20) * average_prob_grid, 1)
            
            average_entrop_grid =-torch.sum(torch.log(pred_tensor_grid[0] + 1e-20) * pred_tensor_grid[0], 2).mean(0)
            
            diff_entropy_grid = entropies_grid - average_entrop_grid
            if i%100 == 0:
                contour_plot(grid_mesh,entropies_grid.cpu().numpy(),config,i,writer = writer,data = data, continuous = True, title = names[config.method])
                contour_plot(grid_mesh,diff_entropy_grid.cpu().numpy(),config,i,writer = writer,data = data, continuous = True, title = 'diff_entropy_grid '+ names[config.method])

                contour_plot(grid_mesh,average_entrop_grid.cpu().numpy(),config,i,writer = writer,data = data, continuous = True, title = 'Mean_Entropy '+ names[config.method])

                contour_plot(grid_mesh,std_prob_grid[:,0].cpu().numpy(),config,i,writer = writer, data = data,continuous = True, title = 'MD '+ names[config.method])
                #print(pred_tensor_grid[1][:,1,:])
                #print(pred_tensor_grid[0][:,1,:])
        
        
            print('-------------------------'+'Testing OOD'+'-------------------------')        

            softmax_ood = ensemble.forward(x_test_ood)[0]
            pred_ood = torch.argmax(softmax_ood, 2)

            average_prob_ood = softmax_ood.mean(0)
            std_prob_ood = softmax_ood.std(0).mean()
            
            ood_confidence = torch.max(average_prob_ood, 1)[0]
            test_confidence = torch.max(Y_t, 1)[0]

            # KL_uniform = -torch.sum(average_prob * torch.log2(
            #    (torch.ones(average_prob.shape[1]) / average_prob.shape[1] )/ average_prob + 1e-20)/log_scale, 1)
            # KL_uniform[KL_uniform != KL_uniform] = 0
            # writer.add_scalar('ood_metrics/AV_KL_uniform', KL_uniform.mean(), i)
            entropies_ood = -torch.sum(torch.log2(average_prob_ood + 1e-20) * average_prob_ood, 1)
            writer.add_scalar('ood_metrics/Av_entropy/', entropies_ood.mean(), i)
            writer.add_scalar('ood_metrics/entropy_ratio/', entropies_test.mean() / entropies_ood.mean(), i)
            
            #diversity = torch.mean(torch.min(pred_ood.sum(0), torch.tensor(pred_ood.shape[0])-pred_ood.sum(0) )/(pred_ood.shape[0]/2))
            #writer.add_scalar('test/diversity', diversity, i)

            rocauc_ood = ood_metrics_entropy(entropies_test.cpu().numpy(), entropies_ood.cpu().numpy(), writer, config, i, name='OOD')
            #rocauc_ood_std = ood_metrics_entropy(std_test.cpu().numpy(), std_prob_ood.cpu().numpy(), writer, config, i, name='OOD')

            writer.add_scalar('ood_metrics/AUROC', rocauc_ood[0], i)
            #writer.add_scalar('ood_metrics/AUROC_std', rocauc_ood_std[0], i)

            writer.add_scalar('ood_metrics/AUPR_IN', rocauc_ood[1], i)
            writer.add_scalar('ood_metrics/AUPR_OUT', rocauc_ood[2], i)

            # ECE calculation
            test_labels_np = test_labels.detach().cpu().numpy().astype(np.int8)
            #ece = um.numpy.ece(test_labels_np, average_prob_ood, num_bins=30)
            #writer.add_scalar('ood_metrics/ECE', ece, i)

            # bs = um.brier_score(labels=test_labels_np, probabilities=average_prob)

            # Confidence histogram
            if i%100 == 0:
                fig_1, axes_1 = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

                sns.kdeplot(ood_confidence.detach().cpu().numpy(), ax=axes_1, fill=True, common_norm=False, palette="crest",
                            alpha=.5, linewidth=3, label='OOD')
                sns.kdeplot(test_confidence.detach().cpu().numpy(), ax=axes_1, fill=True, common_norm=False, palette="crest",
                            alpha=.5, linewidth=3, label='Test')
                axes_1.legend()
                writer.add_figure('ood_metrics/confidence', plt.gcf(), i, close=not config.show_plots)
                plt.close()
                
                fig_2, axes_2 = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

                sns.kdeplot(entropies_ood.detach().cpu().numpy(), ax=axes_2, fill=True, common_norm=False, palette="crest",
                            alpha=.5, linewidth=3, label='OOD')
                sns.kdeplot(entropies_test.detach().cpu().numpy(), ax=axes_2, fill=True, common_norm=False, palette="crest",
                            alpha=.5, linewidth=3, label='Test')
                axes_2.legend()
                writer.add_figure('ood_metrics/entropy', plt.gcf(), i, close=not config.show_plots)
                plt.close()

                # distance matrix between particles .
                

                
    return rocauc_ood[0],rocauc_ood[1],rocauc_ood[2], train_accuracy, test_accuracy, entropies_test.mean().cpu().detach().numpy(), entropies_train.mean().cpu().detach().numpy(), entropies_ood.mean().cpu().detach().numpy(), std_test.cpu().detach().numpy() ,std_prob_ood.cpu().detach().numpy()