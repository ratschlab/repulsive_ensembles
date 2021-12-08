import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as D
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

from utils.distributions import Unorm_post
from methods.method_utils import create_method
from utils.utils import contour_plot
from utils.utils import ood_metrics_entropy

warnings.filterwarnings("ignore")
import seaborn as sns

sns.set()

def train(data, ensemble, device, config,writer):
    """Train the particles using a specific ensemble.

    Args:
        data: A DATASET instance.
        mnet: The model of the main network.
        device: Torch device (cpu or gpu).
        config: The command line arguments.
        writer: The tensorboard summary writer.
    """

# --------------------------------------------------------------------------------
# CREATING TESTING AND OOD DATASET 
# --------------------------------------------------------------------------------
    
    grid_mesh = data.get_input_mesh(x1_range=(-20, 20), x2_range=(-20, 20), grid_size=200)
    
    x_test_0 = torch.tensor(data.get_test_inputs(), dtype=torch.float)
    y_test_0 = torch.tensor(data.get_test_outputs(), dtype=torch.float)
    test_labels = torch.argmax(y_test_0, 1).type(torch.int)

    #p_1 = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor((5., 0)), 0.3 * torch.eye(2))
    #p_2 = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor((-5., 0)), 0.3 * torch.eye(2))
    #s_1 = p_1.sample(torch.Size([100]))
    #s_2 = p_2.sample(torch.Size([100]))
    #x_test_ood = torch.cat([s_1, s_2])
    
    if config.dataset == 'donuts':
        ext_ring_ood = data.sample(10 + 20, 9 + 2, 0, 0, size=200)
        p_in = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor((0., 0.)), 0.4 * torch.eye(2))
        s_in = p_in.sample(torch.Size([50]))
        x_test_ood = torch.cat([s_in, torch.tensor(ext_ring_ood, dtype=torch.float), ])
    elif config.dataset == 'moons':
        # Construct Gaussian Mixture Model in 1D consisting of 5 equally
        # weighted normal distributions
        mu_m = [[4,0],[-4,-0],[0,4],[0,-4]]
        sigma_m = [[0.5,0.5],[0.5,0.5],[0.5,0.5],[0.5,0.5]]
        mix = D.Categorical(torch.tensor([0.25,0.25,0.25,0.25],dtype=torch.float))
        comp = D.Independent(D.Normal(
                     torch.tensor(mu_m,dtype=torch.float), torch.tensor(sigma_m,dtype=torch.float)), 1)
        gmm = torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp)
        x_test_ood = gmm.sample(torch.Size([200]))


# --------------------------------------------------------------------------------
# SETTING PROBABILISTIC and UTILS
# --------------------------------------------------------------------------------
    W = ensemble.particles
    samples = [] 
    optimizer = torch.optim.Adam([W], config.lr, weight_decay=config.weight_decay,
                                 betas=[config.adam_beta1, 0.999])
    #optimizer = torch.optim.SGD([W], config.lr)
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
    log_scale = torch.log2(torch.tensor(data_train.out_shape[0], dtype=torch.float))

# --------------------------------------------------------------------------------
# SVGD ALGORITHM SPECIFICATIONS
# --------------------------------------------------------------------------------

    method = create_method(config, P, optimizer)
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
        elif config.method == 'f_p_SVGD' or config.method == 'mixed_f_p_SVGD':
            #noise_samples = noise.sample(torch.Size([config.batch_size]))
            driving,repulsive = method.step(W,X,T,i,None)
            #driving,repulsive = method.step(W,X,T,noise_samples)

            #driving,repulsive = method.step(W, X, T, None)
        elif config.method == 'SVGLD':
            driving,repulsive,langevin_noise = method.step(W, X, T,i)

        else:
            driving,repulsive = method.step(W, X, T,i)
        #driving_l.append(torch.mean(driving.abs()))
        #repulsive_l.append(torch.mean(repulsive.abs()))

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
                Y = torch.mean(train_pred, 0)
                Y_t = torch.mean(test_pred, 0)
                entropies_test = -torch.sum(torch.log2(Y_t + 1e-20)/log_scale * Y_t, 1)
                entropies_train = -torch.sum(torch.log2(Y + 1e-20)/log_scale * Y, 1)

                train_accuracy = (torch.argmax(Y, 1) == torch.argmax(T, 1)).sum().item() / Y.shape[0] * 100
                test_accuracy = (torch.argmax(Y_t, 1) == test_labels).sum().item() / Y_t.shape[0] * 100
                writer.add_scalar('train/accuracy', train_accuracy, i)
                writer.add_scalar('test/accuracy', test_accuracy, i)
                writer.add_scalar('test/entropy', entropies_test.mean(), i)
                writer.add_scalar('train/entropy', entropies_train.mean(), i)

                print('Train iter:', i, ' train acc:', train_accuracy, 'test_acc', test_accuracy)

# --------------------------------------------------------------------------------
# OOD TESTS
# --------------------------------------------------------------------------------
        if i % 100 == 0:
            pred_tensor_grid = ensemble.forward(torch.tensor(grid_mesh[2], dtype=torch.float))
            #pred_tensor = ensemble.forward(torch.tensor(np.expand_dims(grid_mesh[2].sum(1), 1), dtype=torch.float))[0]
            average_prob_grid = pred_tensor_grid[0].mean(0)
            std_prob_grid = pred_tensor_grid[0].std(0)
            decision_b_grid = torch.argmax(average_prob_grid,1)
            entropies_grid = -torch.sum(torch.log(average_prob_grid + 1e-20)/log_scale * average_prob_grid, 1)
            
            average_entrop_grid =-torch.sum(torch.log(pred_tensor_grid[0] + 1e-20)/log_scale * pred_tensor_grid[0], 2).mean(0)
            
            diff_entropy_grid = entropies_grid - average_entrop_grid
            contour_plot(grid_mesh,entropies_grid,config,i,writer = writer,data = None, continuous = True, title = 'Entropy '+ config.method)
            contour_plot(grid_mesh,average_entrop_grid,config,i,writer = writer,data = None, continuous = True, title = 'Mean_Entropy '+ config.method)
            contour_plot(grid_mesh,diff_entropy_grid,config,i,writer = writer,data = None, continuous = True, title = 'diff_entropy_grid '+ config.method)

            contour_plot(grid_mesh,average_prob_grid[:,0],config,i,writer = writer, data = None, title = 'Softmax'+ config.method)
            contour_plot(grid_mesh,decision_b_grid,config,i,writer = writer, data = data, title = 'Decision Boundary'+ config.method)
            contour_plot(grid_mesh,std_prob_grid[:,0],config,i,writer = writer, data = None,continuous = True, title = 'STD'+ config.method)
            #print(pred_tensor_grid[1][:,1,:])
            #print(pred_tensor_grid[0][:,1,:])
        
        
            print('-------------------------'+'Testing OOD'+'-------------------------')        
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

            softmax_ood = ensemble.forward(x_test_ood)[0]
            pred_ood = torch.argmax(softmax_ood, 2)

            average_prob_ood = softmax_ood.mean(0)
            ood_confidence = torch.max(average_prob_ood, 1)[0]
            test_confidence = torch.max(Y_t, 1)[0]

            # KL_uniform = -torch.sum(average_prob * torch.log2(
            #    (torch.ones(average_prob.shape[1]) / average_prob.shape[1] )/ average_prob + 1e-20)/log_scale, 1)
            # KL_uniform[KL_uniform != KL_uniform] = 0
            # writer.add_scalar('ood_metrics/AV_KL_uniform', KL_uniform.mean(), i)
            entropies_ood = -torch.sum(torch.log2(average_prob_ood + 1e-20)/log_scale * average_prob_ood, 1)
            writer.add_scalar('ood_metrics/Av_entropy/', entropies_ood.mean(), i)
            writer.add_scalar('ood_metrics/entropy_ratio/', entropies_test.mean() / entropies_ood.mean(), i)
            
            diversity = torch.mean(torch.min(pred_ood.sum(0), torch.tensor(pred_ood.shape[0])-pred_ood.sum(0) )/(pred_ood.shape[0]/2))
            writer.add_scalar('test/diversity', diversity, i)

            rocauc = ood_metrics_entropy(entropies_test, entropies_ood, writer, config, i, name='OOD')
            writer.add_scalar('ood_metrics/AUROC', rocauc[0], i)
            writer.add_scalar('ood_metrics/AUPR_IN', rocauc[1], i)
            writer.add_scalar('ood_metrics/AUPR_OUT', rocauc[2], i)

            # ECE calculation
            test_labels_np = test_labels.detach().numpy().astype(np.int8)
            #ece = um.numpy.ece(test_labels_np, average_prob_ood, num_bins=30)
            #writer.add_scalar('ood_metrics/ECE', ece, i)

            # bs = um.brier_score(labels=test_labels_np, probabilities=average_prob)

            # Confidence histogram

            sns.kdeplot(ood_confidence.detach().numpy(), ax=axes, fill=True, common_norm=False, palette="crest",
                        alpha=.5, linewidth=3, label='OOD')
            sns.kdeplot(test_confidence.detach().numpy(), ax=axes, fill=True, common_norm=False, palette="crest",
                        alpha=.5, linewidth=3, label='Test')
            axes.legend()
            writer.add_figure('ood_metrics/confidence', plt.gcf(), i, close=not config.show_plots)
            plt.close()

            # distance matrix between particles .

            dist = pairwise_distances(W.detach().numpy())
            plt.rcParams['figure.figsize'] = [10, 10]
            plt.matshow(dist + np.identity(config.n_particles) * np.max(dist))
            plt.colorbar()
            writer.add_figure('distance_particles', plt.gcf(),
                              i, close=not config.show_plots)
            plt.close()
            embedded_particles = TSNE(n_components=2).fit_transform(W.detach().numpy())
            plt.figure(figsize=(10, 10))
            plt.ylim(-500, 500)
            plt.xlim(-500, 500)
            plt.scatter(embedded_particles[:, 0], embedded_particles[:, 1], s=100, alpha=0.8)
            writer.add_figure('2D_embedding', plt.gcf(),
                              i, close=not config.show_plots)
