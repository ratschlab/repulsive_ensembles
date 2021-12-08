import matplotlib
import torch
from utils.distributions import Unorm_post
from utils.kernel import RBF
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from datetime import datetime
import numpy as np
from skimage.filters import gaussian
from utils.utils import ood_metrics_entropy


def train(data_train,data_test, ensemble, device, config,writer):
    """Train the particles using a specific ensemble.

    Args:
        data: A DATASET instance.
        mnet: The model of the main network.
        device: Torch device (cpu or gpu).
        config: The command line arguments.
        writer: The tensorboard summary writer.
    """

    # particles.train()

    #W = ensemble.particles.clone()

    W = ensemble.particles
    samples = []

    optimizer = torch.optim.Adam([W], config.lr, weight_decay=config.weight_decay,
                                 betas=[config.adam_beta1, 0.999])
#    optimizer = torch.optim.SGD([W], config.lr)

    #K_p = covariance_K()
    K = RBF()
    #K = real_PP_K()
    # prior = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(ensemble.net.num_params),
    #                                                                    config.prior_variance * torch.eye(
    #                                                                        ensemble.net.num_params))


    prior = torch.distributions.normal.Normal(torch.zeros(ensemble.net.num_params).to(device),
                              torch.ones(ensemble.net.num_params).to(device) * config.prior_variance)
    
    noise = torch.distributions.normal.Normal(torch.zeros(data_train.in_shape[0]**2).to(device),
                          torch.ones(data_train.in_shape[0]**2).to(device))

    P = Unorm_post(ensemble, prior, config, data_train.num_train_samples)

    log_scale = torch.log2(torch.tensor(data_train.out_shape[0],dtype=torch.float))
    
    x_test_0 = torch.tensor(data_train.get_test_inputs(), dtype=torch.float)
    y_test_0 = torch.tensor(data_train.get_test_outputs(), dtype=torch.float)
    
    blurred = gaussian(x_test_0[5],sigma=inte,multichannel=False)
    
    x_test_ood = torch.tensor(data_test.get_test_inputs(), dtype=torch.float)
    y_test_ood = torch.tensor(data_test.get_test_outputs(), dtype=torch.float)

    if config.method == 'SGLD':
        method = SGLD(P, K, optimizer)
    elif config.method == 'SGD':
        method = SGD(P,optimizer)
    elif config.method == 'SVGD_annealing':
        method = SVGD_annealing(P, K, optimizer,config)
    elif config.method == 'SVGD_debug':
        method = SVGD_debug(P,K,optimizer)
    elif config.method == 'SVGD':
        K = RBF()
        method = SVGD(P,K,optimizer)
    elif config.method == 'f_SVGD':
        method = functional_SVGD(P,K,optimizer)
    elif config.method == 'r_SGD':
        method = repulsive_SGD(P,K,optimizer)
    elif config.method == 'f_SGD':
        method = functional_SGD(P,K,optimizer)
    elif config.method == 'f_p_SVGD':
        K = real_PP_K()
        method = f_p_SVGD(P,K,optimizer)
    elif config.method == 'mixed_f_p_SVGD':
        #K_p = covariance_K()
        K = RBF()
        K_p = real_PP_K()
        method = mixed_f_p_SVGD(P,K,K_p,optimizer)
    elif config.method == 'log_p_SVGD':
        method = log_p_SVGD(P,K,optimizer)
    elif config.method == 'log_p_SGD':
        method = log_p_SGD(P,K,optimizer)
    elif config.method == 'fisher_SVGD':
        method = fisher_SVGD(P,K,optimizer)
    elif config.method == 'fisher_x_SVGD':
        K = real_PP_K()
        method = fisher_x_SVGD(P,K,optimizer)


    for i in range(config.epochs):

        optimizer.zero_grad()

        batch_train = data_train.next_train_batch(config.batch_size)
        batch_test = data_train.next_test_batch(config.batch_size)
        batch_ood = data_test.next_train_batch(config.batch_size)
        X = data_train.input_to_torch_tensor(batch_train[0], device, mode='train')
        T = data_train.output_to_torch_tensor(batch_train[1], device, mode='train')
        X_t = data_train.input_to_torch_tensor(batch_test[0], device, mode='train')
        T_t = data_train.output_to_torch_tensor(batch_test[1], device, mode='train')

        #X_ood = data_train.input_to_torch_tensor(batch_ood[0], device, mode='train')
        #T_ood = data_train.output_to_torch_tensor(batch_ood[1], device, mode='train')
        
        #Adding noise to test as oood 
        #x_test_ood = x_test_0 + noise.sample(torch.Size([x_test_0.shape[0]]))
        
        

        # if config.clip_grad_value != -1:
        #    torch.nn.utils.clip_grad_value_(optimizer.param_groups[0]['params'],
        #                                    config.clip_grad_value)
        # elif config.clip_grad_norm != -1:
        #    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'],
        #                                    config.clip_grad_norm)
        if config.method == 'SVGD_annealing':
            driving,repulsive = method.step(W, X, T,i)
        elif config.method == 'SVGD_debug':
            driving,repulsive = method.step(W, X, T)
        elif config.method == 'SGD':
            method.step(W, X, T)
        elif config.method == 'f_p_SVGD' or config.method == 'mixed_f_p_SVGD':
            noise_samples = noise.sample(torch.Size([config.batch_size]))
            driving,repulsive = method.step(W,X,T,noise_samples)
            #method.step(W, X, T, None)

        else:
            driving,repulsive = method.step(W, X, T)

        if i % 10 == 0:
            train_loss, train_pred = P.log_prob(W, X, T, return_loss=True)
            test_loss, test_pred = P.log_prob(W, x_test_0, y_test_0, return_loss=True)
            writer.add_scalar('train/train_loss', train_loss, i)
            writer.add_scalar('test/test_loss', test_loss, i)
            if config.method != 'SGD':
                writer.add_scalar('train/driving_force', torch.mean(driving.abs()), i)
                writer.add_scalar('train/repulsive_force', torch.mean(repulsive.abs()), i)
    #            writer.add_scalar('train/bandwith', K.h, i)

            if ensemble.net.classification:
                Y = torch.mean(train_pred, 0)
                Y_t = torch.mean(test_pred, 0)
                entropies_test = -torch.sum(torch.log2(Y_t + 1e-20)/log_scale * Y_t, 1)

                train_accuracy = (torch.argmax(Y, 1) == torch.argmax(T, 1)).sum().item() / Y.shape[0] * 100
                test_accuracy = (torch.argmax(Y_t, 1) == torch.argmax(y_test_0, 1)).sum().item() / Y_t.shape[0] * 100
                writer.add_scalar('train/accuracy', train_accuracy, i)
                writer.add_scalar('test/accuracy', test_accuracy, i)
                writer.add_scalar('test/entropy', entropies_test.mean(), i)

                print('Train iter:',i, ' train acc:', train_accuracy, 'test_acc', test_accuracy)

        if i % 50 == 0:
            if ensemble.net.classification:

                #ood diversity
                softmax_ood = ensemble.forward(x_test_ood)[0]
                pred_ood = torch.argmax(softmax_ood,2)

                average_prob = softmax_ood.mean(0)
                KL_uniform = -torch.sum(average_prob * torch.log2(
                    (torch.ones(average_prob.shape[1]) / average_prob.shape[1] )/ average_prob + 1e-20)/log_scale, 1)
                KL_uniform[KL_uniform != KL_uniform] = 0
                writer.add_scalar('ood_metrics/AV_KL_uniform', KL_uniform.mean(), i)
                entropies_ood = -torch.sum(torch.log2(average_prob + 1e-20)/log_scale * average_prob, 1)
                writer.add_scalar('ood_metrics/Av_entropy', entropies_ood.mean(), i)


                # diversity = torch.mean(torch.min(pred_ood.sum(0), torch.tensor(pred_ood.shape[0])-pred_ood.sum(0) )/(pred_ood.shape[0]/2))
                # writer.add_scalar('ood_metrics/diversity', diversity, i)
                average_prob = softmax_ood.mean(0)
                #entropies = -torch.sum(torch.log(average_prob + 1e-20) * average_prob, 1)
                #average_pred_ood = torch.argmax(average_prob,1)

                rocauc = ood_metrics_entropy(entropies_test,entropies_ood,writer,config,i)
                writer.add_scalar('ood_metrics/AUROC', rocauc[0], i)
                writer.add_scalar('ood_metrics/AUPR_IN', rocauc[1], i)
                writer.add_scalar('ood_metrics/AUPR_OUT', rocauc[2], i)

                #writer.add_hparams(dict(config.__dict__), {})


            # distance matrix between particles .

            dist= pairwise_distances(W.detach().numpy())
            plt.rcParams['figure.figsize'] = [10, 10]
            plt.matshow(dist +np.identity(config.n_particles)*np.max(dist))
            plt.colorbar()
            writer.add_figure('distance_particles', plt.gcf(),
                              i, close=not config.show_plots)
            plt.close()
            embedded_particles = TSNE(n_components=2).fit_transform(W.detach().numpy())
            plt.figure(figsize=(10, 10))
            plt.ylim(-500, 500)
            plt.xlim(-500, 500)
            plt.scatter(embedded_particles[:, 0], embedded_particles[:, 1], s = 100, alpha=0.8)
            writer.add_figure('2D_embedding', plt.gcf(),
                              i, close=not config.show_plots)

        if config.keep_samples != 0 and i % config.keep_samples == 0:
             samples.append(W.detach().clone())
             samples_models = torch.cat(samples)
        #     pred_tensor_samples = ensemble.forward(x_test_0, samples_models)
        #     plot_predictive_distributions(config,writer,i,data, [x_test_0.squeeze()], [pred_tensor_samples.mean(0).squeeze()],
        #                                   [pred_tensor_samples.std(0).squeeze()], save_fig=False, publication_style=False,
        #                                   name=config.method+'smp')

        if config.save_particles !=0 and i% config.save_particles == 0:
            particles = ensemble.particles.detach().numpy()
            np.save(datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.np', particles)

    return samples_models