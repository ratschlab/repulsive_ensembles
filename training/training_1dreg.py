import torch

from utils.distributions import Unorm_post
from utils.utils import plot_predictive_distributions


def train(data, ensemble, device, config,writer):
    """Train the particles using a specific ensemble.

    Args:
        data: A DATASET instance.
        mnet: The model of the main network.
        device: Torch device (cpu or gpu).
        config: The command line arguments.
        writer: The tensorboard summary writer.
    """

    # particles.train()grid_mesh[1].shape

    #W = ensemble.particles.clone()

    W = ensemble.particles
    samples = []

    optimizer = torch.optim.Adam([W], config.lr, weight_decay=config.weight_decay,
                                 betas=[config.adam_beta1, 0.999])
#    optimizer = torch.optim.SGD([W], config.lr)

    K_p = covariance_K()
    K = RBF_2()

    prior = torch.distributions.normal.Normal(torch.zeros(ensemble.net.num_params).to(device),
                              torch.ones(ensemble.net.num_params).to(device) * config.prior_variance)

    P = Unorm_post(ensemble, prior, config, data.num_train_samples)

    if config.method == 'SGLD':
        method = SGLD(P, K, optimizer)
    elif config.method == 'SGD':
        method = SGD(P,K,optimizer)
    elif config.method == 'SVGD_annealing':
        method = SVGD_annealing(P, K, optimizer,config)
    elif config.method == 'SVGD_debug':
        method = SVGD_debug(P,K,optimizer)
    elif config.method == 'SVGD':
        method = SVGD(P,K,optimizer)
    elif config.method == 'f_SVGD':
        method = functional_SVGD(P,K,optimizer)
    elif config.method == 'r_SGD':
        method = repulsive_SGD(P,K,optimizer)
    elif config.method == 'f_SGD':
        method = functional_SGD(P,K,optimizer)
    elif config.method == 'f_p_SVGD':
        method = f_p_SVGD(P,K,optimizer)
    elif config.method == 'mixed_f_p_SVGD':
        method = mixed_f_p_SVGD(P,K,K_p,optimizer)
    elif config.method == 'log_p_SVGD':
        method = log_p_SVGD(P,K,optimizer)
    elif config.method == 'log_p_SGD':
        method = log_p_SGD(P,K,optimizer)
    elif config.method == 'fisher_SVGD':
        method = fisher_SVGD(P,K,optimizer)

    driving_l = []
    repulsive_l = []

    for i in range(config.epochs):

        optimizer.zero_grad()

        batch = data.next_train_batch(config.batch_size)
        X = data.input_to_torch_tensor(batch[0], device, mode='train')
        T = data.output_to_torch_tensor(batch[1], device, mode='train')
        x_test_0 = torch.tensor(data.get_test_inputs(), dtype=torch.float)
        y_test_0 = torch.tensor(data.get_test_outputs(), dtype=torch.float)

        # if config.clip_grad_value != -1:
        #    torch.nn.utils.clip_grad_value_(optimizer.param_groups[0]['params'],
        #                                    config.clip_grad_value)
        # elif config.clip_grad_norm != -1:
        #    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'],
        #                                    config.clip_grad_norm)
        if config.method == 'SVGD_annealing':
            driving,repulsive = method.step(W, X, T,i)
        elif config.method == 'SGD':
            method.step(W, X, T)
            driving = 0
            repulsive = 0
        elif config.method == 'f_p_SVGD' or config.method == 'mixed_f_p_SVGD':
            #grid_mesh_ood[torch.randint(0, grid_mesh_ood.shape[0], (200,))]
            #method.step(W,X,T,grid_mesh_ood[torch.randint(0,grid_mesh_ood.shape[0],(200,))])
            #driving,repulsive = method.step(W,X,T,ood_donuts)
            driving,repulsive = method.step(W,X,T)
            #method.step(W, X, T, None)

        else:
            driving,repulsive = method.step(W, X, T)

        driving_l.append(torch.mean(driving.abs()))
        repulsive_l.append(torch.mean(repulsive.abs()))

        if i % 10 == 0:
            train_loss, train_pred = P.log_prob(W, X, T, return_loss=True)
            test_loss, test_pred = P.log_prob(W, x_test_0, y_test_0, return_loss=True)
            writer.add_scalar('train/train_mse', train_loss, i)
            writer.add_scalar('test/test_loss', test_loss, i)
            writer.add_scalar('train/driving_force', torch.mean(driving.abs()), i)
            writer.add_scalar('train/repulsive_force', torch.mean(repulsive.abs()), i)
#            writer.add_scalar('train/bandwith', K.h, i)

            # writer.add_scalar('train/task_%d/loss_nll' % task_id, loss_nll, i)
            # writer.add_scalar('train/task_%d/log_det_j' % task_id, torch.mean(log),i)
            # writer.add_scalar('train/task_%d/loss' % task_id, loss, i)
            # print('Train iter:', i, train_loss)
            # print('Test iter:', i, test_loss)

            if ensemble.net.classification:
                Y = torch.mean(train_pred, 0)
                Y_t = torch.mean(test_pred, 0)
                train_accuracy = (torch.argmax(Y, 1) == torch.argmax(T, 1)).sum().item() / Y.shape[0] * 100
                test_accuracy = (torch.argmax(Y_t, 1) == torch.argmax(y_test_0, 1)).sum().item() / Y_t.shape[0] * 100
                writer.add_scalar('train/accuracy', train_accuracy, i)
                writer.add_scalar('test/accuracy', test_accuracy, i)
                print('Train iter:',i, ' train acc:', train_accuracy, 'test_acc', test_accuracy)
            # else:
            #     print('Train iter:', i, train_loss)
            #     print('Test iter:', i, test_loss)



            # Plot distribution of mean and log-variance values.
            # mean_outputs = torch.cat([d.clone().view(-1) for d in flow._w_0_mu])
            # logvar_outputs = torch.cat([d.clone().view(-1) for d in flow._w_0_logvar])
            # writer.add_histogram('train/task_%d/input_flow_mean' % task_id,
            # mean_outputs, i)
            # writer.add_histogram('train/task_%d/input_flow_logvar' % task_id,
            # logvar_outputs, i)
        if i % 50 == 0:
            if ensemble.net.classification and data.in_shape[0]==2:
                pred_tensor = ensemble.forward(torch.tensor(grid_mesh[2], dtype=torch.float))[0]
                #pred_tensor = ensemble.forward(torch.tensor(np.expand_dims(grid_mesh[2].sum(1), 1), dtype=torch.float))[0]
                average_prob = pred_tensor.mean(0)
                decision_b = torch.argmax(average_prob,1)
                entropies = -torch.sum(torch.log(average_prob + 1e-20) * average_prob, 1)
                #contour_plot(grid_mesh,entropies,config,i,writer = writer,data = None, title = 'Entropy '+ config.method)
                #contour_plot(grid_mesh,average_prob[:,0],config,i,writer = writer, data = None, title = 'Softmax'+ config.method)
                #contour_plot(grid_mesh,decision_b,config,i,writer = writer, data = data, title = 'Decision Boundary'+ config.method)

                #ood diversity

                pred_ood = torch.argmax((ensemble.forward(torch.tensor(grid_mesh_ood, dtype=torch.float))[0]),2)

                diversity = torch.mean(torch.min(pred_ood.sum(0), torch.tensor(pred_ood.shape[0])-pred_ood.sum(0) )/(pred_ood.shape[0]/2))
                writer.add_scalar('test/diversity', diversity, i)


            elif config.dataset == 'toy_reg':
                pred_tensor = ensemble.forward(torch.tensor(x_test_0, dtype=torch.float))

                plot_predictive_distributions(config,writer,i,data, x_test_0.squeeze(), pred_tensor.mean(0).squeeze(),
                                              pred_tensor.std(0).squeeze(), save_fig=False, publication_style=False,
                                              name=config.method)
            # correlation matrix.

            #dist= pairwise_distances(W.detach().numpy())
            #plt.rcParams['figure.figsize'] = [10, 10]
            #plt.matshow(dist)
            #plt.colorbar()
            #writer.add_figure('distance_particles', plt.gcf(),
            #                  i, close=not config.show_plots)
            #plt.close()
            #embedded_particles = TSNE(n_components=2).fit_transform(W.detach().numpy())
            #plt.figure(figsize=(10, 10))
            #plt.ylim(-500, 500)
            #plt.xlim(-500, 500)
            #plt.scatter(embedded_particles[:, 0], embedded_particles[:, 1], s = 100, alpha=0.8)
            #writer.add_figure('2D_embedding', plt.gcf(),
            #                  i, close=not config.show_plots)

        if config.keep_samples != 0 and i % config.keep_samples == 0:
             samples.append(W.detach().clone())
             samples_models = torch.cat(samples)
        #     pred_tensor_samples = ensemble.forward(x_test_0, samples_models)
        #     plot_predictive_distributions(config,writer,i,data, [x_test_0.squeeze()], [pred_tensor_samples.mean(0).squeeze()],
        #                                   [pred_tensor_samples.std(0).squeeze()], save_fig=False, publication_style=False,
        #                                   name=config.method+'smp')

    return  driving_l, repulsive_l