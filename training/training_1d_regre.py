import seaborn as sns
import torch

from utils.distributions import Unorm_post
from methods.method_utils import create_method
from utils.utils import plot_predictive_distributions

sns.set()

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
# CREATING TESTING 
# --------------------------------------------------------------------------------
    x_test_0 = torch.tensor(data.get_test_inputs(), dtype=torch.float).to(device)
    y_test_0 = torch.tensor(data.get_test_outputs(), dtype=torch.float).to(device)

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
        elif config.method == 'f_p_SVGD' or config.method == 'mixed_f_p_SVGD' or config.method == 'f_s_SVGD' or config.method == 'f_SGD':
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

        if i % 1000 == 0:
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

            pred_tensor = ensemble.forward(torch.tensor(x_test_0, dtype=torch.float))[0]
            plot_predictive_distributions(config,writer,i,data, x_test_0.cpu().squeeze(), pred_tensor.cpu().mean(0).squeeze(),
                                              pred_tensor.cpu().std(0).squeeze(), save_fig=True,
                                              name=names[config.method])
            print('Train iter:',i, ' train mse:', train_loss, 'test mse', test_loss, flush = True)
