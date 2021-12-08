from datetime import datetime
import argparse

def configuration(args = None):
    parser = argparse.ArgumentParser(description='SVGD ' +
                                                 'Bayesian neural networks.')

    tgroup = parser.add_argument_group('Training options')
    tgroup.add_argument('--epochs', type=int, metavar='N', default=5000,
                        help='Number of training epochs. '+
                             'Default: %(default)s.')
    tgroup.add_argument('--batch_size', type=int, metavar='N', default=128,
                        help='Training batch size. Default: %(default)s.')
    tgroup.add_argument('--lr', type=float, default=1e-2,
                        help='Learning rate of optimizer. Default: ' +
                             '%(default)s.')
    tgroup.add_argument('--momentum', type=float, default=0.0,
                        help='Momentum of the optimizer. ' +
                             'Default: %(default)s.')
    tgroup.add_argument('--adam_beta1', type=float, default=0.9,
                        help='The "beta1" parameter when using torch.optim.' +
                             'Adam as optimizer. Default: %(default)s.')
    tgroup.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay of the optimizer(s). Default: ' +
                             '%(default)s.')
    tgroup.add_argument('--method', type=str, default='SVGD',
                        help='Method for optimization, options: SVGD,SGD,SGLD,SVGD_debug. Default: ' +
                             '%(default)s.')
    tgroup.add_argument('--noise', action='store_true', default=False,
                    help='Flag to enable noise injected in the method')
    tgroup.add_argument('--dataset', type=str, default='moons',
                        help='Dataset, options: toy_regression, moons, twod_gaussian, oned_gaussian. Default: ' +
                             '%(default)s.')
    tgroup.add_argument('--optim', type=str, default='Adam',
                        help='Otimizer, options: Adam,SGD. Default: ' +
                             '%(default)s.')
    tgroup.add_argument('--clip_grad_value', type=float, default=-1,
                        help='If not "-1", gradients will be clipped using ' +
                             '"torch.nn.utils.clip_grad_value_". Default: ' +
                             '%(default)s.')
    tgroup.add_argument('--n_particles', type=int, default=5,
                        help='Number of particles used for the approximation of the gradient flow')

    sgroup = parser.add_argument_group('Network options')
    sgroup.add_argument('--num_hidden', type=int, metavar='N', default=1,
                        help='Number of hidden layer in the (student) ' +
                             'network. Default: %(default)s.')
    sgroup.add_argument('--size_hidden', type=int, metavar='N', default=10,
                        help='Number of units in each hidden layer of the ' +
                             '(student) network. Default: %(default)s.')
    sgroup.add_argument('--num_train_samples', type=int, default=20,
                        help='Number of data training points.')
    sgroup.add_argument('--prior_variance', type=float, default=1.,
                        help='Variance of the Gaussian prior. ' +
                             'Default: %(default)s.')
    sgroup.add_argument('--pred_dist_std', type=float, default=1.,
                        help='The standard deviation of the predictive ' +
                             'distribution. Note, this value should be ' +
                             'fixed and reasonable for a given dataset.' +
                             'Default: %(default)s.')

    mgroup = parser.add_argument_group('Miscellaneous options')
    mgroup.add_argument('--use_cuda', action='store_true',
                        help='Flag to enable GPU usage.')
    mgroup.add_argument('--random_seed', type=int, metavar='N', default=42,
                        help='Random seed. Default: %(default)s.')
    mgroup.add_argument('--data_random_seed', type=int, metavar='N', default=42,
                        help='Data random seed. Default: %(default)s.')
    mgroup.add_argument('--dont_show_plot', action='store_false',
                        help='Dont show the final regression results as plot.' +
                             'Note, only applies to 1D regression tasks.')

    dout_dir = './out/'+datetime.now().strftime('%Y-%m-%d')+'/run_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_dir = 'exp_'+datetime.now().strftime('%Y-%m-%d_%H-%M')

    mgroup.add_argument('--out_dir', type=str, default=dout_dir,
                        help='Where to store the outputs of this simulation.')
    mgroup.add_argument('--comment', type=str, default=dout_dir,
                        help='Comment for the running experiment.')
    mgroup.add_argument('--show_plots', action='store_true',
                        help='Whether plots should be shown.')
    
    expgroup = parser.add_argument_group('Experiments options')
    expgroup.add_argument('--annealing_steps', type=int, default=0,
                        help='Annealing steps. ' +
                             'Default: %(default)s.')
    expgroup.add_argument('--keep_samples', type=float, default=0,
                        help='Keep samples during training ' +
                             'Default: %(default)s.')
    expgroup.add_argument('--save_particles', type=float, default=0,
                        help='Save particles in the end or during training ' +
                             'Default: %(default)s.')
    expgroup.add_argument('--gamma', type=float, default=1.,
                    help='SVGD scaling forces ' +
                         'Default: %(default)s.')
    expgroup.add_argument('--ann_sch', type=str, default='linear',
                        help='Annealing schedule options. Default: ' +
                             '%(default)s.')
    expgroup.add_argument('--logit_soft', type=int, default=0,
                        help='If 1 functional SVGD on logit, if 0 softmax. ' +
                             'Default: %(default)s.')
    expgroup.add_argument('--where_repulsive', type=str, default='train',
                        help='If train functional repulsion on train, if test on test, if noise on noisy training points. ' +
                             'Default: %(default)s.')
    expgroup.add_argument('--num_train', action='store_true', default=False,
                    help='Flag to enable GPU usage.')
    expgroup.add_argument('--exp_dir', type=str, default=exp_dir,
                    help='directory for all run same experiment')
    
    
    if args is None:
        return parser.parse_args()
    else:
        return parser.parse_args(args = args)
