import os
from datetime import datetime
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from utils.config import configuration
from data.generate_dataset import generate_dataset
from models.MLP import Net
from models.ensemble import Ensemble
from training.training_1d_regre import train


def run():
    """Run the script.
    """

    config = configuration()
    date =  datetime.now().strftime('%H-%M-%S')
    exp_dir = 'exp_'+datetime.now().strftime('%m-%d-%H-%M')
    torch.manual_seed(config.random_seed)

    if config.noise: 
        alg = 'sto'
    else: 
        alg = 'det'

    f_date = datetime.now().strftime('%Y-%m-%d')

    dout_dir = './out/'+ f_date + '/'+ config.dataset +'_'+ config.exp_dir +'/'+config.method +'/'+ config.ann_sch+str(config.annealing_steps) +'/'+config.where_repulsive + 'part_'+str(config.n_particles)+'/lr_'+str(config.lr)+'/seed_'+str(config.random_seed)+'_run_' + date 
    config.out_dir = dout_dir


    writer = SummaryWriter(log_dir=os.path.join(config.out_dir, 'summary'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    writer.add_text('Comment', config.comment, 0)
    configu = dict(config.__dict__)
    del configu['comment']

    writer.add_text('Hparams', str(configu), 0)
    data, classification = generate_dataset(config)
    layer_sizes = [data.in_shape[0], data.out_shape[0]]

    for i in range(config.num_hidden):
        layer_sizes.insert(-1, config.size_hidden)

    mnet = Net(layer_sizes, classification = classification, act=F.relu,out_act = None).to(device)
    
    #l = []
    #for _ in range(config.n_particles):
    #    l.append(torch.cat([p.flatten() for p in Net(layer_sizes, classification = True, act=F.relu,out_act = F.softmax, bias = True, no_weights=False).parameters()][len(mnet.param_shapes):]).detach())

    #initial_particles = torch.stack(l).to(device)

    ensemble = Ensemble(device = device, net=mnet, n_particles = config.n_particles)
    #ensemble = Ensemble(device = device, net=mnet,particles=initial_particles)

    #ensemble = Ensemble(device = device, net=mnet, n_particles = config.n_particles)

    train(data, ensemble, device, config,writer)

    #particles = ensemble.particles.detach().numpy()

    #np.save(date+'.np', particles)

if __name__ == '__main__':
    run()