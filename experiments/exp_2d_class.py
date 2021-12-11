import os
os.chdir('..')
from utils.config import configuration
import torch
from data.generate_dataset import generate_dataset
from models.MLP import Net
from models.ensemble import Ensemble
from tensorboardX import SummaryWriter
import os
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from training.training_2d_class import train

from datetime import datetime
import sys
sys.stdout.flush()
import yaml 

def run():
    """Run the script.
    """
    config = configuration()
    date =  datetime.now().strftime('%H-%M-%S')
    exp_dir = 'exp_'+datetime.now().strftime('%m-%d-%H-%M')
    torch.manual_seed(config.random_seed)

    if config.logit_soft == 0: 
        l_s='softmax'
    else: 
        l_s='logit'

    if config.noise: 
        alg = 'sto'
    else: 
        alg = 'det'

    f_date = datetime.now().strftime('%Y-%m-%d')

    dout_dir = './out/'+ f_date + '/'+ config.dataset +'_'+ config.exp_dir +'/'+config.method +'/'+ config.ann_sch+str(config.annealing_steps) +'/'+ l_s +'/'+alg+'/'+config.where_repulsive + 'part_'+str(config.n_particles)+'/lr_'+str(config.lr)+'/seed_'+str(config.random_seed)+'_run_' + date 
    config.out_dir = dout_dir
        

    writer = SummaryWriter(log_dir=os.path.join(config.out_dir, 'summary'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    writer.add_text('Comment', config.comment, 0)
    configu = dict(config.__dict__)
    del configu['comment']

    writer.add_text('Hparams', str(configu), 0)

    #writer.add_hparams(configu,{})

    data, classification = generate_dataset(config)

    #data_ood = data[1]

    #data_ood = generate_mnist()

    layer_sizes = [data.in_shape[0], data.out_shape[0]]

    for i in range(config.num_hidden):
        layer_sizes.insert(-1, config.size_hidden)

    mnet = Net(layer_sizes, classification = classification, act=F.relu,out_act = F.softmax, bias = True ).to(device)

    ensemble = Ensemble(device = device, net=mnet, n_particles = config.n_particles)

    metrics = train(data, ensemble, device, config,writer)

    results.append(metrics)

    particles = ensemble.particles.cpu().detach().numpy()

    np.save(dout_dir+'/'+date+'particles', particles)

    np.save(dout_dir+'/'+date+'results', np.array(metrics))

    dictionary_parameters = vars(config)

    with open(dout_dir+'/'+date+ 'parameters.yml', 'w') as yaml_file:
        yaml.dump(dictionary_parameters, stream=yaml_file, default_flow_style=False)
            
    np.save('./out/'+ f_date + '/'+ config.dataset +'_'+ config.exp_dir +'/'+config.method +'_'+ config.ann_sch +'_'+ l_s +'_'+config.where_repulsive+'results', np.array(results))
if __name__ == '__main__':
    run()