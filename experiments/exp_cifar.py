import os
os.chdir('..')
from utils.config import configuration
import torch
from data.generate_dataset import generate_dataset
from models.ensemble import Ensemble
from tensorboardX import SummaryWriter
import os
import numpy as np
from data.toy_classification.generate_classification import generate_svhn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
from training.training_mnist_corruption import train
from datetime import datetime
from models.mnets_resnet import ResNet
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import sys
sys.stdout.flush()
import yaml 

def run():
    """Run the script.
    """
    n_runs = 1 
    results = []
    exp_dir = 'exp_'+datetime.now().strftime('%m-%d-%H-%M')
    config = configuration()
    torch.manual_seed(config.random_seed)
    for i in range(n_runs):
        date =  datetime.now().strftime('%H-%M-%S')
        if config.logit_soft == 0: 
            l_s='softmax'
        else: 
            l_s='logit'
        
        if config.noise: 
            alg = 'sto'
        else: 
            alg = 'det'
            
        f_date = datetime.now().strftime('%Y-%m-%d')

        dout_dir = './out/'+ f_date + '/'+ config.dataset +'_'+ config.exp_dir +'/'+config.method +'/'+ config.ann_sch+str(config.annealing_steps) +'/'+ l_s +'/'+alg+'/'+config.where_repulsive + 'part_'+str(config.n_particles)+'/hidden_'+str(config.size_hidden) +'/lr_'+str(config.lr)+'/l2_'+str(config.prior_variance)+'/'+'/seed_'+str(config.random_seed)+'_run_' + date 
        config.out_dir = dout_dir

        writer = SummaryWriter(log_dir=os.path.join(config.out_dir, 'summary'))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        writer.add_text('Comment', config.comment, 0)
        configu = dict(config.__dict__)
        del configu['comment']

        writer.add_text('Hparams', str(configu), 0)
        print(str(configu))
        dictionary_parameters = vars(config)

        with open(dout_dir+'/'+date+ 'parameters.yml', 'w') as yaml_file:
            yaml.dump(dictionary_parameters, stream=yaml_file, default_flow_style=False)
        #writer.add_hparams(configu,{})

        data, classification = generate_dataset(config)

        #data_ood = data[1]

        data_ood = generate_svhn()
        
        mnet = ResNet(out_act = F.softmax, use_batch_norm = False).to(device)

        l = []
        for _ in range(config.n_particles):
            l.append(torch.cat([p.flatten() for p in ResNet(out_act=F.softmax, no_weights=False,
                                                            use_batch_norm=False).parameters()]).detach())

        initial_particles = torch.stack(l).to(device)

        #ensemble = Ensemble(device = device, net=mnet, n_particles = config.n_particles)

        ensemble = Ensemble(device=device, net=mnet, particles=initial_particles)

        metrics = train(data, data_ood, ensemble, device, config,writer)
        
        results.append(metrics)

        particles = ensemble.particles.cpu().detach().numpy()

        np.save(dout_dir+'/'+date+'particles', particles)

        np.save(dout_dir+'/'+date+'results', np.array(metrics))


            
    #np.save('./out/'+ f_date + '/'+ config.dataset +'_'+ config.exp_dir +'/'+config.method +'_'+ config.ann_sch +'_'+ l_s +'_'+config.where_repulsive+'results', np.array(results))
    
if __name__ == '__main__':
    run()