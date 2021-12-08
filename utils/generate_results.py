import os
import numpy as np 
import pandas as pd 
import torch
from natsort import natsorted, index_natsorted
from random import shuffle
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description='results')
parser.add_argument('--path', type=str, default=None,
                        help='Folder where to find the results.')
parser.add_argument('--seeds', type=int, default=5,
                        help='Number of seeds used.')
parser.add_argument('--subf', action='store_true', default=False,
                    help='If the folder contains subfolders on which to interate')
args = parser.parse_args()

if os.path.exists(args.path+"/results.csv"):
    print('Old results found, removing...')
    os.remove(args.path+"/results.csv")
    
if os.path.exists(args.path+"/params.csv"):
    os.remove(args.path+"/params.csv")


cols = ['AUROC','AUR_IN','AUR_OUT', 'train_acc', 'test_acc', 'H_test', 'H_train', 'H_ood', 'std_ood', 'std_test', 'ECE', 'NLL', 'AUROC_std']
paths_l = [] 
name_l = []
for root, dirs, files in os.walk(args.path+"/", topdown=False):
    for name in files:
        if 'results.npy' in name and len(name)<31:
            print(name)
            path = os.path.join(root, name)
            paths_l.append(path)
            name = path.replace(args.path,'')
            name_l.append(name[:])

idx = index_natsorted(name_l)
name_l = [name_l[i] for i in idx]
paths_l = [paths_l[j] for j in idx]
mean_res = [np.load(i) for i in paths_l]
std_res = [np.load(i) for i in paths_l]
name_cpopy = []
for n in name_l: 
    index = n.find('seed') #stores the index of a substring or char
    name_cpopy.append(n[:index])
my_dict = {i:name_cpopy.count(i) for i in name_cpopy}
name_cpopy = list(set(name_cpopy))
name_cpopy = natsorted(name_cpopy)

g = []
for n in natsorted(name_cpopy): 
    l=[]
    for p in paths_l: 
        if n in p: 
            #print(n)
            r = np.load(p,allow_pickle=True)
            l.append(r)
    g.append(l)     

mean_res = [np.mean(resu,0) for resu in g]
res = [np.array(resu) for resu in g]

std_res = [np.std(resu,0)/len(resu) for resu in g]
results_20p = pd.DataFrame.from_dict(dict(zip(name_cpopy, mean_res)), orient='index')
results_20p.columns = cols
results_20p['ratio_entr'] = results_20p['H_ood']/results_20p['H_test']
results_20p['ratio_MD'] = results_20p['std_ood']/results_20p['std_test']
results_20p = results_20p.sort_values(by =['test_acc'] , ascending=False).round(3)

############# Computing error of entropy ratios ##############
mean_ratios = [m_r[8]/m_r[9] for m_r in mean_res]
cov_l = [np.cov(np.squeeze(np.stack([np.expand_dims(r[:,9],1),np.expand_dims(r[:,8],1)]),2))[0,1] for r in res]
errors=[]
for i in range(len(mean_ratios)):
    errors.append(1/mean_res[i][9]**2*(std_res[i][8]**2 - 2*mean_ratios[i]*cov_l[i] + mean_ratios[i]**2*std_res[i][9]**2 ))
md_ratio_error = pd.DataFrame.from_dict(dict(zip(natsorted(name_cpopy) , np.sqrt(np.abs(errors))/5)), orient='index')
md_ratio_error.columns = ['ratio_MD']
md_ratio_error = md_ratio_error.reindex(results_20p.index)

############# Computing error of std ratios ##############
mean_ratios = [m_r[7]/m_r[5] for m_r in mean_res]
cov_l = [np.cov(np.squeeze(np.stack([np.expand_dims(r[:,5],1),np.expand_dims(r[:,7],1)]),2))[0,1] for r in res]
errors=[]
for i in range(len(mean_ratios)):
    errors.append(1/mean_res[i][5]**2*(std_res[i][7]**2 - 2*mean_ratios[i]*cov_l[i] + mean_ratios[i]**2*std_res[i][5]**2 ))
h_ratio_error = pd.DataFrame.from_dict(dict(zip(natsorted(name_cpopy) , np.sqrt(np.abs(errors))/5)), orient='index')
h_ratio_error.columns = ['ratio_entr']
h_ratio_error = h_ratio_error.reindex(results_20p.index)

std_20p = pd.DataFrame.from_dict(dict(zip(name_cpopy, std_res)), orient='index')
std_20p.columns = cols
std_20p = std_20p.reindex(results_20p.index)
std_20p = pd.concat([std_20p, h_ratio_error,md_ratio_error], axis=1)
#std_20p = std_20p.rename({'md_ratio_entr':'ratio_MD'},axis='columns')
std_20p_array =  std_20p.round(3).astype(str).values

df3 = results_20p.round(3).applymap(str) + 'pm'+std_20p.round(3).applymap(str)
df3 = df3[['AUROC','AUROC_std', 'test_acc','ratio_entr','ratio_MD','ECE','NLL']]

df3.to_csv(args.path+'/results.csv')
print('Results saved in:', args.path+'/results.csv')

name_cpopy_res = name_cpopy

paths_l = [] 
name_l = []
for root, dirs, files in os.walk(args.path+"/", topdown=False):
    for name in files:
        if 'parameters' in name and len(name)<31:
            #print(name)
            path = os.path.join(root, name)
            paths_l.append(path)
            name = path.replace(args.path,'')
            name_l.append(name[:])
#print(natsorted(name_l))

idx = index_natsorted(name_l)
name_l = [name_l[i] for i in idx]
paths_l = [paths_l[j] for j in idx]
name_cpopy = []
for n in name_l: 
    index = n.find('seed') #stores the index of a substring or char
    name_cpopy.append(n[:index])
#name_cpopy = list(set(name_cpopy))
name_cpopy = natsorted(name_cpopy)

import pandas as pd
import yaml
dfs = []
for p in paths_l:
    with open(p, 'r') as f:
        d = pd.io.json.json_normalize(yaml.load(f))
        #d.index(name_cpopy[0])
        dfs.append(d)
        
df = pd.concat(dfs)
df = df.reset_index(drop = True)
df_new = pd.DataFrame(data=df.values, index=name_cpopy, columns = list(df.columns)).drop(['comment','out_dir'],axis=1)
print(df_new)
# I take one every 5 rows because only the random seed is changing
df_new.iloc[::args.seeds, :].reindex(name_cpopy_res).to_csv(args.path+'/params.csv')
print('Params saved in:', args.path+'/params.csv')