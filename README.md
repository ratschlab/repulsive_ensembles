# Repulsive Deep Ensembles are Bayesian

This repo contains the code of the paper [Repulsive deep ensembles are Bayesian](https://proceedings.neurips.cc/paper/2021/hash/1c63926ebcabda26b5cdb31b5cc91efb-Abstract.html). In the following some usage examples can be found

## Sampling from synthetic distributions experiments 
The experiment for the synthetic distributions can be found in 'notebooks/WGD_synthetic.ipynb'

## 1d regression experiments

The 1d toy regression problem can be explored. Example run:

```console
$ python3 experiments/exp_regr.py --epochs 5000 --lr 1e-2 --n_particles 100 --size_hidden 10 --num_hidden 2 --method SGD --prior_variance 1 --annealing_steps 1000  --batch_size 32 --dataset toy_reg --ann_sch None 

```

## 2d classification experiments

The 2d classification problem can be explored. Example run:

```console
$ python3 experiments/exp_2d_class.py --epochs 10000 --lr 1e-2 --n_particles 100 --size_hidden 10 --num_hidden 2 --method SVGD --prior_variance 1 --annealing_steps 1000 --batch_size 128 --dataset twod_gaussian --ann_sch None 

```
## Citation

If you use our code or consider our ideas in your research project, please consider citing our paper.
```
@article{d2021repulsive,
  title={Repulsive Deep Ensembles are Bayesian},
  author={D'Angelo, Francesco and Fortuin, Vincent},
  journal={arXiv preprint arXiv:2106.11642},
  year={2021}
}
```
