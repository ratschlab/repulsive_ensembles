from data.toy_classification.oned_gaussian import oned_gaussian
from data.toy_classification.twod_gaussian import twod_gaussian
from data.toy_classification.donuts import Donuts
from data.toy_classification.moons import Moons
from data.mnist.mnist_data import MNISTData
from data.mnist.split_mnist import get_split_mnist_handlers
from data.fashion_mnist.fashion_data import FashionMNISTData
from data.cifar.cifar10_data import CIFAR10Data
from data.svhn.data_svhn_data import SVHNData
import os

def generate_moons(config):

    data = Moons(n_train=1500, n_test=500, noise=0.1, rseed = config.data_random_seed)

    return data

def generate_oned_gaussian():

    data = oned_gaussian()

    return data

def generate_twod_gaussian(config):

    data = twod_gaussian(rseed = 42,mu=[], sigma=[[1.,1.]for i in range(5)],n_train = 40, n_test = 20)

    return data

def generate_donuts(config):

    data = Donuts(r_1 = (9,10),r_2 = (3,4), c_outer_1 = (0,0), c_outer_2 = (0,0), n_train=100, n_test = 80, rseed = config.data_random_seed)

    return data

def generate_mnist():
    data = MNISTData(os.getcwd()+'/mnist', True)

    return data

def generate_split_mnist():
    data = get_split_mnist_handlers(os.getcwd()+'/mnist', True,num_classes_per_task = 5)

    return data

def generate_f_mnist():

    data = FashionMNISTData(os.getcwd()+'/fashion_mnist', True)

    return data

def generate_cifar():

    data = CIFAR10Data(os.getcwd()+'/cifar', True)

    return data

def generate_svhn():

    data = SVHNData(os.getcwd()+'/svhn', True)

    return data