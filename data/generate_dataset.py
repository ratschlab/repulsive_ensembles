from data.toy_classification.generate_classification import generate_moons, generate_oned_gaussian, generate_twod_gaussian, generate_mnist, generate_f_mnist, generate_donuts, generate_split_mnist, generate_cifar, generate_svhn
from data.toy_regression.generete_regression import generate_1d_dataset


def generate_dataset(config):
    """Generate a dataset.

    Args:
        config: Command-line arguments.

    Returns:
        data_handlers(DATASET): A data handlers.
        classification(bool): Whether the dataset is a classification task or not
    """
    if config.dataset == 'toy_reg':
        classification = False
        return generate_1d_dataset(show_plots=True, task_set=4,
                                 data_random_seed=config.data_random_seed), classification
    elif config.dataset == 'moons':
        classification = True
        return generate_moons(config), classification

    elif config.dataset == 'oned_gaussian':
        classification = True
        return generate_oned_gaussian(), classification

    elif config.dataset == 'twod_gaussian':
        classification = True
        return generate_twod_gaussian(config), classification

    elif config.dataset == 'mnist':
        classification = True
        return generate_mnist(), classification
    
    elif config.dataset == 's_mnist':
        classification = True
        return generate_split_mnist(), classification

    elif config.dataset == 'f_mnist':
        classification = True
        return generate_f_mnist(), classification

    elif config.dataset == 'donuts':
        classification = True
        return generate_donuts(config), classification
    
    elif config.dataset == 'cifar':
        classification = True
        return generate_cifar(), classification
    
    elif config.dataset == 'svhn':
        classification = True
        return generate_svhn(), classification