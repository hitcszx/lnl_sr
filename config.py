import torch
import torch.nn as nn
from losses import *



import torch
import torch.nn as nn
from losses import *


MNIST_CONFIG = {
    "CE": nn.CrossEntropyLoss(),
    "FL": FocalLoss(gamma=0.5),
    "MAE": MAELoss(num_classes=10),
    "GCE": GCELoss(num_classes=10, q=0.01),
    "SCE": SCELoss(num_classes=10),
    # "NLNL": NLNL(train_loader, num_classes=10),
    "NFL": NormalizedFocalLoss(gamma=0.5, num_classes=10),
    "NGCE": NGCELoss(num_classes=10),
    "NCE": NCELoss(num_classes=10),
    "NFL+RCE": NFLandRCE(alpha=1, beta=100, num_classes=10, gamma=0.5),
    "NCEandMAE": NCEandMAE(alpha=1, beta=100, num_classes=10),
    "NCEandRCE": NCEandRCE(alpha=1, beta=100, num_classes=10),
}

CIFAR10_CONFIG = {
    "CE": nn.CrossEntropyLoss(),
    "FL": FocalLoss(gamma=0.5),
    "MAE": MAELoss(num_classes=10),
    "GCE": GCELoss(num_classes=10, q=0.01),
    "SCE": SCELoss(num_classes=10, a=0.1, b=1),
    # "NLNL": NLNL(train_loader, num_classes=10),
    "NFL": NormalizedFocalLoss(gamma=0.5, num_classes=10),
    "NGCE": NGCELoss(num_classes=10),
    "NCE": NCELoss(num_classes=10),
    "NFL+RCE": NFLandRCE(alpha=1, beta=1, num_classes=10, gamma=0.5),
    "NCEandMAE": NCEandMAE(alpha=1, beta=1, num_classes=10),
    "NCEandRCE": NCEandRCE(alpha=1, beta=1, num_classes=10),
}

CIFAR100_CONFIG = {
    "CE": nn.CrossEntropyLoss(),
    "FL": FocalLoss(gamma=0.5),
    "MAE": MAELoss(num_classes=100),
    "GCE": GCELoss(num_classes=100, q=0.001),
    "SCE": SCELoss(num_classes=100, a=6, b=0.1),
    # "NLNL": NLNL(train_loader, num_classes=10),
    "NFL": NormalizedFocalLoss(gamma=0.5, num_classes=100),
    "NGCE": NGCELoss(num_classes=100),
    "NCE": NCELoss(num_classes=100),
    "NFL+RCE": NFLandRCE(alpha=10, beta=1, num_classes=100, gamma=0.5),
    "NCEandMAE": NCEandMAE(alpha=10, beta=1, num_classes=100),
    "NCEandRCE": NCEandRCE(alpha=10, beta=1, num_classes=100),
}

def get_loss_config(dataset, train_loader, num_classes, loss='CE', is_sparse=True):
    if loss == 'GCE' and not is_sparse:
        return GCELoss(num_classes=num_classes)
    if dataset == 'MNIST':
        if loss == 'NLNL':
            return NLNL(train_loader, num_classes=10)
        elif loss in MNIST_CONFIG:
            return MNIST_CONFIG[loss]
        else:
            raise ValueError('Not Implemented')
    if dataset == 'CIFAR10':
        if loss == 'NLNL':
            return NLNL(train_loader, num_classes=10)
        elif loss in CIFAR10_CONFIG:
            return CIFAR10_CONFIG[loss]
        else:
            raise ValueError('Not Implemented')
    if dataset == 'CIFAR100':
        if loss == 'NLNL':
            return NLNL(train_loader, num_classes=100)
        elif loss in CIFAR100_CONFIG:
            return CIFAR100_CONFIG[loss]
        else:
            raise ValueError('Not Implemented')


def get_mnist_exp_criterions_and_names(num_classes):
    return list(MNIST_CONFIG.keys()), list(MNIST_CONFIG.values())

def get_cifar10_exp_criterions_and_names(num_classes, train_loader=None):
    return list(CIFAR10_CONFIG.keys()), list(CIFAR10_CONFIG.values())

def get_cifar100_exp_criterions_and_names(num_classes, train_loader):
    return list(CIFAR100_CONFIG.keys()), list(CIFAR100_CONFIG.values())



MNIST_params = {
    'CE+SR': (0.1, 0.1, 4, 2, 5),
    'FL+SR': (0.1, 0.1, 4, 2, 5),
    'GCE+SR': (0.5, 0.1, 3, 2, 5)
}
CIFAR10_params = {
    'CE+SR': (0.5, 0.1, 1.2, 1.03, 1),
    'FL+SR': (0.5, 0.1, 1.2, 1.03, 1),
    'GCE+SR': (0.5, 0.1, 1.2, 1.03, 1),
}
CIFAR100_params = {
    'CE+SR': (0.5, 0.01, 10, 1.02, 1),
    'FL+SR': (0.5, 0.01, 10, 1.02, 1),
    'GCE+SR': (0.5, 0.01, 10, 1.02, 1),
}

def get_params_sr(dataset, label):
    if label.endswith('+SR'):
        if dataset == 'MNIST':
            return MNIST_params[label]
        elif dataset == 'CIFAR10':
            return CIFAR10_params[label]
        elif dataset == 'CIFAR100':
            return CIFAR100_params[label]
    else:
        return 0, 0, 0, 0, 0
