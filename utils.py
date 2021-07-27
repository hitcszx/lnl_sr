
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


def log(p='\n', f=None):
    if f is None:
        print(p)
    else:
        f.write(p + '\n')


def get_transforms(dataset):
    if dataset == 'MNIST':
        MEAN = [0.1307]
        STD = [0.3081]
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    elif dataset == 'CIFAR10':
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])
    elif dataset == 'CIFAR100':
        CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
        CIFAR_STD = [0.2673, 0.2564, 0.2762]

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    else:
        raise ValueError('Invalid value {}'.format(dataset))

    return train_transform, test_transform

def save_accs(path, label, accs):
    with open(os.path.join(path, label+'.csv'), 'w') as f:
        m = accs.shape[0]
        f.write(','.join(['test ' + str(i+1) for i in range(m)]) + '\n')
        for i in range(accs.shape[1]):
            f.write(','.join([str(f) for f in accs[:,i]]) + '\n')

def save_acc(path, label, accs):
    with open(os.path.join(path, label+'.csv'), 'w') as f:
        for a in accs:
            f.write(str(a) + '\n')

