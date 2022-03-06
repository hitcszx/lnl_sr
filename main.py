
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import resnet34
from dataset import DatasetGenerator
from models import *
from losses import *
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import random
from utils import *
from config import *
from norm import pNorm

import matplotlib.pyplot as plt
import seaborn as sb
sb.set_style('darkgrid')
plt.switch_backend('agg')
plt.figure(figsize=(20, 20), dpi=600)


parser = argparse.ArgumentParser(description='Robust loss for learning with noisy labels')
parser.add_argument('--dataset', type=str, default="MNIST", metavar='DATA', help='Dataset name (default: CIFAR10)')
parser.add_argument('--root', type=str, default="../database/", help='the data root')
parser.add_argument('--noise_type', type=str, default='symmetric', help='the noise type: clean, symmetric, pairflip, asymmetric')
parser.add_argument('--noise_rate', type=float, default=0.8, help='the noise rate')
parser.add_argument('--gpus', type=str, default='1')
# learning settings
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_workers', type=int, default=10, help='the number of worker for loading data')
parser.add_argument('--grad_bound', type=float, default=5., help='the gradient norm bound')
parser.add_argument('--seed', type=int, default=123)


parser.add_argument('--is_sparse', type=int, default=1, help='if use the sparse regularizatoin mechanism')
parser.add_argument('--loss', type=str, default='GCE', help='the loss functions: CE, FL, GCE')

args = parser.parse_args()

if args.is_sparse:
    args.is_sparse = True
    label = args.loss + '+SR'
else:
    args.is_sparse = False
    label = args.loss

if args.noise_rate == 0.0:
    args.noise_type = 'clean'

if args.noise_type == 'asymmetric':
    asymm = True
else:
    asymm = False


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
device = 'cuda' if torch.cuda.is_available()  else 'cpu'
print('We are using', device)


if device == 'cuda':
    torch.cuda.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)


seed = 123
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

print(args)

def evaluate(loader, model):
    model.eval()
    correct = 0.
    total = 0.
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        z = model(x)
        probs = F.softmax(z, dim=1)
        pred = torch.argmax(probs, 1)
        total += y.size(0)
        correct += (pred==y).sum().item()

    acc = float(correct) / float(total)
    return acc

def calculate_loss(criterion, out, y, norm=None, lamb=None, tau=None, p=None):
    if args.is_sparse:
        if args.dataset != 'MNIST':
            out = F.normalize(out, dim=1)
        loss = criterion(out / tau, y) + lamb * norm(out / tau, p)
    else:
        loss = criterion(out, y)
    return loss


data_loader = DatasetGenerator(data_path=os.path.join(args.root, args.dataset),
                               num_of_workers=args.num_workers,
                               seed=args.seed,
                               asym=args.noise_type=='asymmetric',
                               dataset_type=args.dataset,
                               noise_rate=args.noise_rate
                               )

data_loader = data_loader.getDataLoader()
train_loader = data_loader['train_dataset']
test_loader = data_loader['test_dataset']
tau, p, lamb, rho, freq = get_params_sr(args.dataset, label)

if args.dataset == 'MNIST':
    in_channels = 1
    num_classes = 10
    weight_decay = 1e-3
    lr = 0.01
    epochs = 50
elif args.dataset == 'CIFAR10':
    in_channels = 3
    num_classes = 10
    weight_decay = 1e-4
    lr = 0.01
elif args.dataset == 'CIFAR100':
    in_channels = 3
    num_classes = 100
    weight_decay = 1e-5
    lr = 0.1
    epochs = 200
    lamb = 10 if asymm else 4
else:
    raise ValueError('Invalid value {}'.format(args.dataset))

criterion = get_loss_config(args.dataset, train_loader, num_classes=num_classes, loss=args.loss, is_sparse=args.is_sparse)
if args.is_sparse:
    norm = pNorm(p)
else:
    norm = None
print(label)

if args.dataset != 'CIFAR100':
    model = CNN(type=args.dataset).to(device)
else:
    model = ResNet34(num_classes=100).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
# scheduler = StepLR(optimizer, gamma=0.1, step_size=25)
for ep in range(epochs):
    model.train()
    total_loss = 0.
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        model.zero_grad()
        optimizer.zero_grad()
        out = model(batch_x)
        loss = calculate_loss(criterion, out, batch_y, norm, lamb, tau, p)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_bound)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    test_acc = evaluate(test_loader, model)
    log('Iter {}: loss={:.4f}, test_acc={:.4f}'.format(ep, total_loss, test_acc))
    if (ep + 1) % freq == 0:
        lamb = lamb * rho
