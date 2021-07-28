
import os
import argparse
import warnings
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
from data.cifar import ImbCIFAR10, ImbCIFAR100
import torchvision.transforms as transforms
import torchvision.datasets as datasets


parser = argparse.ArgumentParser(description='Robust loss for learning with noisy labels')
parser.add_argument('--dataset', type=str, default="CIFAR100", metavar='DATA', help='Dataset name (default: CIFAR10)')
parser.add_argument('--root', type=str, default="../database/", help='the data root')
parser.add_argument('--gpus', type=str, default='1')
# learning settings
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_workers', type=int, default=10, help='the number of worker for loading data')
parser.add_argument('--grad_bound', type=float, default=5., help='the gradient norm bound')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
gpu_ids = ['1']
device = 'cuda' if torch.cuda.is_available() and len(gpu_ids) > 0 else 'cpu'
print('We are using', device)



seed = 123
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


print(args)
def evaluate(loader, model, softsort=None, vector=None):
    model.eval()
    correct = 0.
    total = 0.
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        z = model(x)
        if softsort:
            out = softsort(z)
            # out = sinkhorn(out)
            probs = out.permute(0, 2, 1).matmul(vector).squeeze(-1)
        else:
            probs = F.softmax(z, dim=1)
        pred = torch.argmax(probs, 1)
        total += y.size(0)
        correct += (pred==y).sum().item()

    acc = float(correct) / float(total)
    return acc

if args.dataset == 'CIFAR10':
    in_channels = 3
    num_classes = 10
    weight_decay = 1e-4
    lr = 0.01
    epochs=120
    is_norm = False
    tau = 0.5
    p=0.1
    lamb = 1
    rho = 1.03
    freq = 1
elif args.dataset == 'CIFAR100':
    in_channels = 3
    num_classes = 100
    weight_decay = 1e-5
    lr = 0.1
    epochs=200
    is_norm = True
    tau = 0.5
    p = 0.01
    lamb = 1
    rho = 1.02
    freq = 1
else:
    raise ValueError('Invalid value {}'.format(args.dataset))


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset == 'CIFAR10':
    train_dataset = ImbCIFAR10(root='../database/CIFAR10', imb_type=args.imb_type, imb_factor=args.imb_factor, train=True, download=True, transform=transform_train, seed=args.seed)
    val_dataset = datasets.CIFAR10(root='../database/CIFAR10', train=False, download=True, transform=transform_val)
elif args.dataset == 'CIFAR100':
    train_dataset = ImbCIFAR100(root='../database/CIFAR100', imb_type=args.imb_type, imb_factor=args.imb_factor, train=True, download=True, transform=transform_train, seed=args.seed)
    val_dataset = datasets.CIFAR100(root='../database/CIFAR100', train=False, download=True, transform=transform_val)
else:
    raise ValueError('Not implemented!')


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers, pin_memory=True
)
test_loader = loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=True
)


path = './results/' + args.dataset +'/' + args.imb_type + '/' +str(args.imb_factor)
if not os.path.exists(path):
    os.makedirs(path)

times = 1

criterions = [nn.CrossEntropyLoss()]
labels = ['CE+SR']

for criterion, label in zip(criterions, labels):
    if not label.endswith('+SR'):
        is_norm = False
    accs = np.zeros((times, epochs))
    for i in range(times):
        if args.dataset == 'MNIST':
            model = CNN(type=args.dataset, show=False, norm=False).to(device)
        elif args.dataset == 'CIFAR10':
            model = CNN(type=args.dataset, show=False, norm=False).to(device)
        else:
            model = ResNet34(num_classes=100).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
        norm = pNorm(p=p)
        for ep in range(epochs):
            model.train()
            total_loss = 0.
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                model.zero_grad()
                optimizer.zero_grad()
                out = model(batch_x)
                if label.endswith('+SR'):
                    if is_norm:
                        out = F.normalize(out, dim=1)
                    loss = criterion(out / tau, batch_y) + lamb * norm(out / tau)
                else:
                    loss = criterion(out, batch_y)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_bound)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            test_acc = evaluate(test_loader, model)
            accs[i, ep] = test_acc
            log('Iter {}: loss={:.4f}, test_acc={:.4f}'.format(ep, total_loss, test_acc))
            if (ep + 1) % freq == 0:
                lamb = lamb * rho
    save_accs(path, label, accs)
    print('The validation accuracy is %.2f' % (100 * test_acc))