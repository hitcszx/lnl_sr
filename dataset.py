from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from numpy.testing import assert_array_almost_equal
import numpy as np
import os
import torch
import random
from data.mnist import MNIST
from data.cifar import CIFAR10, CIFAR100


def build_for_cifar100(size, noise):
    """ random flip between two random classes.
    """
    assert(noise >= 0.) and (noise <= 1.)

    P = (1. - noise) * np.eye(size)
    for i in np.arange(size - 1):
        P[i, i+1] = noise

    # adjust last row
    P[size-1, 0] = noise

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def other_class(n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class


class MNISTNoisy(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True, nosiy_rate=0.0, asym=False, seed=0):
        super(MNISTNoisy, self).__init__(root, transform=transform, target_transform=target_transform, download=download)
        self.targets = self.targets.numpy()
        if asym:
            P = np.eye(10)
            n = nosiy_rate

            P[7, 7], P[7, 1] = 1. - n, n
            # 2 -> 7
            P[2, 2], P[2, 7] = 1. - n, n

            # 5 <-> 6
            P[5, 5], P[5, 6] = 1. - n, n
            P[6, 6], P[6, 5] = 1. - n, n

            # 3 -> 8
            P[3, 3], P[3, 8] = 1. - n, n

            y_train_noisy = multiclass_noisify(self.targets, P=P, random_state=seed)
            actual_noise = (y_train_noisy != self.targets).mean()
            assert actual_noise > 0.0
            print('Actual noise %.2f' % actual_noise)
            self.targets = y_train_noisy

        else:
            n_samples = len(self.targets)
            n_noisy = int(nosiy_rate * n_samples)
            print("%d Noisy samples" % (n_noisy))
            class_index = [np.where(np.array(self.targets) == i)[0] for i in range(10)]
            class_noisy = int(n_noisy / 10)
            noisy_idx = []
            for d in range(10):
                noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                noisy_idx.extend(noisy_class_index)
                print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
            for i in noisy_idx:
                self.targets[i] = other_class(n_classes=10, current_class=self.targets[i])
            print(len(noisy_idx))

        print("Print noisy label generation statistics:")
        for i in range(10):
            n_noisy = np.sum(np.array(self.targets) == i)
            print("Noisy class %s, has %s samples." % (i, n_noisy))

        return


class cifar10Nosiy(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True, nosiy_rate=0.0, asym=False):
        super(cifar10Nosiy, self).__init__(root, download=download, transform=transform, target_transform=target_transform)
        self.download = download
        if asym:
            # automobile < - truck, bird -> airplane, cat <-> dog, deer -> horse
            source_class = [9, 2, 3, 5, 4]
            target_class = [1, 0, 5, 3, 7]
            for s, t in zip(source_class, target_class):
                cls_idx = np.where(np.array(self.targets) == s)[0]
                n_noisy = int(nosiy_rate * cls_idx.shape[0])
                noisy_sample_index = np.random.choice(cls_idx, n_noisy, replace=False)
                for idx in noisy_sample_index:
                    self.targets[idx] = t
            return
        elif nosiy_rate > 0:
            n_samples = len(self.targets)
            n_noisy = int(nosiy_rate * n_samples)
            print("%d Noisy samples" % (n_noisy))
            class_index = [np.where(np.array(self.targets) == i)[0] for i in range(10)]
            class_noisy = int(n_noisy / 10)
            noisy_idx = []
            for d in range(10):
                noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                noisy_idx.extend(noisy_class_index)
                print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
            for i in noisy_idx:
                self.targets[i] = other_class(n_classes=10, current_class=self.targets[i])
            print(len(noisy_idx))
            print("Print noisy label generation statistics:")
            for i in range(10):
                n_noisy = np.sum(np.array(self.targets) == i)
                print("Noisy class %s, has %s samples." % (i, n_noisy))
            return


class cifar100Nosiy(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True, nosiy_rate=0.0, asym=False, seed=0):
        super(cifar100Nosiy, self).__init__(root, download=download, transform=transform, target_transform=target_transform)
        self.download = download
        if asym:
            """mistakes are inside the same superclass of 10 classes, e.g. 'fish'
            """
            nb_classes = 100
            P = np.eye(nb_classes)
            n = nosiy_rate
            nb_superclasses = 20
            nb_subclasses = 5

            if n > 0.0:
                for i in np.arange(nb_superclasses):
                    init, end = i * nb_subclasses, (i+1) * nb_subclasses
                    P[init:end, init:end] = build_for_cifar100(nb_subclasses, n)

                    y_train_noisy = multiclass_noisify(np.array(self.targets), P=P, random_state=seed)
                    actual_noise = (y_train_noisy != np.array(self.targets)).mean()
                assert actual_noise > 0.0
                print('Actual noise %.2f' % actual_noise)
                self.targets = y_train_noisy.tolist()
            return
        elif nosiy_rate > 0:
            n_samples = len(self.targets)
            n_noisy = int(nosiy_rate * n_samples)
            print("%d Noisy samples" % (n_noisy))
            class_index = [np.where(np.array(self.targets) == i)[0] for i in range(100)]
            class_noisy = int(n_noisy / 100)
            noisy_idx = []
            for d in range(100):
                noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                noisy_idx.extend(noisy_class_index)
                print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
            for i in noisy_idx:
                self.targets[i] = other_class(n_classes=100, current_class=self.targets[i])
            print(len(noisy_idx))
            print("Print noisy label generation statistics:")
            for i in range(100):
                n_noisy = np.sum(np.array(self.targets) == i)
                print("Noisy class %s, has %s samples." % (i, n_noisy))
            return



class DatasetGenerator():
    def __init__(self,
                 train_batch_size=128,
                 eval_batch_size=256,
                 data_path='data/',
                 seed=123,
                 num_of_workers=4,
                 asym=False,
                 dataset_type='CIFAR10',
                 is_cifar100=False,
                 cutout_length=16,
                 noise_rate=0.4):
        self.seed = seed
        np.random.seed(seed)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.data_path = data_path
        self.num_of_workers = num_of_workers
        self.cutout_length = cutout_length
        self.noise_rate = noise_rate
        self.dataset_type = dataset_type
        self.asym = asym
        if self.asym:
            self.noise_type = 'asymmetric'
        else:
            self.noise_type = 'symmetric'
        if self.noise_rate == 0:
            self.noise_type = 'clean'

        self.data_loaders = self.loadData()
        return

    def getDataLoader(self):
        return self.data_loaders

    def loadData(self):
        if self.dataset_type == 'MNIST':
            MEAN = [0.1307]
            STD = [0.3081]
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD)])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD)])
            train_dataset = MNIST(root=self.data_path+'/MNIST',
                                  train=True,
                                  transform=train_transform,
                                  download=False,
                                  noise_type=self.noise_type,
                                  noise_rate=self.noise_rate,
                                  )
            test_dataset = MNIST(
                root=self.data_path+'/MNIST',
                train=False,
                transform=test_transform,
                download=False,
                noise_type=self.noise_type,
                noise_rate=self.noise_rate,
            )
            # train_dataset = MNISTNoisy(root=self.data_path,
            #                            train=True,
            #                            transform=train_transform,
            #                            download=False,
            #                            asym=self.asym,
            #                            seed=self.seed,
            #                            nosiy_rate=self.noise_rate)
            #
            # test_dataset = datasets.MNIST(root=self.data_path,
            #                               train=False,
            #                               transform=test_transform,
            #                               download=False)

        elif self.dataset_type == 'CIFAR100':
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

            train_dataset = cifar100Nosiy(root=self.data_path,
                                          train=True,
                                          transform=train_transform,
                                          download=True,
                                          asym=self.asym,
                                          seed=self.seed,
                                          nosiy_rate=self.noise_rate)

            test_dataset = datasets.CIFAR100(root=self.data_path,
                                             train=False,
                                             transform=test_transform,
                                             download=True)

        elif self.dataset_type == 'CIFAR10':
            CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
            CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

            train_dataset = cifar10Nosiy(root=self.data_path,
                                         train=True,
                                         transform=train_transform,
                                         download=True,
                                         asym=self.asym,
                                         nosiy_rate=self.noise_rate)

            test_dataset = datasets.CIFAR10(root=self.data_path,
                                            train=False,
                                            transform=test_transform,
                                            download=True)
        else:
            raise("Unknown Dataset")

        data_loaders = {}

        data_loaders['train_dataset'] = DataLoader(dataset=train_dataset,
                                                   batch_size=self.train_batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=self.num_of_workers)

        data_loaders['test_dataset'] = DataLoader(dataset=test_dataset,
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=self.num_of_workers)

        print("Num of train %d" % (len(train_dataset)))
        print("Num of test %d" % (len(test_dataset)))

        return data_loaders


class WebVisionDataset:
    def __init__(self, path, file_name='webvision_mini_train', transform=None, target_transform=None):
        self.target_list = []
        self.path = path
        self.load_file(os.path.join(path, file_name))
        self.transform = transform
        self.target_transform = target_transform
        return

    def load_file(self, filename):
        f = open(filename, "r")
        for line in f:
            train_file, label = line.split()
            self.target_list.append((train_file, int(label)))
        f.close()
        return

    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, index):
        impath, target = self.target_list[index]
        img = Image.open(os.path.join(self.path, impath)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class ImageNetMiniVal:
    def __init__(self, path, file_name='ILSVRC2012_mini_val.txt', transform=None, target_transform=None):
        self.target_list = []
        self.path = path
        self.load_file(os.path.join(path, file_name))
        self.transform= transform
        self.target_transform = target_transform
        return

    def load_file(self, filename):
        f = open(filename, "r")
        for line in f:
            train_file, label = line.split()
            self.target_list.append((train_file, int(label)))
        f.close()
        return
    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, index):
        impath, target = self.target_list[index]
        img = Image.open(os.path.join(self.path, impath)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class WebVisionDatasetLoader:
    def __init__(self, setting='mini', train_batch_size=128, eval_batch_size=256, train_data_path='data/', valid_data_path='data/', num_of_workers=4):
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path
        self.num_of_workers = num_of_workers
        self.setting = setting
        self.data_loaders = self.loadData()

    def getDataLoader(self):
        return self.data_loaders

    def loadData(self):
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ColorJitter(brightness=0.4,
                                                                     contrast=0.4,
                                                                     saturation=0.4,
                                                                     hue=0.2),
                                              transforms.ToTensor(),
                                              transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

        test_transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

        if self.setting == 'mini':
            train_dataset = WebVisionDataset(path=self.train_data_path,
                                             file_name='webvision_mini_train.txt',
                                             transform=train_transform)

            test_dataset = WebVisionDataset(path=self.valid_data_path,
                                           file_name='webvision_mini_val.txt',
                                           transform=test_transform)

        elif self.setting == 'full':
            train_dataset = WebVisionDataset(path=self.train_data_path,
                                             file_name='train_filelist_google.txt',
                                             transform=train_transform)

            test_dataset = WebVisionDataset(path=self.valid_data_path,
                                            file_name='val_filelist.txt',
                                            transform=test_transform)

        elif self.setting == 'full_imagenet':
            train_dataset = WebVisionDataset(path=self.train_data_path,
                                             file_name='train_filelist_google',
                                             transform=train_transform)

            test_dataset = datasets.ImageNet(root=self.valid_data_path,
                                             split='val',
                                             transform=test_transform)

        else:
            raise(NotImplementedError)

        data_loaders = {}

        print('Training Set Size %d' % (len(train_dataset)))
        print('Test Set Size %d' % (len(test_dataset)))

        data_loaders['train_dataset'] = DataLoader(dataset=train_dataset,
                                                   batch_size=self.train_batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=self.num_of_workers)

        data_loaders['test_dataset'] = DataLoader(dataset=test_dataset,
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=self.num_of_workers)

        return data_loaders

def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for data, _ in tqdm(loader):

        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

if __name__ == '__main__':
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ColorJitter(brightness=0.4,
                                                                 contrast=0.4,
                                                                 saturation=0.4,
                                                                 hue=0.2),
                                          transforms.ToTensor(),
                                          transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    a = WebVisionDataset(path='../database/WebVision', file_name='webvision_mini_val.txt',transform=test_transform)