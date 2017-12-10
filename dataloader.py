import os
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.autograd import Variable, grad
from torch.utils.data import DataLoader

from utils import *



class dataloader():
    def __init__(self, dataset, datadir, batch_size):
        self.dataset = dataset
        self.datadir = datadir
        self.batch_size = batch_size

    def load(self):
        # load dataset
        if self.dataset == 'mnist':
            datadir = os.path.join(self.datadir, 'mnist')
            mnist_loader = DataLoader(datasets.MNIST(datadir,
                                                        train=True,
                                                        download=True,
                                                        transform=transforms.Compose(
                                                             [transforms.ToTensor()])),
                                          batch_size=self.batch_size, shuffle=True)
            return mnist_loader

        elif self.dataset == 'fashion-mnist':
            datadir = os.path.join(self.datadir, 'fashion-mnist')
            fashion_mnist = DataLoader(datasets.FashionMNIST(datadir,
                                                                train=True,
                                                                download=True,
                                                                transform=transforms.Compose(
                                                                [transforms.ToTensor()])),
                                                                batch_size=self.batch_size, shuffle=True)
            return fashion_mnist

        elif self.dataset == 'celebA':
            datadir = os.path.join(self.datadir, 'celebA')
            celebA_loader = load_celebA(datadir,
                                                transform=transforms.Compose(
                                                [transforms.CenterCrop(160), 
                                                    transforms.Resize(64), 
                                                    transforms.ToTensor()]), 
                                                batch_size=self.batch_size,
                                                shuffle=True)
            return celebA_loader

        elif self.dataset == 'small-imagenet':
            # Data loading code
            small_imagenet_dir = os.path.join(self.datadir, 'small-imagenet/')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            print('Loading training data')
            small_imagenet_dataset = datasets.ImageFolder(
                small_imagenet_dir,
                transforms.Compose([
                    transforms.Resize(64),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

            train_sampler = None

            small_imagenet_loader = torch.utils.data.DataLoader(small_imagenet_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=True,
                                                       num_workers=self.nThreads,
                                                       pin_memory=True,
                                                       sampler=train_sampler)
            return small_imagenet_loader

        elif self.dataset == 'imagenet':

            full_imagenet_dir = os.path.join(args.datadir, 'imagenet/')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])


            full_imagenet = datasets.ImageFolder(full_imagenet_dir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))


            full_imagenet_loader = torch.utils.data.DataLoader(full_imagenet,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        num_workers=args.workers,
                                                        pin_memory=True)
            return full_imagenet_loader
