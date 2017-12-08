import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, dataset='imagenet', arch='infogan'):
        super(discriminator, self).__init__()
        if dataset == 'mnist' or dataset == 'fashion-mnist':
            self.input_height = 28
            self.input_width = 28
            self.input_channel = 1
            self.output_channel = 1
        elif dataset == 'celebA':
            self.input_height = 64
            self.input_width = 64
            self.input_channel = 3
            self.output_channel = 1
        elif dataset == 'imagenet':
            self.input_height = 64
            self.input_width = 64
            self.input_channel = 3
            self.output_channel = 1

        # set up the architecture for discriminator
        self.arch = arch

        if self.arch == 'infogan':
            self.conv = nn.Sequential(
                nn.Conv2d(self.input_channel, 64, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
            )
            self.fc = nn.Sequential(
                nn.Linear(128 * (self.input_height // 4) * (self.input_width // 4), 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.2),
                nn.Linear(1024, self.output_channel),
                nn.Sigmoid(),
            )
            utils.initialize_weights(self)

    def forward(self, input):
        output = self.conv(input)
        output = output.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4))
        output = self.fc(output)
        return output
