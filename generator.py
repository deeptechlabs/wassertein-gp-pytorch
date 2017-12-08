import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class generator(nn.Module):
    def __init__(self, dataset='imagenet', arch='infogan', z_dim=512):
        super(generator, self).__init__()
        self.dataset = dataset
        self.z_dim = z_dim
        if self.dataset == 'mnist' or self.dataset == 'fashion-mnist':
            self.input_height = 28
            self.input_width = 28
            self.input_channel = 62
            self.output_channel = 1 # if black and while it's 1
        elif self.dataset == 'celebA':
            self.input_height = 64
            self.input_width = 64
            self.input_dim = 62
            self.output_channel = 3
        elif self.dataset == 'imagenet':
            self.input_height = 1
            self.input_width = 1
            self.input_channel = self.z_dim # this is hidden size
            self.output_height = 64 # this is size of height
            self.output_width = 64 # this is size of image w
            self.output_channel = 3 # this is color

        self.arch = arch
        if self.dataset == 'celebA' or self.dataset == 'mnist':
            self.fc = nn.Sequential(
                nn.Linear(self.output_channel, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, 128 * (self.input_height // 4) * (self.input_width // 4)),
                nn.BatchNorm1d(128 * (self.input_height // 4) * (self.input_width // 4)),
                nn.ReLU(),
            )
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, self.output_channel, 4, 2, 1),
                nn.Sigmoid(),
            )
            utils.initialize_weights(self)

        elif self.dataset == 'imagenet':
            self.fc = nn.Sequential(
                nn.Linear(self.input_channel, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, 128 * (self.output_height // 4) * (self.output_width // 4)),
                nn.BatchNorm1d(128 * (self.output_height // 4) * (self.output_width // 4)),
                nn.ReLU(),
            )
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, self.output_channel, 4, 2, 1),
                nn.Sigmoid(),
            )
            utils.initialize_weights(self)

    def forward(self, input):
        '''
        if self.dataset == 'celebA' or self.dataset == 'mnist':
            output = self.fc(input)
            output = output.view(-1, 128, (self.input_height // 4), (self.input_width // 4))
            output = self.deconv(output)
            return output

        elif self.dataset is 'imagenet':'''

        output = input.view(-1, self.input_channel * self.input_height * self.input_width)
        output = self.fc(output)
        output = output.view(-1, 128, (self.output_height // 4), (self.output_width // 4))
        output = self.deconv(output)
        #print(type(output))
        return output
