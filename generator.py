import utils 
import torch
import torch.nn as nn
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class INFOGAN_generator(nn.Module):
    def __init__(self, dataset='small-imagenet', z_dim=512):
        super(INFOGAN_generator, self).__init__()
        self.dataset = dataset
        self.z_dim = z_dim
        if self.dataset == 'mnist' or self.dataset == 'fashion-mnist':
            self.output_height = 28
            self.output_width = 28
            self.input_channel = z_dim
            self.output_channel = 1 # if black and while it's 1
        elif self.dataset == 'celebA':
            self.output_height = 64
            self.output_width = 64
            self.input_channel = z_dim
            self.output_channel = 3
        elif self.dataset == 'imagenet':
            self.input_height = 1
            self.input_width = 1
            self.input_channel = self.z_dim # this is hidden size
            self.output_height = 64 # this is size of height
            self.output_width = 64 # this is size of image w
            self.output_channel = 3 # this is color
        elif self.dataset == 'small-imagenet':
            self.input_height = 1
            self.input_width = 1
            self.input_channel = self.z_dim # this is hidden size
            self.output_height = 64 # this is size of height
            self.output_width = 64 # this is size of image w
            self.output_channel = 3 # this is color


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
        if self.dataset == 'mnist':
            output = self.fc(input)

        if self.dataset == 'fashion-mnist':
            output = self.fc(input)

        elif self.dataset == 'celebA':
            output = self.fc(input)

        elif self.dataset == 'small-imagenet':
            output = input.view(-1, self.input_channel * self.input_height * self.input_width)
            output = self.fc(output)

        elif self.dataset == 'imagenet':
            output = input.view(-1, self.input_channel * self.input_height * self.input_width)
            output = self.fc(output)

        output = output.view(-1, 128, (self.output_height // 4), (self.output_width // 4))
        output = self.deconv(output)
        return output


class DCGAN_generator(nn.Module):
    def __init__(self, ngpu, n_zim=512, output_channel=3, input_dimensions=64):
        super(DCGAN_generator, self).__init__()
        self.ngpu = ngpu
        self.nz = n_zim # latent variable
        self.ngf = input_dimensions
        self.nc = output_channel
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2,     self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    self.ngf,      self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        input = input.view(-1, self.nz, 1, 1)
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
