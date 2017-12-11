import utils
import torch
import torch.nn as nn

from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class INFOGAN_discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, dataset='small-imagenet'):
        super(INFOGAN_discriminator, self).__init__()
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
        elif dataset == 'small-imagenet':
            self.input_height = 64
            self.input_width = 64
            self.input_channel = 3
            self.output_channel = 1



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


class DCGAN_discriminator(nn.Module):
    def __init__(self, ngpu, input_channels=3, input_dimensions=64):
        super(DCGAN_discriminator, self).__init__()
        self.ngpu = ngpu
        self.nc = input_channels
        self.ndf = input_dimensions
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)
