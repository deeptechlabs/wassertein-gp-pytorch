import utils 
import torch
import torch.nn as nn
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import *

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


class ResnetGenerator(nn.Module):
    def __init__(self, ngpu, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.ngpu = ngpu
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        #print("test")
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.model, input, range(self.ngpu))
        else:
            output = self.model(input)
        return output



# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, ngpu, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        self.ngpu = ngpu

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.model, input, range(self.ngpu))
        else:
            output = self.model(input)
        return output


def build_generator(ngpu, input_nc=3, output_nc=3, ngf=64, which_model_netG='resnet_9blocks', norm='batch', use_dropout=False, init_type='normal'):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    print("building generator")

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(ngpu, input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(ngpu, input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(ngpu, input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(ngpu, input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    init_weights(netG, init_type=init_type)
    return netG
