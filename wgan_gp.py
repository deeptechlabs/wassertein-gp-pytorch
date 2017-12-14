import utils
import torch
import time
import os
import pickle
import numpy as np
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import datetime

from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from visdom import Visdom
from utils.inception_score import inception_score
from sklearn.preprocessing import scale
from dataloader import dataloader

from generator import *
from discriminator import *

class WGAN_GP(object):
    def __init__(self, args):
        # parameters
        self.model_name = args.model_name
        self.epoch = args.epoch
        self.sample_num = args.sample_num
        self.z_dim = args.z_dim
        self.input_channels = args.input_channels
        self.output_channels = args.output_channels
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.datadir = args.datadir
        self.log_dir = args.log_dir
        self.generator_arch = args.generator
        self.discriminator_arch = args.discriminator
        self.nThreads = args.nThreads
        self.gpu_mode = args.gpu_mode
        self.lambda_ = args.lambda_grad_penalty
        self.n_critic = args.n_critic
        self.visualize = args.visualize 
        self.ngpu = args.ngpu
        self.env_display = str(args.dataset)
        self.vis = Visdom(server=args.visdom_server, port=args.visdom_port)
        self.calculate_inception = args.calculate_inception

        # networks init
        if self.generator_arch == 'infogan':
            self.G = INFOGAN_generator(nc=self.input_channels, ngf=self.z_dim)
        elif self.generator_arch == 'dcgan':
            self.G = DCGAN_generator(self.ngpu, nc=self.input_channels, ngf=self.z_dim)
        elif self.generator_arch == 'resnet_6blocks':
            self.G = build_generator(self.ngpu, ngf=self.z_dim, which_model_netG='resnet_6blocks')
        elif self.generator_arch == 'resnet_9blocks':
            self.G = build_generator(self.ngpu, ngf=self.z_dim, which_model_netG='resnet_9blocks')
        elif self.generator_arch == 'unet_128':
            self.G = build_generator(self.ngpu, ngf=self.z_dim, which_model_netG='unet_128')
        elif self.generator_arch == 'unet_256':
            self.G = build_generator(self.ngpu, ngf=self.z_dim, which_model_netG='unet_256')

        if self.discriminator_arch == 'infogan':
            self.D = INFOGAN_discriminator(nc=self.input_channels, ndf=self.z_dim)
        elif self.discriminator_arch == 'dcgan':
            self.D = DCGAN_discriminator(self.ngpu, nc=self.input_channels, ndf=self.z_dim)
        elif self.discriminator_arch == 'basic':
            self.D = build_discriminator(self.ngpu, ndf=self.z_dim, which_model_netD='basic')
        elif self.discriminator_arch == 'n_layers':
            self.D = build_discriminator(self.ngpu, ndf=self.z_dim, which_model_netD='n_layers')
        elif self.discriminator_arch == 'pixel':
            self.D = build_discriminator(self.ngpu, ndf=self.z_dim, which_model_netD='pixel')

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        data = dataloader(self.dataset, self.datadir, self.batch_size, self.nThreads)

        self.data_loader = data.load()

        print("Data loaded")

        #self.z_dim = 512

        # fixed noise
        if self.gpu_mode:
            self.sample_z_ = Variable(torch.rand((self.batch_size, self.input_channels, self.batch_size, self.batch_size)).cuda(), volatile=True)
            #self.sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim, 1, 1)).cuda(), volatile=True)
        else:
            self.sample_z_ = Variable(torch.rand((self.batch_size, self.input_channels, self.batch_size, self.batch_size)), volatile=True)
            #self.sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim, 1, 1)), volatile=True)

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = Variable(torch.ones((self.batch_size, self.input_channels, self.batch_size, self.batch_size)).cuda()), Variable(torch.zeros((self.batch_size, self.input_channels, self.batch_size, self.batch_size)).cuda())
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones((self.batch_size, self.input_channels, self.batch_size, self.batch_size))), Variable(torch.zeros((self.batch_size, self.input_channels, self.batch_size, self.batch_size)))

        self.D.train()

        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, (x_, _) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break


                z_ = torch.rand((self.batch_size, self.input_channels, self.batch_size, self.batch_size))
                #z_ = torch.rand((self.batch_size, self.z_dim, 1, 1))
                #z_ = torch.rand(self.batch_size, 3)

                if self.gpu_mode:
                    x_, z_ = Variable(x_.cuda()), Variable(z_.cuda())
                else:
                    x_, z_ = Variable(x_), Variable(z_)
                #print(x_.size())
                #print(z_.size())
                # update D network
                self.D_optimizer.zero_grad()

                D_real = self.D(x_)
                D_real_loss = -torch.mean(D_real)

                #print(z_)

                G_ = self.G(z_)
                #print(G_)
                D_fake = self.D(G_)
                D_fake_loss = torch.mean(D_fake)

                # gradient penalty
                if self.gpu_mode:
                    alpha = torch.rand(x_.size()).cuda()
                else:
                    alpha = torch.rand(x_.size())
                # the only reason the imagenet might fail 
                # is that the miss dimension

                x_hat = Variable(alpha * x_.data + (1 - alpha) * G_.data, requires_grad=True)

                pred_hat = self.D(x_hat)
                if self.gpu_mode:
                    gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                                 create_graph=True, retain_graph=True, only_inputs=True)[0]
                else:
                    gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()),
                                     create_graph=True, retain_graph=True, only_inputs=True)[0]

                gradient_penalty = self.lambda_ * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

                D_loss = D_real_loss + D_fake_loss + gradient_penalty

                D_loss.backward()
                self.D_optimizer.step()

                if ((iter+1) % self.n_critic) == 0:
                    # update G network
                    self.G_optimizer.zero_grad()

                    G_ = self.G(z_)
                    D_fake = self.D(G_)
                    G_loss = -torch.mean(D_fake)
                    self.train_hist['G_loss'].append(G_loss.data[0])

                    G_loss.backward()
                    self.G_optimizer.step()

                    self.train_hist['D_loss'].append(D_loss.data[0])

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.data[0], G_loss.data[0]))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            self.visualize_results((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
                                 self.epoch)
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            if self.gpu_mode:
                sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim)).cuda(), volatile=True)
            else:
                sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim)), volatile=True)

            samples = self.G(sample_z_)

        copy_samples = samples

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
            presamples = copy_samples.cpu().data.numpy()
            presamples = 2*(presamples - np.max(presamples))/-np.ptp(presamples)-1
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)
            presamples = copy_samples.data.numpy()
            presamples = 2*(presamples - np.max(presamples))/-np.ptp(presamples)-1

        images = samples[:image_frame_dim * image_frame_dim, :, :, :]

        # Calculate inception score
        if self.calculate_inception:

            score = inception_score(presamples,
                                    cuda=True,
                                    resize=True,
                                    batch_size=self.batch_size)

            # display inception mean and std

            print("Inception score mean and std are", score)

        # save images and display if set
        utils.save_images(self.vis, images, [image_frame_dim, image_frame_dim], 
                        self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' 
                        + self.model_name + '_epoch%03d' % epoch + '.png', 
                        env=self.env_display,
                        visualize=self.visualize)


    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))
