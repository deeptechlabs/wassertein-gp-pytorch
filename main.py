import argparse
import os

from wgan_gp import WGAN_GP


"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default='small-imagenet',
                        choices=['imagenet', 'small-imagenet', 'mnist', 'fashion-mnist', 'celebA'], 
                        help='The name of dataset')
    parser.add_argument('--discriminator', type=str, default='n_layers',
                        choices=['basic','n_layers','pixel',
                        'infogan', 'dcgan'],
                        help='Discriminator architecture')
    parser.add_argument('--generator', type=str, default='resnet_6blocks',
                        choices=['resnet_6blocks','resnet_9blocks', 
                        'unet_128','unet_256','infogan', 'dcgan'], 
                        help='Generator architecture')
    parser.add_argument('--epoch', type=int, default=100, 
                        help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='The size of batch')
    parser.add_argument('--datadir', type=str, default='/data/milatmp1/suhubdyd/datasets/', 
                        help='Directory name to save the model')
    parser.add_argument('--save_dir', type=str, default='/data/milatmp1/suhubdyd/models/gans/', 
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='/data/milatmp1/suhubdyd/models/gans/imagesgen/', 
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='/data/milatmp1/suhubdyd/models/gans/logs/', 
                        help='Directory name to save training logs')
    parser.add_argument('--visualize', type=bool, default=True, 
                        help='Display results on Visdom')
    parser.add_argument('--visdom_server', type=str, default='http://suhubdy.com', 
                        help='Visdom Server to display results')
    parser.add_argument('--visdom_port', type=int, default=51401, 
                        help='Visdom Server port')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--sample_num', type=int, default=64)
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--output_channels', type=int, default=3)
    parser.add_argument('--lambda_grad_penalty', type=float, default=10)
    parser.add_argument('--n_critic', type=float, default=1)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--calculate_inception', type=bool, default=True)
    parser.add_argument('--nThreads', '-j', default=5, type=int, metavar='N',
                        help='number of data loading threads (default: 2)')
    parser.add_argument('--model_name', type=str, default='wgan-gp')
    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --discriminator architecture
    try:
        assert args.discriminator is not None
    except:
        print('You must specify a discriminator architecture')

    # --geenrator architecture
    try:
        assert args.generator is not None
    except:
        print('You must specify a generator architecture')

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    print(args)
    if args is None:
        exit()

    gan = WGAN_GP(args)

    # launch the graph in a session
    gan.train()
    print(" [*] Training finished!")

    # visualize learned generator
    gan.visualize_results(args.epoch)
    print(" [*] Testing finished!")

if __name__ == '__main__':
    main()
