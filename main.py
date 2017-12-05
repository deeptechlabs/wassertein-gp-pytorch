import argparse
import os

from wgan_gp import WGAN_GP


"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['imagenet', 'mnist', 'fashion-mnist', 'celebA'], 
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=25, 
                        help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='The size of batch')
    parser.add_argument('--datadir', type=str, default='/data/milatmp1/suhubdyd/datasets/imagenet64/', 
                        help='Directory name to save the model')
    parser.add_argument('--save_dir', type=str, default='/data/milatmp1/suhubdyd/models/gans/', 
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='/data/milatmp1/suhubdyd/models/gans/imagesgen/', 
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', 
                        help='Directory name to save training logs')
    parser.add_argument('--visdom', type=str, default='0', 
                        help='Display results on Visdom')
    parser.add_argument('--visdom-sever', type=str, default='http://suhubdy.com', 
                        help='Visdom Server to display results')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--lambda_grad_penalty', type=float, default=0.25)
    parser.add_argument('--n_critic', type=float, default=5)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--distributed', type=bool, default=True, 
                        help='Distributed sampling')
    parser.add_argument('--nThreads', '-j', default=2, type=int, metavar='N',
                        help='number of data loading threads (default: 2)')
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
