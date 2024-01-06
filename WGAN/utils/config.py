import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Pytorch implementation of GAN models.")

    parser.add_argument('--model', type=str, default='WGAN-GP', choices=['GAN', 'DCGAN', 'WGAN-CP', 'WGAN-GP'])
    parser.add_argument('--is_train', type=str, default='False')
    parser.add_argument('--dataroot', required=False, help='path to dataset')
    parser.add_argument('--dataset', type=str, default='lungsCT_three_channels', choices=['mnist', 'fashion-mnist', 'cifar', 'stl10', 'CTDATA', 'CTDATA_three_channels','lungsCT_three_channels'],
                            help='The name of dataset')
    parser.add_argument('--download', type=str, default='False')
    parser.add_argument('--epochs', type=int, default=500, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=256, help='The size of batch')
    parser.add_argument('--cuda',  type=str, default='True', help='Availability of cuda')
    # parser.add_argument('--channels', type=int, default=3, help='The channels of Img')

    parser.add_argument('--load_D', type=str, default='LIDC_model\\1\\1\\discriminator.pkl', help='Path for loading Discriminator network')
    parser.add_argument('--load_G', type=str, default='LIDC_model\\1\\1\\generator.pkl', help='Path for loading Generator network')
    parser.add_argument('--number', type=int, default=200, help='The number of epochs to run')
    parser.add_argument('--save_path', type=str,  default='pytorch-wgan-master\CommonData\\1')
    parser.add_argument('--generator_iters', type=int, default=3001, help='The number of iterations for generator in WGAN model.')
    return check_args(parser.parse_args())


# Checking arguments
def check_args(args):
    # --epoch
    try:
        assert args.epochs >= 1
    except:
        print('Number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('Batch size must be larger than or equal to one')

    if args.dataset == 'cifar' or args.dataset == 'stl10' or args.dataset == 'CTDATA_three_channels' or args.dataset == 'lungsCT_three_channels':
        args.channels = 3
    else:
        args.channels = 1
    args.cuda = True if args.cuda == 'True' else False
    return args
