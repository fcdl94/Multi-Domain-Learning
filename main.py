import network as net
import torch.nn as nn
from custom_layers import MaskedConv2d
import training
import argparse

d_names = {'imagenet12': 1000,
           'aircraft': 100,
           'cifar100': 100,
           'daimlerpedcls': 2,
           'dtd': 47,
           'gtsrb': 43,
           'omniglot': 1623,
           'svhn': 10,
           'ucf101': 101,
           'vgg-flowers': 102}

parser = argparse.ArgumentParser(description='Masked model for VDA challenge')
parser.add_argument('--net', type=str, default='piggyback',
                    help='Network that we want to train. Possible values: resnet, piggyback, quantized.')
parser.add_argument('--pretrained', type=str, default=None,
                    help='Whether to use a pretrained model.')
parser.add_argument('--dataset', type=str, default='vgg-flowers',
                    help='Dataset on which the model must be trained.')
parser.add_argument('--prefix', type=str, default='./models',
                    help='Where to store the checkpoints')
parser.add_argument('--bn', type=int, default=0,
                    help='Whether to tune Batch-Normalization layers')
parser.add_argument('--mirror', type=int, default=1,
                    help='Whether to apply mirroring as data-augmentation.')
parser.add_argument('--scaling', type=int, default=1,
                    help='Whether to apply scaling as data-augmentation.')
parser.add_argument('--frozen', type=int, default=0,
                    help='Whether to use feature extractor (0 - DEF) or fine tuning (1).')

args = parser.parse_args()

bn = args.bn

if args.net == 'resnet':
    model = net.wide_resnet(d_names.values(), args.pretrained, args.frozen)
elif args.net == 'piggyback':
    model = net.piggyback_net(d_names.values(), args.pretrained)
else:
    raise ValueError(
        'The network you specified is not included in our model.'
        'Please specify one of the following: resnet, piggyback or quantized.')

#if args.dataset in d_names.keys():
    #training.train(model, args.dataset, args.prefix, mirror=args.mirror,
    #               bn=args.bn, scaling=args.scaling)

print(model)
