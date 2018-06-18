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


if __name__ == '__main__':
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
    parser.add_argument('--epochs', type=int, default=60,
                        help='How many epochs to use for training')
    parser.add_argument('--old', type=int, default=0,
                        help='Whether it is old or new model.')
    parser.add_argument('--test', type=int, default=0,
                        help='Whether it is only to test or also to train.')
    parser.add_argument('--output', type=str, default=None,
                        help='Whether it is only to test or also to train.')
    parser.add_argument('--visdom', type=str, default="training",
                        help='Select the visdom environment.')

    args = parser.parse_args()

    bn = args.bn

    if args.net == 'resnet':
        model = net.wide_resnet(d_names.values(), args.pretrained, args.frozen, args.old)
    elif args.net == 'piggyback':
        model = net.piggyback_net(d_names.values(), args.pretrained, args.old)
    else:
        raise ValueError(
            'The network you specified is not included in our model.'
            'Please specify one of the following: resnet, piggyback or quantized.')

    accuracy = 0
    if args.dataset in d_names.keys():
        if not args.test:
            accuracy = training.train(model, args.dataset, args.prefix, mirror=args.mirror, bn=args.bn,
                                      scaling=args.scaling, epochs=args.epochs, visdom=args.visdom)
        else:
            accuracy = training.test(model, args.dataset)

    if args.output:
        file = open(args.output, "a")
        file.write("Dataset: " + args.dataset + " " + "Top1= " + str(accuracy) + "\n")
        file.close()
