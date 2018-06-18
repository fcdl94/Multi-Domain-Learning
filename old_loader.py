import old_networks as net
import network
import argparse
import main
import training as t
import torch


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Masked model for VDA challenge')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Whether to use a pretrained model.')
    parser.add_argument('--old', type=int, default=1,
                        help='Whether it is old or new model.')

    args = parser.parse_args()

    # Initialize the specified architecture.
    # old_model = net.resnet28(args.pretrained, 1000)
    model = network.wide_resnet(main.d_names.values(), args.pretrained, 0, args.old)

    # d_names = dict(old_model.named_parameters()).keys()
    # old_d_names = torch.load(args.pretrained)['state_dict']

#    for name in d_names:
#        if name not in old_d_names:
#            print("NEW: "+name)

    t.test(model, "imagenet12")
