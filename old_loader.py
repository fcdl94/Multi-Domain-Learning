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
    model1 = network.piggyback_net(main.d_names.values(), "models/tmpvgg_checkpoint.pth")
    model2 = network.piggyback_net(main.d_names.values(), "models/tmpufc_checkpoint.pth")
    #model = network.wide_resnet(main.d_names.values(), args.pretrained, 0, args.old)

    d_names_1 = dict(model1.named_parameters())
    d_names_2 = dict(model2.named_parameters())
    #old_d_names = torch.load(args.pretrained)['state_dict']

    for n1 in d_names_1:
        if not torch.equal(d_names_1[n1], d_names_2[n1]):
            if "8" not in n1 and "4" not in n1:
                print(n1 + " is different")


    import numpy as np
    model1.eval()
    model2.eval()

    accuracy1 = np.empty(100)
    accuracy2 = np.empty(100)

    for i in range(0, 100):
        accuracy1[i] = t.test(model1, "vgg-flowers")

    for i in range(0, 100):
        accuracy2[i] = t.test(model2, "vgg-flowers")

    mean1 = accuracy1.mean()
    mean2 = accuracy2.mean()

    print("accuracy 1 mean= " + str(mean1)) #75.64
    print("accuracy 2 mean= " + str(mean2)) #75.43, 74.03

